import os
import sys
import logging
import json
import torch
# Force online mode for Hugging Face
os.environ["HF_HUB_OFFLINE"] = "0"
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("MAVIS_CAST_SDXL")

def load_config():
    return {
        "model_id": "segmind/SSD-1B", # SDXL Mini (Distilled, 50% smaller)
        "char_dir": "output/images/characters",
        "chars_file": "output/characters.json",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "steps": 20,           # More steps → sharper, more consistent reference images
        "guidance_scale": 7.5, # Stronger adherence to prompt
        "height": 768,
        "width": 768
    }

def generate_cast():
    cfg = load_config()
    
    if not os.path.exists(cfg["chars_file"]):
        logger.error(f"Characters file not found at {cfg['chars_file']}")
        return

    # Load SDXL Mini (SSD-1B)
    logger.info(f"Loading SDXL Mini (SSD-1B): {cfg['model_id']}...")
    try:
        # FIX: Load VAE in FP32 separately to avoid black/fried images & type mismatch errors
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16 # Use the fixed FP16 VAE which is faster & doesn't break
        )
        

        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg["model_id"], 
            vae=vae, # Inject fixed VAE
            torch_dtype=torch.float16 if cfg["device"] == "cuda" else torch.float32,
            variant="fp16",
            use_safetensors=True
        )
        

        if cfg["device"] == "cuda":
            logger.info("  Enabling Memory Optimizations (CPU Offload, Tiling)...")
            # Aggressive Memory Optimization
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
            
            # Try xformers for speed boost
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("  Enabled xformers memory efficient attention.")
            except Exception:
                logger.warning("  xformers not installed/available. Using attention slicing.")
                pipe.enable_attention_slicing()
        else:
            logger.warning("  Running on CPU. Expect slow generation.")
            pipe.to(cfg["device"])


        # Optimize: Switch to DPMSolverMultistepScheduler (Faster convergence)
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, 
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++" 
        )

    except Exception as e:
        logger.error(f"Failed to load specific optimizations or model: {e}")
        return

    # Load Registry
    with open(cfg["chars_file"], "r", encoding="utf-8") as f:
        registry = json.load(f)

    if not os.path.exists(cfg["char_dir"]): os.makedirs(cfg["char_dir"])

    logger.info("--- GENERATING 4-VIEW REFERENCE SHEETS (SDXL) ---")


    for char_name, details in registry.items():
        logger.info(f"> Processing Character: {char_name}")
        
        # --- SKIP CHECK: Skip character entirely if ALL views already exist ---
        char_subdir_check = os.path.join(cfg["char_dir"], char_name.replace(" ", "_"))
        all_views = [
            "waist_front", "waist_back", "waist_side", "full_front",
            "close_up_face", "seated_front", "three_quarter", "full_back"
        ]
        existing_views = [
            v for v in all_views
            if os.path.exists(os.path.join(char_subdir_check, f"{char_name.lower()}_{v}.png"))
               and os.path.getsize(os.path.join(char_subdir_check, f"{char_name.lower()}_{v}.png")) > 0
        ]
        if len(existing_views) == len(all_views):
            logger.info(f"  Skipping {char_name}: all {len(all_views)} view images already exist.")
            continue
        
        # DNA Construction
        vis = details.get("visual_details", {})
        physical = vis.get("physical", "defined features") if isinstance(vis, dict) else str(vis)
        outfit = details.get("clothing_style", "casual clothes")
        
        # Create character specific folder
        char_subdir = os.path.join(cfg["char_dir"], char_name.replace(" ", "_"))
        if not os.path.exists(char_subdir):
            os.makedirs(char_subdir)
            
        # Reordered prompt: Outfit BEFORE Physical to anchor clothing color and reduce bleeding from other features (like "green eyes")
        # Added quality tokens
        quality_tokens = "extremely detailed, high resolution, photorealistic, sharp focus"
        
        neg_prompt = (
            "drawing, painting, cartoon, 3d render, low quality, bad anatomy, deformed, mutated, text, watermark"
        )

        # Updated strict prompt instructions (Framing without style bias)
        views = {
            "waist_front":    "medium shot, waist up, front view, looking at camera",
            "waist_back":     "medium shot, waist up, back view, from behind, facing away",
            "waist_side":     "medium shot, waist up, profile view, looking side",
            "full_front":     "full body shot, standing, head to toe, front view",
            "close_up_face":  "close up portrait, headshot, detailed face and eyes, looking at camera",
            "seated_front":   "medium wide shot, sitting on chair, relaxed, front view",
            "three_quarter":  "medium shot, waist up, three quarter view, looking slightly away",
            "full_back":      "full body shot, standing, head to toe, back view, facing away",
        }

        # Base seed for the character
        seed = abs(hash(char_name)) % (2**32)
        logger.info(f"  Using base seed: {seed}")
        
        # Base Negative Prompt
        base_neg = neg_prompt + ", close up, cropped, missing limbs"

        for view_key, view_prompt_prefix in views.items():
            view_filename = f"{char_name.lower()}_{view_key}.png"
            filepath = os.path.join(char_subdir, view_filename)
            
            # Dynamic Negative Prompting
            current_neg_prompt = base_neg
            if "waist" in view_key or "three_quarter" in view_key:
                # Force cut at waist by forbidding lower body features
                current_neg_prompt += ", full body, legs, feet, shoes, boots, knees, wide shot, long shot, far away"
            
            if "back" in view_key:
                # Strictly forbid face features for back view
                current_neg_prompt += ", face, eyes, mouth, nose, front view, looking at camera"

            if "close_up" in view_key:
                # Close up face: forbid body below shoulders
                current_neg_prompt += ", full body, torso, waist, legs, feet, wide shot"

            if "seated" in view_key:
                # Seated: forbid standing
                current_neg_prompt += ", standing, full standing body, far away"

            # Reset generator seed for EACH view to ensure identical starting noise (better consistency)
            generator = torch.Generator(device=cfg["device"]).manual_seed(seed)
            
            # Construct the full prompt - framing FIRST
            # Prompt Structure: [View], [Character], [Outfit], [Physical], [Style]
            full_prompt = f"{view_prompt_prefix}, {char_name}, wearing {outfit}, {physical}, {quality_tokens}, neutral studio background"
            
            # --- SKIP CHECK: Skip individual view if it already exists ---
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                logger.info(f"  Skipping {view_key} (already exists): {view_filename}")
                continue

            logger.info(f"  Generating {view_key}...")
            
            try:
                image = pipe(
                    prompt=full_prompt,
                    negative_prompt=current_neg_prompt, # Use improved neg prompt
                    height=cfg["height"],
                    width=cfg["width"],
                    num_inference_steps=20, # Increased steps for better quality
                    guidance_scale=cfg["guidance_scale"],
                    generator=generator
                ).images[0]
                
                image.save(filepath)
            except Exception as e:
                logger.error(f"  Failed {view_key}: {e}")
                
    logger.info("Cast Generation Complete.")

if __name__ == "__main__":
    generate_cast()
