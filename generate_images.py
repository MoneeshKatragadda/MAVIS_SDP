import os

# Force online mode for Hugging Face
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import json
import torch
import logging

# Kernel auto-tuner — free speed win after first image
torch.backends.cudnn.benchmark = True

import gc
import random
import re
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    LCMScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.utils import load_image

try:
    from generate_cast import generate_cast
except ImportError:
    generate_cast = None

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("MAVIS_IMG")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config():
    return {
        "model_id": "Lykon/dreamshaper-8",
        "ip_adapter_repo": "h94/IP-Adapter",
        "ip_adapter_subfolder": "models",
        "ip_adapter_weight": "ip-adapter_sd15.bin",
        "lcm_lora_id": "latent-consistency/lcm-lora-sdv1-5",
        "output_dir": "output/images",
        "char_dir": "output/images/characters",
        "events_file": "output/events.json",
        "chars_file": "output/characters.json",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "steps": 4,
        "guidance_scale": 1.5,
        "height": 512,
        "width": 512,
        "ip_adapter_scale": 0.72,   # Raised from 0.6 for stronger identity
    }


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Pipeline loader  (SD 1.5 + IP-Adapter-Plus + LCM-LoRA)
# ---------------------------------------------------------------------------
def load_pipeline(cfg, use_ip_adapter=True):
    logger.info(f"Loading SD 1.5 Pipeline: {cfg['model_id']}  |  IP-Adapter: {use_ip_adapter}")
    try:
        # 1. SD 1.5 pipeline  (~2.5 GB fp16)
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.float16 if cfg["device"] == "cuda" else torch.float32,
            use_safetensors=True,
            safety_checker=None,
        )

        # 2. VRAM Optimizations for 4GB GPUs
        if cfg["device"] == "cuda":
            pipe.to("cuda")
            pipe.unet.to(memory_format=torch.channels_last)
            
            # Use VAE slicing/tiling instead of attention slicing. 
            # This prevents memory spikes during image decoding without breaking IP-Adapter.
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            
            # Note: We intentionally DO NOT call enable_attention_slicing() here.
            # It overwrites IP-Adapter's custom cross-attention logic, causing the 'tuple' error.

        # 3. Load IP-Adapter
        if use_ip_adapter:
            try:
                logger.info(f"  Loading IP-Adapter...")
                pipe.load_ip_adapter(
                    cfg["ip_adapter_repo"],
                    subfolder=cfg["ip_adapter_subfolder"],
                    weight_name=cfg["ip_adapter_weight"],
                )
                pipe.set_ip_adapter_scale(cfg["ip_adapter_scale"])
                logger.info("  IP-Adapter loaded successfully.")
            except Exception as e:
                logger.error(f"  IP-Adapter load FAILED: {e}")
                use_ip_adapter = False

        # 4. Load LCM-LoRA
        try:
            logger.info(f"  Loading LCM-LoRA: {cfg['lcm_lora_id']} ...")
            pipe.load_lora_weights(cfg["lcm_lora_id"])
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            logger.info("  LCM-LoRA loaded → 4 steps, CFG 1.5")
        except Exception as e:
            logger.warning(f"  LCM-LoRA failed: {e}. Falling back to DPM++ 25 steps.")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++",
            )
            cfg["steps"] = 25
            cfg["guidance_scale"] = 7.5

        logger.info(
            f"Pipeline ready — steps: {cfg['steps']}, CFG: {cfg['guidance_scale']}, "
            f"res: {cfg['width']}×{cfg['height']}, IP-Adapter: {use_ip_adapter}"
        )
        return pipe, use_ip_adapter

    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return None, False


# ---------------------------------------------------------------------------
# Character DNA helpers
# ---------------------------------------------------------------------------
def load_dna_map(cfg):
    """Build {char_name: prompt_fragment} from characters.json."""
    if not os.path.exists(cfg["chars_file"]):
        return {}
    with open(cfg["chars_file"], "r", encoding="utf-8") as f:
        registry = json.load(f)
    char_prompts = {}
    for char_name, details in registry.items():
        vis = details.get("visual_details", {})
        physical = vis.get("physical", "") if isinstance(vis, dict) else ""
        outfit = details.get("clothing_style", "casual clothes")
        # Keep DNA short: outfit is the most visually distinctive part.
        # Long DNA fragments push the combined prompt over CLIP's 77 token limit.
        short_dna = f"wearing {outfit}"
        if physical:
            # Append only the first descriptive phrase (e.g. "short black hair, blue eyes")
            first_trait = physical.split(",")[0].strip()
            short_dna += f", {first_trait}"
        char_prompts[char_name] = short_dna
    return char_prompts


def get_character_images(dna_map, char_dir):
    """
    Loads ALL available view images per character into a dict:
      { char_name: { view_key: PIL.Image } }
    """
    char_images = {}
    if not os.path.exists(char_dir):
        return char_images

    for char_name in dna_map.keys():
        safe_name = char_name.replace(" ", "_")
        char_subdir = os.path.join(char_dir, safe_name)
        views = {}

        if os.path.exists(char_subdir) and os.path.isdir(char_subdir):
            for fname in sorted(os.listdir(char_subdir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    fpath = os.path.join(char_subdir, fname)
                    if os.path.getsize(fpath) == 0:
                        continue
                    # Extract view key: "julian_waist_front.png" → "waist_front"
                    stem = os.path.splitext(fname)[0]
                    prefix = char_name.lower().replace(" ", "_") + "_"
                    view_key = stem[len(prefix):] if stem.lower().startswith(prefix) else stem
                    try:
                        views[view_key] = load_image(fpath)
                    except Exception as e:
                        logger.warning(f"  Failed to load {fname}: {e}")

        if views:
            char_images[char_name] = views
            logger.info(f"  Loaded {len(views)} views for '{char_name}': {list(views.keys())}")
        else:
            logger.warning(f"  No reference images found for '{char_name}'")

    return char_images


# Shot-type → preferred primary + secondary reference views
SHOT_VIEW_PRIORITY = {
    "CLOSE_UP":    ["close_up_face", "waist_front", "three_quarter"],
    "MEDIUM":      ["three_quarter", "waist_front", "waist_side"],
    "WIDE":        ["full_front",    "waist_front", "seated_front"],
    "ESTABLISHING":["full_front",    "waist_front"],
}


def select_reference_images(char_name, char_views, shot_type, visual_prompt):
    """
    Returns up to 2 PIL reference images for IP-Adapter, chosen based on shot type
    and contextual cues in the visual prompt (e.g. 'sitting' → prefer seated_front).
    """
    if not char_views:
        return []

    priority = list(SHOT_VIEW_PRIORITY.get(shot_type, ["waist_front", "three_quarter"]))

    # Context-aware override: if prompt mentions sitting/seated, prefer seated view
    prompt_lower = visual_prompt.lower()
    if any(kw in prompt_lower for kw in ["sitting", "seated", "sit"]):
        priority = ["seated_front"] + [v for v in priority if v != "seated_front"]

    selected = []
    for view in priority:
        if view in char_views and len(selected) < 2:
            selected.append(char_views[view])

    # Fill up to 2 from any available view if priority list exhausted
    for view, img in char_views.items():
        if len(selected) >= 2:
            break
        if img not in selected:
            selected.append(img)

    return selected



# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_images(events_path="output/events.json"):
    cfg = load_config()
    cfg["events_file"] = events_path

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["char_dir"], exist_ok=True)

    # 0. Cast generation
    if generate_cast:
        all_views = [
                    "waist_front", "waist_back", "waist_side", "full_front",
                    "close_up_face", "seated_front", "three_quarter", "full_back"
                ]
        cast_needed = False
        if os.path.exists(cfg["chars_file"]):
            with open(cfg["chars_file"], "r", encoding="utf-8") as f:
                registry = json.load(f)
            for char_name in registry.keys():
                char_subdir = os.path.join(cfg["char_dir"], char_name.replace(" ", "_"))
                missing = [v for v in all_views if not os.path.exists(os.path.join(char_subdir, f"{char_name.lower()}_{v}.png"))]
                if missing:
                    cast_needed = True
                    break
        else:
            cast_needed = True

        if cast_needed:
            logger.info("--- TRIGGERING CAST GENERATION ---")
            generate_cast()
            flush_memory()
        else:
            logger.info("--- SKIPPING CAST GENERATION: All character images already exist ---")
    else:
        logger.warning("Could not import generate_cast.")

    # 1. Load events
    try:
        with open(cfg["events_file"], "r", encoding="utf-8") as f:
            data = json.load(f)
            timeline = data.get("timeline", [])
    except FileNotFoundError:
        return logger.error(f"Events file not found at {events_path}")

    # 2. Load character DNA + reference images
    dna_map = load_dna_map(cfg)
    char_images_map = get_character_images(dna_map, cfg["char_dir"])
    should_use_ip_adapter = len(char_images_map) > 0

    # 3. Load SD 1.5 pipeline
    pipe, ip_adapter_active = load_pipeline(cfg, use_ip_adapter=should_use_ip_adapter)
    if not pipe: return

    # 4. Scene generation loop
    logger.info(f"--- STARTING SCENE GENERATION (Steps: {cfg['steps']} | CFG: {cfg['guidance_scale']}) ---")

    CINEMATIC_SHOTS = {"CLOSE_UP", "MEDIUM", "WIDE", "ESTABLISHING"}
    VIEWS = ["front view", "side view", "three quarter view", "low angle shot", "high angle shot", "cinematic angle"]
    VIEW_KEYWORDS = ["view", "angle", "shot", "profile", "close-up", "full body", "looking at"]

    for scene in timeline:
        for beat in scene.get("beats", []):
            shot_type = beat.get("shot_type", "NONE")
            if shot_type not in CINEMATIC_SHOTS or "visual_prompt" not in beat: continue

            shot_id = beat.get("sub_scene_id", "unknown")
            filepath = os.path.join(cfg["output_dir"], f"{shot_id}.png")

            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                logger.info(f"Skipping {shot_id} (already exists)")
                continue

            final_prompt = beat["visual_prompt"]
            present_char_images = []

            for name, dna in dna_map.items():
                if name in final_prompt:
                    pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
                    final_prompt = pattern.sub(f"{name} ({dna})", final_prompt)
                    # Smart view selection per shot type
                    if name in char_images_map:
                        refs = select_reference_images(
                            name, char_images_map[name], shot_type, final_prompt
                        )
                        present_char_images.extend(refs)


            shot_label = shot_type.lower().replace("_", " ")
            if shot_label not in final_prompt.lower():
                final_prompt = f"{shot_label}, {final_prompt}"

            if not any(k in final_prompt.lower() for k in VIEW_KEYWORDS) and present_char_images:
                # Add cinematic angles
                final_prompt += f", {random.choice(VIEWS + ['three quarter view', 'cinematic angle'] * 2)}"

            # --- Prompt Engineering for IP-Adapter ---
            if shot_type in {"CLOSE_UP", "MEDIUM"} and present_char_images:
                final_prompt += ", detailed face, sharp eyes"

            # Keep quality suffix short to stay within CLIP's 77-token limit
            final_prompt += ", masterpiece, cinematic lighting, ultra-detailed"
            neg_prompt = "drawing, painting, illustration, cartoon, 3d render, low quality, distorted, bad anatomy, bad hands, mutated, poorly drawn face, ugly, disfigured, text, watermark, (solid background:1.5), studio background, simple background"

            logger.info(f"Generating {shot_id} [{shot_type}] char_refs: {len(present_char_images)}")
            
            try:
                generator = torch.Generator(device="cpu").manual_seed(int(hash(shot_id) % (2 ** 32)))
                kwargs = {
                    "prompt": final_prompt,
                    "negative_prompt": neg_prompt,
                    "height": cfg["height"],
                    "width": cfg["width"],
                    "num_inference_steps": cfg["steps"],
                    "guidance_scale": cfg["guidance_scale"],
                    "generator": generator,
                }

                if ip_adapter_active:
                    if present_char_images:
                        pipe.set_ip_adapter_scale(cfg["ip_adapter_scale"])
                        # ip-adapter_sd15.bin supports ONE image only — use the best (primary) view
                        kwargs["ip_adapter_image"] = present_char_images[0]
                    else:
                        pipe.set_ip_adapter_scale(0.0)
                        kwargs["ip_adapter_image"] = Image.new("RGB", (224, 224), (0, 0, 0))

                image = pipe(**kwargs).images[0]
                image.save(filepath)
                logger.info(f"  > SAVED: {shot_id}.png")

            except Exception as e:
                logger.error(f"  Error generating {shot_id}: {e}")

    logger.info("=== Scene Generation Complete ===")
    flush_memory()

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "output/events.json"
    generate_images(path)