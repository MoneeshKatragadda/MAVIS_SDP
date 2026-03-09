import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import json
import torch
import logging
import gc
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

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("MAVIS_IMG")


# ------------------------------------------------
# CONFIG
# ------------------------------------------------
def load_config():
    return {
        "model_id": "Lykon/dreamshaper-8",
        "ip_adapter_repo": "h94/IP-Adapter",
        "ip_adapter_subfolder": "models",
        "ip_adapter_weight": "ip-adapter-plus_sd15.bin",
        "lcm_lora_id": "latent-consistency/lcm-lora-sdv1-5",

        "output_dir": "output/images",
        "char_dir": "output/images/characters",
        "events_file": "output/events.json",
        "chars_file": "output/characters.json",

        "device": "cuda" if torch.cuda.is_available() else "cpu",

        "steps": 15,
        "guidance_scale": 2.0,

        "height": 512,
        "width": 512,

        "ip_adapter_scale": 0.30,
    }


# ------------------------------------------------
# MEMORY
# ------------------------------------------------
def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------
# PIPELINE
# ------------------------------------------------
def load_pipeline(cfg, use_ip_adapter=True):

    logger.info(f"Loading pipeline {cfg['model_id']}")

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16 if cfg["device"] == "cuda" else torch.float32,
        use_safetensors=True,
        safety_checker=None
    )

    if cfg["device"] == "cuda":
        pipe.to("cuda")
        pipe.unet.to(memory_format=torch.channels_last)

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    if use_ip_adapter:
        try:
            logger.info("Loading IP Adapter")
            pipe.load_ip_adapter(
                cfg["ip_adapter_repo"],
                subfolder=cfg["ip_adapter_subfolder"],
                weight_name=cfg["ip_adapter_weight"],
            )

            pipe.set_ip_adapter_scale(cfg["ip_adapter_scale"])

        except Exception as e:
            logger.warning(f"IP adapter failed {e}")
            use_ip_adapter = False

    try:

        logger.info("Loading LCM LoRA")

        pipe.load_lora_weights(cfg["lcm_lora_id"])

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    except:

        logger.warning("LCM failed, switching to DPM++")

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )

        cfg["steps"] = 25
        cfg["guidance_scale"] = 7.5

    return pipe, use_ip_adapter


# ------------------------------------------------
# CHARACTER DNA
# ------------------------------------------------
def load_dna_map(cfg):

    if not os.path.exists(cfg["chars_file"]):
        return {}

    with open(cfg["chars_file"], "r", encoding="utf-8") as f:
        registry = json.load(f)

    dna = {}

    for char_name, details in registry.items():

        outfit = details.get("clothing_style", "")

        physical = details.get("visual_details", {}).get("physical", "")

        dna[char_name] = f"wearing {outfit}, {physical}".strip(", ")

    return dna


# ------------------------------------------------
# CHARACTER IMAGES
# ------------------------------------------------
def get_character_images(dna_map, char_dir):

    char_images = {}

    for char_name in dna_map.keys():

        safe = char_name.replace(" ", "_")

        char_path = os.path.join(char_dir, safe)

        if not os.path.exists(char_path):
            continue

        views = {}

        for f in os.listdir(char_path):

            if f.endswith(".png") or f.endswith(".jpg"):

                path = os.path.join(char_path, f)

                if os.path.getsize(path) == 0:
                    continue

                try:
                    views[f] = load_image(path)
                except:
                    pass

        if views:
            char_images[char_name] = views

    return char_images


# ------------------------------------------------
# IDENTITY SEED
# ------------------------------------------------
def build_seed(shot_id, char_name=None):

    scene_seed = abs(hash(shot_id)) % (2**32)

    if char_name:

        char_seed = abs(hash(char_name)) % (2**32)

        return (scene_seed ^ char_seed) % (2**32)

    return scene_seed


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def generate_images(events_path="output/events.json"):

    cfg = load_config()

    os.makedirs(cfg["output_dir"], exist_ok=True)

    os.makedirs(cfg["char_dir"], exist_ok=True)

    if generate_cast:
        logger.info("Ensuring character images exist")
        generate_cast()

    with open(events_path, "r", encoding="utf-8") as f:

        data = json.load(f)

        timeline = data["timeline"]

    dna_map = load_dna_map(cfg)

    char_images = get_character_images(dna_map, cfg["char_dir"])

    use_ip = len(char_images) > 0

    pipe, ip_active = load_pipeline(cfg, use_ip)

    for scene in timeline:

        for beat in scene["beats"]:

            if "visual_prompt" not in beat:
                continue

            shot_id = beat.get("sub_scene_id", "scene")

            shot_type = beat.get("shot_type", "MEDIUM")

            filepath = os.path.join(cfg["output_dir"], f"{shot_id}.png")

            if os.path.exists(filepath):
                continue

            base_prompt = beat["visual_prompt"]

            present_chars = []

            char_refs = []

            active_dna = []

            for name, dna in dna_map.items():

                if name in base_prompt:

                    present_chars.append(name)

                    # We don't inline replacing name with DNA to keep action words prominent
                    active_dna.append(f"{name} is {dna}")

                    if name in char_images:
                        char_refs.extend(list(char_images[name].values()))

            # Use prompt weighting: strongly emphasize action/bg, isolate DNA
            prompt = f"({shot_type.lower()} shot of {base_prompt.strip()}:1.3)."
            
            if active_dna:
                prompt += f" Characters description: {', '.join(active_dna)}."

            prompt += " cinematic lighting, ultra detailed, masterpiece, vivid colors, detailed background"

            negative = "blurry, low quality, deformed, mutated, missing objects, ignoring prompt, plain background, empty background, bad anatomy, cartoon, illustration, 3d render, monochrome"

            logger.info(f"Generating {shot_id}")

            seed = build_seed(shot_id, present_chars[0] if present_chars else None)

            generator = torch.Generator(device=cfg["device"]).manual_seed(seed)

            kwargs = dict(
                prompt=prompt,
                negative_prompt=negative,
                height=cfg["height"],
                width=cfg["width"],
                num_inference_steps=cfg["steps"],
                guidance_scale=cfg["guidance_scale"],
                generator=generator,
            )

            if ip_active:

                # Global IP-Adapter overwrites the image with one concept.
                # If there are multiple characters, we disable it and rely on the text DNA to draw a group.
                if shot_type in ["CLOSE_UP", "MEDIUM"] and len(present_chars) == 1 and char_refs:

                    pipe.set_ip_adapter_scale(cfg["ip_adapter_scale"])

                    # Wrap in a list because we have 1 IP-Adapter but multiple images for it
                    kwargs["ip_adapter_image"] = [char_refs]

                else:

                    pipe.set_ip_adapter_scale(0.0)

                    kwargs["ip_adapter_image"] = [Image.new("RGB", (224,224))]

            image = pipe(**kwargs).images[0]

            image.save(filepath)

            logger.info(f"Saved {shot_id}")

    flush_memory()

    logger.info("Generation complete")


if __name__ == "__main__":

    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "output/events.json"

    generate_images(path)