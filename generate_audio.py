import json
import os
import torch
import gc
import logging
import soundfile as sf
import shutil
import numpy as np
from tqdm import tqdm

# --- FORCE ONLINE MODE ---
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

try:
    # Patch missing transformers utility functions (Safe Patch)
    from transformers import pytorch_utils
    if not hasattr(pytorch_utils, "isin_mps_friendly"):
        def isin_mps_friendly(elements, test_elements):
            import torch
            return torch.isin(elements,  )
        pytorch_utils.isin_mps_friendly = isin_mps_friendly
except (ImportError, AttributeError):
    pass 

# --- PARLER TTS COMPATIBILITY PATCH ---
try:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers.generation.utils import GenerationMixin
    from transformers import LogitsProcessorList
    
    def _patched_sample(self, input_ids, logits_processor=None, stopping_criteria=None, logits_warper=None, **kwargs):
        if logits_warper is None:
            logits_warper = LogitsProcessorList()
        return GenerationMixin._sample(
            self, 
            input_ids, 
            logits_processor=logits_processor, 
            logits_warper=logits_warper, 
            stopping_criteria=stopping_criteria, 
            **kwargs
        )
    ParlerTTSForConditionalGeneration._sample = _patched_sample
except (ImportError, AttributeError):
    pass

# --- PYTORCH FIXES ---
_original_load = torch.load
def strict_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = strict_load

# --- CONFIGURATION ---
FORCE_CPU_CASTING = False 
CLEAN_SLATE = False
SKIP_EXISTING = True 

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("MAVIS_DIRECTOR")

# Constants
OUTPUT_DIR = "output/audio"
VOICES_DIR = "output/voices"
EVENTS_FILE = "output/events.json"

# --- DYNAMIC SEED GENERATION (Story-Agnostic) ---
def get_character_seed(character_name):
    """Generate deterministic seed from character name hash."""
    import hashlib
    hash_digest = hashlib.md5(character_name.encode('utf-8')).hexdigest()
    return int(hash_digest[:8], 16) % (2**31 - 1)  # Ensure positive 32-bit int

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- PHASE 1: CASTING (Parler TTS) ---
class CastingDirector:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu" if FORCE_CPU_CASTING else ("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        model_id = "parler-tts/parler-tts-mini-v1"
        logger.info(f"  > [Phase 1] Loading Parler TTS ({model_id}) on {self.device}...")
        
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def generate_master_reference(self, character_name, archetype, gender):
        import hashlib

        filename = f"{character_name}_master.wav"
        filepath = os.path.join(VOICES_DIR, filename)
        if os.path.exists(filepath):
            return

        # -------- VOICE DESCRIPTION --------
        male_desc = (
            "A male speaker with a neutral professional voice, "
            "clear articulation, steady pace, studio-quality recording."
        )

        female_desc = (
            "A female speaker with a neutral professional voice, "
            "clear articulation, steady pace, studio-quality recording."
        )

        is_female = "female" in gender.lower() or "woman" in gender.lower()
        desc = female_desc if is_female else male_desc

        full_prompt = (
            f"{desc} Speak clearly with neutral emotion "
            "and consistent pronunciation."
        )

        # -------- 4–5 SECOND REFERENCE TEXT --------
        # ~12–14 words ideal for Parler stability
        ref_text = (
            "This is my voice reference. "
            "I speak clearly, naturally, and steadily."
        )

        # -------- TOKENIZATION --------
        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_ids = self.tokenizer(ref_text, return_tensors="pt").input_ids.to(self.device)

        # -------- DETERMINISTIC SEED --------
        seed = int(hashlib.md5(character_name.encode()).hexdigest(), 16) % (2**31 - 1)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # -------- GENERATION SETTINGS --------
        with torch.no_grad():
            generation = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_ids,
                do_sample=True,
                temperature=0.4,        # Stable but natural
                top_p=0.9,
                repetition_penalty=1.1,
                max_new_tokens=220,      # CRITICAL → ~4–5 sec audio
                pad_token_id=self.model.config.pad_token_id
            )

        audio = generation.cpu().float().numpy().squeeze()

        # -------- CLEAN POST-PROCESSING --------
        # Trim trailing silence/noise
        threshold = 0.003
        idx = np.where(np.abs(audio) > threshold)[0]
        if len(idx) > 0:
            audio = audio[:idx[-1] + 1]

        # Peak normalization
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95

        sf.write(filepath, audio, self.model.config.sampling_rate)
        logger.info(f"    > Cast {character_name} [4-5s Clean Master]")


    def unload(self):
        if hasattr(self, 'model'): del self.model
        if hasattr(self, 'tokenizer'): del self.tokenizer
        flush_memory()

# --- PHASE 2: PRODUCTION (XTTS) ---
class AudioProducer:
    def __init__(self):
        self.model = None
    
    def load(self):
        logger.info("  > [Phase 2] Loading XTTS...")
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def generate_line(self, text, speaker_name, emotion_label, output_path):
        ref_path = os.path.join(VOICES_DIR, f"{speaker_name}_master.wav")
        if not os.path.exists(ref_path): 
            ref_path = os.path.join(VOICES_DIR, "Narrator_master.wav")
        
        if not os.path.exists(ref_path):
            logger.error(f"Missing Master Audio for {speaker_name}. Skipping.")
            return

        # --- TEXT NORMALIZATION & PARAMETER TUNING ---
        speed = 0.9
        text_processed = text.strip()
        emotion = emotion_label.lower() if emotion_label else "neutral"

        # 1. Expand Numbers (Fixes "4th Street" -> "Fourth Street")
        replacements = {
            "4th": "Fourth", "1st": "First", "2nd": "Second", "3rd": "Third",
            "5th": "Fifth", "6th": "Sixth", "7th": "Seventh", "8th": "Eighth"
        }
        for k, v in replacements.items():
            text_processed = text_processed.replace(k, v)

        # 2. Emotion Logic (Conservative Speed)
        if emotion in ["anger", "fury", "annoyance"]:
            speed = 0.98 # Slightly faster, but safe
            # Do NOT uppercase entire text. Just ensure exclamation.
            if not text_processed.endswith(("!", "?")): text_processed += "!" 
        elif emotion in ["fear", "nervous", "suspense"]:
            speed = 0.95
            text_processed = text_processed.replace(", ", "... ")
        elif emotion in ["sadness", "grief", "whisper"]:
            speed = 0.90
            text_processed = text_processed.lower()

        # 3. Ensure proper ending punctuation (DO NOT PAD SHORT TEXT)
        if not text_processed.endswith((".", "!", "?", "...")):
            text_processed += "."
            
        try:
            # CRITICAL: Optimized for ACCURACY (Low WER) over naturalness
            # Short text needs special handling - disable splitting to prevent hallucinations
            is_short_text = len(text_processed) < 30
            
            self.model.tts_to_file(
                text=text_processed,
                file_path=output_path,
                speaker_wav=ref_path,
                language="en",
                speed=speed,
                split_sentences=False,        # DISABLED: Prevents hallucinations
                temperature=0.3,              # Very low = maximum fidelity to text
                repetition_penalty=10.0,      # Maximum anti-stutter/hallucination
                length_penalty=1.0 if is_short_text else 1.2  # Don't encourage extra words for short text
            )
        except Exception as e:
            logger.error(f"XTTS Error {output_path}: {e}")

# --- PHASE 3: SCORING (MusicGen) ---
class MusicComposer:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu" if FORCE_CPU_CASTING else ("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        logger.info(f"  > [Phase 3] Loading MusicGen on {self.device}...")
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(self.device)

    def compose_bgm(self, style, duration, output_path):
        if not style or style.lower() in ["none", "silence"]:
            return
            
        prompt_map = {
            "Rainy Noir Ambience": "Rain drops, thunder, distant jazz saxophone.",
            "Dark Suspense Drone": "Dark ambient drone, low synth, cinematic thriller.",
            "Tense Industrial Pulse": "Industrial rhythmic pulse, metallic, tense.",
            "Melancholic Saxophone": "Slow sad saxophone solo, noir jazz, reverb.",
            "Aggressive Bass Drone": "Distorted heavy bass drone, horror texture.",
            "Smooth Jazz": "Smooth slow jazz, double bass, piano, relaxing.",
            "Mystery Piano": "Minimal mysterious piano melody, reverb."
        }
        base_prompt = prompt_map.get(style, f"A {style} music track, cinematic, high quality.")
        
        inputs = self.processor(text=[base_prompt], padding=True, return_tensors="pt").to(self.device)
        duration = min(duration, 30.0) 
        max_tokens = int(duration * 50) + 10

        with torch.no_grad():
            audio_values = self.model.generate(**inputs, max_new_tokens=max_tokens, guidance_scale=3.0)

        sampling_rate = self.model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()
        sf.write(output_path, audio_data, sampling_rate)
        logger.info(f"    > Composed '{style}'")

    def unload(self):
        del self.model
        del self.processor
        flush_memory()

# --- PHASE 4: FOLEY (MusicGen Small) ---
class FoleyArtist:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu" if FORCE_CPU_CASTING else ("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        logger.info(f"  > [Phase 4] Loading SFX Model on {self.device}...")
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(self.device)

    def generate_sfx(self, description, duration, output_path):
        if not description or description.lower() == "none": return
        
        prompt = f"Sound effect of {description}, realistic, high fidelity, no music."
        duration = min(duration, 2.0) 
        max_tokens = int(duration * 50) + 5

        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            audio_values = self.model.generate(**inputs, max_new_tokens=max_tokens, guidance_scale=3.5)

        sampling_rate = self.model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()
        sf.write(output_path, audio_data, sampling_rate)
        logger.info(f"    > Created SFX '{description}'")

    def unload(self):
        del self.model
        del self.processor
        flush_memory()

def generate_audio(events_path="output/events.json"):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(VOICES_DIR): os.makedirs(VOICES_DIR)
    
    if not os.path.exists(events_path): 
        logger.error(f"Events file not found at {events_path}")
        return

    with open(events_path, "r") as f: data = json.load(f)
    registry = data.get("character_registry", {})
    
    char_data = {}
    if os.path.exists("output/characters.json"):
        with open("output/characters.json", "r") as f:
            char_data = json.load(f)

    cast_members = data.get("global_assets", {}).get("cast", [])
    for member in cast_members:
        if member not in registry:
            c_info = char_data.get(member, {})
            registry[member] = {
                "archetype": c_info.get("archetype", "Default"), 
                "gender": c_info.get("gender", "Male")
            } 

    if "Narrator" not in registry:
        registry["Narrator"] = {"archetype": "Deep Narrative Voice", "gender": "Male"}

    # Phase 1: Casting
    caster = CastingDirector()
    caster.load()
    logger.info("--- CASTING CALL ---")
    for name, details in registry.items():
        gender = details.get("gender", "Male")
        if name in char_data:
            gender = char_data[name].get("gender", gender)
        caster.generate_master_reference(name, details.get("archetype", "Generic"), gender)
    caster.unload() 
    
    # Phase 2: Production
    producer = AudioProducer()
    producer.load()
    logger.info("--- RECORDING SESSION ---")
    
    for scene in tqdm(data.get("timeline", []), desc="Recording"):
        for beat in scene.get("beats", []):
            if beat["type"] not in ["dialogue", "narration"]: continue
            beat_id = beat.get("sub_scene_id")
            text = beat.get("text")
            emotion = beat.get("emotion", {}).get("label", "neutral")
            speaker = beat.get("speaker", "Narrator")
            if beat["type"] == "narration": speaker = "Narrator"

            out_path = os.path.join(OUTPUT_DIR, f"{beat_id}.wav")
            if SKIP_EXISTING and os.path.exists(out_path): continue
            if os.path.exists(out_path): os.remove(out_path)

            producer.generate_line(text, speaker, emotion, out_path)
    
    del producer
    flush_memory()
    
    # Phase 3: Scoring
    composer = MusicComposer()
    composer.load()
    logger.info("--- SCORING SESSION ---")
    
    bgm_registry = {} 
    for scene in data.get("timeline", []):
        for beat in scene.get("beats", []):
            prod = beat.get("production", {})
            bgm = prod.get("bgm", {})
            style = bgm.get("style", "None")
            if style and style.lower() not in ["none", "silence"]:
                key = style.replace(" ", "_").replace("/", "-").lower()
                dur = beat.get("duration", 10.0)
                if key not in bgm_registry:
                    bgm_registry[key] = {"style": style, "duration": dur}
                else:
                    bgm_registry[key]["duration"] = max(bgm_registry[key]["duration"], dur)
    
    bgm_dir = os.path.join(OUTPUT_DIR, "bgm")
    if not os.path.exists(bgm_dir): os.makedirs(bgm_dir)
    
    for key, info in tqdm(bgm_registry.items(), desc="Composing"):
        out_path = os.path.join(bgm_dir, f"{key}.wav")
        if SKIP_EXISTING and os.path.exists(out_path): continue
        if os.path.exists(out_path): os.remove(out_path)
        composer.compose_bgm(info["style"], info["duration"], out_path)
            
    composer.unload()

    # Phase 4: Foley
    foley = FoleyArtist()
    foley.load()
    logger.info("--- FOLEY SESSION ---")
    
    sfx_dir = os.path.join(OUTPUT_DIR, "sfx")
    if not os.path.exists(sfx_dir): os.makedirs(sfx_dir)
    
    sfx_registry = {}
    for scene in data.get("timeline", []):
        for beat in scene.get("beats", []):
            prod = beat.get("production", {})
            sfx_list = prod.get("sfx", [])
            for s in sfx_list:
                name = s.get("name", "None")
                if name and name.lower() != "none":
                    key = name.replace(" ", "_").replace("/", "-").lower()
                    dur = 2.0 
                    if key not in sfx_registry:
                        sfx_registry[key] = {"name": name, "duration": dur}

    for key, info in tqdm(sfx_registry.items(), desc="Foley"):
        out_path = os.path.join(sfx_dir, f"{key}.wav")
        if SKIP_EXISTING and os.path.exists(out_path): continue
        if os.path.exists(out_path): os.remove(out_path)
        foley.generate_sfx(info["name"], info["duration"], out_path)
    
    foley.unload()
    logger.info("Production Wrap!")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "output/events.json"
    generate_audio(path)