# -*- coding: utf-8 -*-
from llama_cpp import Llama
import re
import logging

logger = logging.getLogger("MAVIS_LLM")

class LLMReasoner:
    def __init__(self, config):
        m = config["models"]
        self.llm = Llama(
            model_path=m["llm_model_path"],
            n_ctx=m.get("llm_context_window", 2048),
            n_gpu_layers=min(m.get("llm_gpu_layers", 12), 20),
            verbose=False
        )
        
        self.character_visuals = {} # Store consistency profiles
        self.character_metadata = {} # Store gender and style

        self.VALID_EMOTIONS = {
            "suspicion", "paranoia", "dread", "calculating", "cynical", 
            "defensive", "threatening", "desperate", "sarcastic", "cold",
            "relieved", "urgent", "intimidating", "neutral", "anger",
            "fear", "annoyance", "curiosity", "joy", "sadness", "amusement",
            "disapproval"
        }

        self.FORBIDDEN_LABELS = {
            "intensity", "emotion", "score", "value", "label", "tone", 
            "for", "refers", "implies", "suggests", "phrase", "word", 
            "meaning", "context", "indicates", "shows", "reflects", "is", "a", "the"
        }

    def _parse_key_value(self, text, key):
        # Improved Regex to stop at newlines or common delimiters
        pattern = rf"{key}[:\-\s]+([a-zA-Z0-9_\.]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _clean_visual_output(self, text):
        """
        Removes hallucinations like 'Output:', 'Input:', brackets, and meta-text.
        """
        # Remove common LLM prefixes
        text = re.sub(r"^(Output|Response|Visual|Description|Input|Stable Diffusion)[:\-\s]*", "", text, flags=re.IGNORECASE)
        
        # Remove anything in square brackets [Mood] or (Context)
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?\)", "", text)
        
        # Remove quotes
        text = text.replace('"', '').replace("'", "")
        
        # Remove trailing "lighting" if it dangles (e.g., "Julian stood up, lighting.")
        text = re.sub(r",\s*lighting\.?$", "", text, flags=re.IGNORECASE)
        
        return text.strip()

    def analyze_cast_profiles(self, story_text, characters):
        char_list = ", ".join(characters)
        intro_text = story_text[:1000].replace("\n", " ")
        
        prompt = f"""Instruction: Assign a 3-word Noir Archetype to each character.
Story: {intro_text}...
Characters: {char_list}

Format:
Name: Adjective, Adjective, Adjective

Response:
"""
        out = self.llm(prompt, max_tokens=256, stop=["Instruction:", "Story:"])
        raw_text = out["choices"][0]["text"]
        
        profiles = {}
        for line in raw_text.split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                name = parts[0].strip()
                arch = parts[1].strip()
                for c in characters:
                    if c in name and len(arch) > 3:
                        profiles[c] = arch
                        break
        
        defaults = ["Stoic", "Nervous", "Femme Fatale", "Enforcer"]
        for i, c in enumerate(characters):
            if c not in profiles:
                profiles[c] = f"Noir Character, {defaults[i % len(defaults)]}"
                
        return profiles

    def analyze_cast_visuals(self, story_text, characters):
        """Generates rigid Visual DNA for characters (Immutable Physical + Signature Outfit)."""
        char_list = ", ".join(characters)
        intro_text = story_text[:1500].replace("\n", " ")
        
        prompt = f"""Instruction: Create rigid Visual DNA for consistent video generation.
Story Context: {intro_text}...
Characters: {char_list}

Rules:
1. Physical Traits: face, ethnicity, hair, age, build. (Immutable)
2. Signature Outfit: The specific clothing they wear throughout the scene.
3. Be descriptive but concise.

Format:
Name | Gender | Physical Traits | Signature Outfit

Response:
"""
        out = self.llm(prompt, max_tokens=400, stop=["Instruction:", "Story:"], temperature=0.1)
        raw_text = out["choices"][0]["text"]
        
        visuals = {} # name -> {physical, outfit}
        metadata = {}

        for line in raw_text.split("\n"):
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    name = parts[0]
                    gender = parts[1]
                    physical = parts[2]
                    outfit = parts[3]
                    
                    if not physical.strip(): continue
                    
                    # Fuzzy match name
                    for c in characters:
                        if c in name:
                            visuals[c] = {
                                "physical": physical,
                                "outfit": outfit,
                                "full_desc": f"{physical}, wearing {outfit}"
                            }
                            metadata[c] = {"gender": gender, "style": outfit}
                            break
        
        # Defaults - User Request: All 20s, No Beards
        default_looks = [
            ("Male", "short black hair, blue eyes, clean shaven, 20s", "Brown Trench Coat"),
            ("Female", "red wavy hair, 20s", "White Blouse and High-Waisted Grey Trousers"),
            ("Male", "clean shaven face, sharp jawline, 20s", "Grey Suit"),
            ("Male", "buzz cut, clean shaven, 20s", "Black Leather Jacket")
        ]
        
        for i, c in enumerate(characters):
            if c not in visuals:
                gender, phys, outf = default_looks[i % len(default_looks)]
                visuals[c] = {
                    "physical": phys,
                    "outfit": outf,
                    "full_desc": f"{phys}, wearing {outf}"
                }
                metadata[c] = {"gender": gender, "style": outf}
        
        self.character_visuals = visuals
        self.character_metadata = metadata
        logger.info(f"Generated Visual DNA: {visuals}")
        return visuals

    def refine_dialogue_emotion(self, speaker, text, archetype, context_window, base_emotion):
        prompt = f"""Instruction: Act as a Noir Film Director. Define the EMOTION and DELIVERY.
Character: {speaker} ({archetype})
Line: "{text}"
Context: {context_window}
Surface Emotion: {base_emotion}

Task: Choose a specific, evocative adjective for how this line is spoken.
Examples: Sarcastic, Fearful, Angry, Hesitant, Flirtatious, Cold, Desperate, Commanding, Shaky, Breathless.
Avoid "Neutral" if possible.

Format:
EMOTION: [Adjective]
INTENSITY: [0.1 - 1.0]

Response:
"""
        out = self.llm(prompt, max_tokens=60, stop=["Instruction:", "Line:"], temperature=0.1)
        raw_text = out["choices"][0]["text"]
        
        pred_label = self._parse_key_value(raw_text, "EMOTION")
        pred_score = self._parse_key_value(raw_text, "INTENSITY")
        
        final_label = base_emotion
        final_score = 0.8

        if pred_label:
            clean_label = pred_label.lower().strip()
            # Allow any reasonable word, just filter garbage
            if clean_label not in self.FORBIDDEN_LABELS and len(clean_label) > 2:
                final_label = clean_label

        if pred_score:
            try:
                val = float(pred_score)
                final_score = min(max(val, 0.1), 1.0)
            except ValueError:
                final_score = 0.8

        return {"label": final_label, "intensity": final_score}

    def analyze_narration_tone(self, text, context_window):
        prompt = f"""Instruction: Act as a Noir Film Director. Define the ATMOSPHERE and TONE for this narration.
Context: {context_window}
Narrator Line: "{text}"

Task: Choose a specific, evocative adjective to describe the narration style.
Examples: Ominous, Melancholic, Urgent, Detached, Cynical, Whispering, Harsh, Reflective, Suspenseful, Gloomy.
Avoid "Neutral".

Format:
TONE: [Adjective]
INTENSITY: [0.1 - 1.0]

Response:
"""
        out = self.llm(prompt, max_tokens=60, stop=["Instruction:", "Narrator Line:"], temperature=0.1)
        raw_text = out["choices"][0]["text"]
        
        tone = "neutral"
        intensity = 0.5
        
        pred_tone = self._parse_key_value(raw_text, "TONE")
        pred_int = self._parse_key_value(raw_text, "INTENSITY")
        
        if pred_tone:
            clean_tone = pred_tone.lower().strip()
            if clean_tone not in self.FORBIDDEN_LABELS and len(clean_tone) > 2:
                tone = clean_tone
                
        if pred_int:
             try:
                val = float(pred_int)
                intensity = min(max(val, 0.1), 1.0)
             except ValueError:
                intensity = 0.5
                
        return {"label": tone, "intensity": intensity}



    def generate_visual_prompt_v2(self, beat_data, location, active_cast):
        """
        New handler for strict visual prompts.
        beat_data: dict containing 'type', 'text', 'speaker', 'emotion'
        location: str current location name/desc
        active_cast: list of char names present in scene
        """
        b_type = beat_data['type']
        text = beat_data['text']
        emotion = beat_data.get('emotion', {}).get('label', 'neutral')
        
        # Resolve Characters
        # For dialogue, focus on speaker
        if b_type == 'dialogue':
            speaker = beat_data.get('speaker', 'Unknown')
            
            # Robust lookup with Visual DNA
            vis_data = self.character_visuals.get(speaker, {})
            if isinstance(vis_data, dict):
                char_dna = vis_data.get("full_desc", f"{speaker}, noir character")
            else:
                char_dna = str(vis_data) # Fallback

            # "High quality color cinematic shot of [Speaker], [Emotion], [Action]..."
            # Precise, consistent structure + Speaking indicator
            prompt = f"High quality color cinematic shot of {speaker}, {emotion} expression, speaking with mouth slightly open, blurry background of {location} with dimly lit lights"
            return prompt

        # For Narration
        else:
            # 1. Identify which characters are actually in this specific narration line
            relevant_chars = []
            lower_text = text.lower()
            for c in active_cast:
                # Case-insensitive check + basic pronoun check (risky, so just sticking to names for now)
                if c.lower() in lower_text: 
                     relevant_chars.append(c)
            
            # If no specific char found but scene has them, default to all active cast
            # This ensures we don't lose context in vague sentences
            if not relevant_chars:
                 relevant_chars = active_cast

            # 2. Build Context with Visual DNA
            vis_context = []
            for c in relevant_chars:
                vis = self.character_visuals.get(c, {})
                # Handle both dict and string cases matching previous logic
                if isinstance(vis, dict):
                    desc = vis.get("full_desc", "noir figure")
                else:
                    desc = str(vis)
                vis_context.append(f"{c} is {desc}")
            
            vis_block = ". ".join(vis_context)

            # Simplified Prompt for Phi-2 (Completion style)
            # Detailed Extraction for Phi-2
            # We want: [Action + Objects], blurry background...
            
            prompt = f"""Task: Extract visual subject, count, action and objects for an image prompt.
Input: "Lena rolled her eyes, her fingers dancing across the screen of a tablet."
Output: Lena rolling eyes, fingers tapping tablet screen
Input: "Inside, three people sat at the corner booth."
Output: Three people sitting at corner booth
Input: "Silas took a slow sip of his black coffee."
Output: Silas sipping black coffee
Input: "{text}"
Output:"""
            
            # 1. Get Action
            out = self.llm(prompt, max_tokens=45, stop=["\n", "Input:"], temperature=0.1)
            action = out["choices"][0]["text"].strip()
            if len(action) < 5: action = text # Fallback
            
            # 2. Build Prompt (CLEAN - No Physical DNA)
            # format: "High quality color cinematic shot of [Action], blurry background of [Location] with dimly lit lights"
            
            gen = f"High quality color cinematic shot of {action}, blurry background of {location} with dimly lit lights"
            return gen

    def analyze_beat_production(self, beat_data):
        """
        Determines BGM and SFX for a specific beat.
        BGM: specific to the scene emotion/context.
        SFX: strictly derived from text actions.
        """
        text = beat_data['text']
        b_type = beat_data['type']
        emotion_label = beat_data.get('emotion', {}).get('label', 'neutral')
        
        # Defaults
        bgm_style = "Silence"
        bgm_vol = 0.0
        sfx_list = []

        if b_type == 'narration':
            # --- BGM LOGIC ---
            # Map emotions to Noir BGM styles
            noir_scores = {
                "neutral": "Low Hum / City Ambience",
                "ominous": "Dark Suspense Drone",
                "fear": "Tense Industrial Pulse",
                "desperate": "Rapid Heartbeat / High Strings",
                "sadness": "Melancholic Saxophone & Rain",
                "joy": "Light Jazz", # Rare in Noir
                "anger": "Aggressive Bass Drone",
                "approval": "Smooth Jazz",
                "curiosity": "Mystery Piano"
            }
            
            # Default based on emotion
            bgm_style = noir_scores.get(emotion_label, "Suspenseful Drone")
            if "rain" in text.lower() or "storm" in text.lower():
                bgm_style = "Rainy Noir Ambience"

            import random
            bgm_vol = round(random.uniform(0.35, 0.40), 2)

            # --- SFX LOGIC ---
            # Strict few-shot prompt for Phi-2
            prompt = f"""Task: List audible SFX.
Input: "The rain drummed on the roof."
Output: SFX: Rain, Drumming
Input: "He smelled the coffee."
Output: SFX: None
Input: "He sipped the tea."
Output: SFX: Sipping
Input: "Fingers dancing on the screen."
Output: SFX: Tapping
Input: "Julian hissed."
Output: SFX: Hissing
Input: "{text}"
Output:"""
            
            out = self.llm(prompt, max_tokens=20, stop=["Input:", "\n"], temperature=0.1)
            raw = out["choices"][0]["text"].strip()
            
            # Parse SFX
            if "SFX:" in raw:
                try:
                    raw_sfx = raw.split("SFX:")[1].split("\n")[0].strip()
                    if raw_sfx.lower() != "none":
                        items = [x.strip() for x in raw_sfx.split(",")]
                        for item in items:
                            name = re.sub(r"\(.*?\)", "", item).strip()
                            if name and name.lower() != "none":
                                sfx_list.append({
                                    "name": name,
                                    "timing": {"start": 0.1, "end": 0.9}
                                })
                except Exception as e:
                    logger.error(f"SFX Parsing Error: {e}")

        return {
            "bgm": {
                "style": bgm_style,
                "volume": bgm_vol
            },
            "sfx": sfx_list
        }

    def determine_shot_type(self, beat_data):
        """
        Determines the cinematic shot type.
        Returns: CLOSE_UP, MEDIUM, WIDE, ESTABLISHING, or NONE.
        """
        text = beat_data['text']
        b_type = beat_data['type']
        
        prompt = f"""Instruction: act as a Cinematographer. Choose the best Camera Shot.
Scene Line: "{text}"
Type: {b_type}

Options:
1. CLOSE_UP (Emotions, face details, crucial dialogue)
2. MEDIUM (Actions, interactions, waist-up)
3. WIDE (Movement, full body, multiple characters)
4. ESTABLISHING (Setting the scene, narration about location)
5. NONE (Minor beat, audio only, no visual change needed)

Format:
SHOT: [Option]

Response:"""
        
        out = self.llm(prompt, max_tokens=15, stop=["Instruction:", "Scene Line:"], temperature=0.1)
        raw = out["choices"][0]["text"]
        
        shot = "MEDIUM" # Default
        if "SHOT:" in raw:
            val = raw.split("SHOT:")[1].strip().upper()
            if "CLOSE" in val: shot = "CLOSE_UP"
            elif "WIDE" in val: shot = "WIDE"
            elif "ESTABLISH" in val: shot = "ESTABLISHING"
            elif "NONE" in val: shot = "NONE"
            elif "MEDIUM" in val: shot = "MEDIUM"
            
        return shot

    def generate_rich_registry(self, characters, profiles):
        registry = {}
        for char in characters:
            meta = self.character_metadata.get(char, {"gender": "Unknown", "style": "Noir"})
            registry[char] = {
                "voice_model_id": f"en_us_generic_{char.lower()}",
                "archetype": profiles.get(char, "Standard"),
                "gender": meta.get("gender", "Unknown"),
                "clothing_style": meta.get("style", "Noir Standard")
            }
        return registry