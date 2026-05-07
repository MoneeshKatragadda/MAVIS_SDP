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
            n_gpu_layers=m.get("llm_gpu_layers", -1),
            verbose=False
        )
        
        self.character_visuals = {} # Store consistency profiles
        self.character_metadata = {} # Store gender and style

        self.VALID_EMOTIONS = [
            "amusement", "anger", "annoyance", "anxious", "curious", 
            "disgust", "fear", "joy", "neutral", "ominous", 
            "sad", "surprise", "disappointment"
        ]

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

    def _resolve_pronouns(self, text, subject_name, secondary_name=None):
        """
        Deterministically replaces third-person pronouns with character names
        BEFORE the text reaches the LLM.

        Subject pronouns (he/she/they) → subject_name (the primary character).
        Object pronouns  (him/his/her) → secondary_name when a second character
          is known, otherwise falls back to subject_name.

        Example:
          "She sat beside him, handing one over."
          subject_name=Meera, secondary_name=Arjun
          → "Meera sat beside Arjun, handing one over."
        """
        if not subject_name:
            return text

        obj_name = secondary_name if secondary_name else subject_name

        # Map subject pronouns → primary character
        subject_patterns = {
            r'\bshe\b': subject_name,
            r'\bhe\b':  subject_name,
            r'\bthey\b': subject_name,
        }
        # Map possessive-subject pronouns → primary character's
        subject_poss_patterns = {
            r'\bhis\b':   f"{subject_name}'s",
            r'\bher\b':   f"{subject_name}'s",  # ambiguous: also used as obj pronoun below
            r'\btheir\b': f"{subject_name}'s",
        }
        # Map object pronouns → secondary character (different person)
        object_patterns = {
            r'\bhim\b':  obj_name,
            r'\bhers\b': f"{obj_name}'s",
            r'\bits\b':  f"{subject_name}'s",
        }

        result = text
        for pattern, replacement in subject_patterns.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        for pattern, replacement in subject_poss_patterns.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        for pattern, replacement in object_patterns.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

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

    # Keywords that strongly suggest a character is an animal, not a human
    _ANIMAL_KEYWORDS = {
        "dog", "cat", "horse", "bird", "rabbit", "fox", "wolf", "bear", "puppy",
        "kitten", "pup", "hound", "canine", "feline", "pet", "collar", "barking",
        "tail", "paws", "fur", "meowing", "growling", "splashing"
    }

    def _is_animal_character(self, name, story_text):
        """Check if a named character is an animal based on story context.
        Uses direct subject-verb patterns rather than a proximity window
        to avoid false positives when animal words appear near human characters.
        """
        import re as _re
        lower_story = story_text.lower()
        lower_name  = _re.escape(name.lower())

        # --- Strong positive signals: character is explicitly an animal ---
        # Pattern 1: collar is labeled with this name ("collar read 'Leo'")
        if _re.search(r'collar\s+read\s+["\']?' + lower_name, lower_story):
            return True

        # Pattern 2: species noun directly adjacent to name
        species_nouns = ["dog", "puppy", "pup", "cat", "kitten", "horse", "bird", "rabbit"]
        for sp in species_nouns:
            if _re.search(rf'\b{sp}\s+(named|called)\s+{lower_name}\b', lower_story):
                return True
            if _re.search(rf'\b{lower_name}\s+the\s+{sp}\b', lower_story):
                return True

        # Pattern 3: character is grammatical SUBJECT of a purely-animal verb
        animal_verbs = ["bark", "barked", "barking", "chased", "chase",
                        "growl", "growled", "wag", "wagged", "sniff", "sniffed",
                        "paw", "pawed", "meow", "meowed"]
        for av in animal_verbs:
            # "Leo chased", "leo barked", etc.
            if _re.search(rf'\b{lower_name}\s+{av}\b', lower_story):
                return True

        # --- Strong negative signals: character performs human actions ---
        # If ANY of these match, character is definitely human.
        human_verbs = [
            "said", "smiled", "laughed", "replied", "sat", "nodded",
            "called", "tossed", "watched", "asked", "told", "whispered",
            "shouted", "walked", "stood", "ran", "held", "handed",
            "burst", "looked", "knew", "liked", "tried"
        ]
        for hv in human_verbs:
            if _re.search(rf'\b{lower_name}\s+{hv}\b', lower_story):
                return False  # Definitively human

        # Default: assume human when uncertain
        return False

    def analyze_cast_visuals(self, story_text, characters):
        """Generates rigid Visual DNA for characters (Immutable Physical + Signature Outfit)."""
        char_list = ", ".join(characters)
        intro_text = story_text[:1500].replace("\n", " ")

        # --- Pre-classify: identify which characters are animals ---
        self.animal_characters = set()
        for c in characters:
            if self._is_animal_character(c, story_text):
                self.animal_characters.add(c)
                logger.info(f"Character '{c}' identified as ANIMAL — skipping human Visual DNA.")
        
        human_characters = [c for c in characters if c not in self.animal_characters]

        prompt = f"""Instruction: Create rigid Visual DNA for consistent video generation.
Story Context: {intro_text}...
Characters: {', '.join(human_characters)}

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
                    for c in human_characters:
                        if c in name:
                            visuals[c] = {
                                "physical": physical,
                                "outfit": outfit,
                                "full_desc": f"{physical}, wearing {outfit}"
                            }
                            metadata[c] = {"gender": gender, "style": outfit}
                            break
        
        # Defaults for unresolved human characters
        default_looks = [
            ("Male", "short black hair, dark eyes, clean shaven, 20s", "casual shirt and jeans"),
            ("Female", "dark wavy hair, warm eyes, 20s", "light kurta and dupatta"),
            ("Male", "clean shaven face, sharp jawline, 20s", "plain t-shirt and trousers"),
            ("Male", "buzz cut, clean shaven, 20s", "hoodie and jeans")
        ]
        
        for i, c in enumerate(human_characters):
            if c not in visuals:
                import re as _re
                lower_story = story_text.lower()
                c_lower = c.lower()
                idx = lower_story.find(c_lower)
                window = lower_story[max(0, idx - 150) : idx + 150] if idx != -1 else lower_story
                fem_count = len(_re.findall(r'\b(she|her|hers)\b', window))
                masc_count = len(_re.findall(r'\b(he|him|his)\b', window))
                
                if fem_count > masc_count:
                    gender, phys, outf = default_looks[1] # Female
                else:
                    gender, phys, outf = default_looks[0] # Male fallback
                
                visuals[c] = {
                    "physical": phys,
                    "outfit": outf,
                    "full_desc": f"{phys}, wearing {outf}"
                }
                metadata[c] = {"gender": gender, "style": outf}

        # Set animal character visual descriptions — species-accurate, NOT human
        for c in self.animal_characters:
            lower_story = story_text.lower()
            lower_name = c.lower()
            # Search ALL occurrences of the character name for nearby animal keywords
            dog_keywords = ["dog", "puppy", "pup", "hound", "canine", "barking", "collar", "splashing"]
            cat_keywords = ["cat", "kitten", "feline", "meowing"]
            found_species = None
            idx = lower_story.find(lower_name)
            while idx != -1 and found_species is None:
                window = lower_story[max(0, idx - 120):idx + 120]
                if any(kw in window for kw in dog_keywords):
                    found_species = "dog"
                elif any(kw in window for kw in cat_keywords):
                    found_species = "cat"
                idx = lower_story.find(lower_name, idx + 1)
            # Also do a full-story scan as last resort
            if found_species is None:
                if any(kw in lower_story for kw in dog_keywords):
                    found_species = "dog"
                elif any(kw in lower_story for kw in cat_keywords):
                    found_species = "cat"

            if found_species == "dog":
                species_desc = f"small fluffy dog named {c}, with a collar"
                outfit_desc  = f"collar with name tag '{c}'"
            elif found_species == "cat":
                species_desc = f"small cat named {c}, soft fur"
                outfit_desc  = "no collar"
            else:
                species_desc = f"small animal named {c}"
                outfit_desc  = "no outfit"

            visuals[c] = {
                "physical":  species_desc,
                "outfit":    outfit_desc,
                "full_desc": species_desc,
                "is_animal": True
            }
            metadata[c] = {"gender": "Animal", "style": outfit_desc}

        
        self.character_visuals = visuals
        self.character_metadata = metadata
        logger.info(f"Generated Visual DNA: {visuals}")
        return visuals

    def analyze_story_background(self, story_text):
        intro_text = story_text[:1500].replace("\n", " ")
        prompt = f"""Instruction: Act as a Cinematographer. Read the story and describe the PRIMARY real-world setting in 5-10 words. 
Be specific about the environment, lighting, and mood. Use architectural or natural keywords (e.g. 'rocky cliff', 'brick wall', 'misty').

Examples:
Story about someone by a lake: "wide calm lake, golden sunset light, scenic reflections, mountain background"
Story at a house: "sunlit wooden porch, suburban garden, lush green lawn, bright day"
Story in an office: "sleek modern office, glass windows, city skyline, cold blue lighting"

Now describe the background for this story:
Story: {intro_text}

BACKGROUND:"""
        out = self.llm(prompt, max_tokens=30, stop=["\n", "Story:", "Instruction:"], temperature=0.2)
        raw_text = out["choices"][0]["text"]
        
        bg = raw_text.strip().strip('"').strip("'").split("\n")[0].strip()
        if bg.upper().startswith("BACKGROUND:"):
            bg = bg[len("BACKGROUND:"):].strip()
        
        if not bg or len(bg) < 5 or bg.lower() in ["none", "n/a", "unknown"]:
            import re as _re
            loc_match = _re.search(
                r'\b(?:by|at|near|beside|along|in)\s+(?:the\s+)?'
                r'([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)',
                story_text
            )
            if loc_match:
                loc_phrase = loc_match.group(0).strip()
                bg = f"scenic outdoor {loc_phrase}, natural lighting"
            else:
                bg = "scenic outdoor landscape, clear sky, natural lighting"
            
        logger.info(f"Generated Global Background: {bg}")
        return bg


    def refine_dialogue_emotion(self, speaker, text, archetype, context_window, base_emotion):
        valid_list = ", ".join([e.capitalize() for e in self.VALID_EMOTIONS])
        prompt = f"""Instruction: Act as an Expert Voice Director. Analyze the subtext to define the exact EMOTION and INTENSITY required for the actor's delivery.
Character: {speaker} ({archetype})
Line: "{text}"
Context: {context_window}
Surface Emotion: {base_emotion}

Task: 
1. Briefly analyze the character's true underlying emotion based on the context and the overall scene mood.
2. Choose a specific emotion strictly from this exact list: {valid_list}. Do NOT use any other words.
3. CRITICAL RULES:
   - If the context suggests friendly banter, teasing, or playful interaction, prefer Amusement or Joy over Annoyance.
   - Annoyance should ONLY be used when the character is genuinely irritated or frustrated.
   - If no strong emotion is present, default to Neutral.
   - Consider the relationship between characters — friends teasing each other are amused, not annoyed.

Format your response exactly like this:
ANALYSIS: [One sentence explaining the emotional subtext]
EMOTION: [Selected Emotion]
INTENSITY: [0.1 - 1.0]

Response:
"""
        out = self.llm(prompt, max_tokens=150, stop=["Instruction:", "Line:"], temperature=0.1)
        raw_text = out["choices"][0]["text"]
        
        pred_label = self._parse_key_value(raw_text, "EMOTION")
        pred_score = self._parse_key_value(raw_text, "INTENSITY")
        
        final_label = base_emotion
        final_score = 0.8

        if pred_label:
            clean_label = pred_label.lower().strip()
            for ve in self.VALID_EMOTIONS:
                if ve in clean_label:
                    final_label = ve
                    break

        if final_label not in self.VALID_EMOTIONS:
            final_label = "neutral"

        if pred_score:
            try:
                val = float(pred_score)
                final_score = min(max(val, 0.1), 1.0)
            except ValueError:
                final_score = 0.8

        return {"label": final_label, "intensity": final_score}

    def analyze_narration_tone(self, text, context_window):
        # Exclude emotions that are almost never appropriate for narration
        excluded_narration_tones = {"annoyance"}
        
        # For narration with explicitly positive actions, also exclude dark tones
        positive_action_words = {"laughed", "smiled", "grinned", "chuckled", "burst into laughter",
                                  "happily", "joyfully", "playfully", "cheerfully"}
        lower_text = text.lower()
        lower_context = context_window.lower() if context_window else ""
        combined_text = lower_text + " " + lower_context
        if any(pw in combined_text for pw in positive_action_words):
            excluded_narration_tones.update({"ominous", "disgust", "fear", "anger"})
        
        valid_emotions = [e for e in self.VALID_EMOTIONS if e not in excluded_narration_tones]
        valid_list = ", ".join([e.capitalize() for e in valid_emotions])
        prompt = f"""Instruction: Act as an Expert Voiceover Director. Analyze the narrative pacing and subtext to define the exact ATMOSPHERE and TONE for this narration.
Context: {context_window}
Narrator Line: "{text}"

Task: 
1. Briefly analyze the underlying atmospheric subtext and tension based on the context.
2. Choose a specific tone strictly from this exact list: {valid_list}. Do NOT use any other words.
3. CRITICAL: Match the tone to the ACTUAL mood of the scene:
   - If characters are laughing, smiling, or playing, use Amusement or Joy.
   - If the scene is calm and reflective, use Neutral.
   - Only use Ominous or Fear for genuinely dark/tense scenes.
   - Default to Neutral when unsure.

Format your response exactly like this:
ANALYSIS: [One sentence explaining the atmospheric subtext]
TONE: [Selected Tone]
INTENSITY: [0.1 - 1.0]

Response:
""" 
        out = self.llm(prompt, max_tokens=150, stop=["Instruction:", "Narrator Line:"], temperature=0.1)
        raw_text = out["choices"][0]["text"]
        
        tone = "neutral"
        intensity = 0.5
        
        pred_tone = self._parse_key_value(raw_text, "TONE")
        pred_int = self._parse_key_value(raw_text, "INTENSITY")
        
        if pred_tone:
            clean_tone = pred_tone.lower().strip()
            for ve in valid_emotions:
                if ve in clean_tone:
                    tone = ve
                    break
                    
        if tone not in valid_emotions:
            tone = "neutral"
                
        if pred_int:
             try:
                val = float(pred_int)
                intensity = min(max(val, 0.1), 1.0)
             except ValueError:
                intensity = 0.5
                
        return {"label": tone, "intensity": intensity}

    def generate_visual_prompt_v2(self, beat_data, location, active_cast, context_text=None, scene_props=None, fallback_speaker=None):
        """
        Strict visual prompt generator.

        Key guarantees:
        - Pronouns are resolved to named characters before the LLM sees the text.
        - Unknown speaker beats show the REACTION character, never an empty scene.
        - context_text (previous beat summary) is injected into the narration prompt.
        - The dog few-shot example has been replaced with a pronoun-resolution example
          to prevent contamination of unrelated scenes.
        - All dialogue beats include 'mouth slightly open mid-speech'.
        - Multi-character DNA blocks are always appended.
        """
        b_type = beat_data['type']
        text = beat_data['text']
        emotion = beat_data.get('emotion', {}).get('label', 'neutral')

        animal_chars = getattr(self, 'animal_characters', set())
        human_cast = [c for c in active_cast if c not in animal_chars]

        # ------------------------------------------------------------------ #
        # -- DIALOGUE BEATS                                                 -- #
        # ------------------------------------------------------------------ #
        if b_type == 'dialogue':
            speaker = beat_data.get('speaker', 'Unknown')

            # Resolve Unknown speaker ------------------------------------------
            if not speaker or speaker == 'Unknown':
                # 1. Try semantic subject
                semantic_subject = beat_data.get('semantic', {}).get('subject')
                if semantic_subject and semantic_subject in self.character_visuals:
                    speaker = semantic_subject
                # 2. Try fallback_speaker (last known active character)
                elif fallback_speaker and fallback_speaker in self.character_visuals:
                    speaker = fallback_speaker
                else:
                    # 3. Show the REACTION character (who is being spoken to),
                    #    not an empty wide shot.  The listener is the first
                    #    human in active_cast or the fallback.
                    react_char = fallback_speaker or (human_cast[0] if human_cast else None)
                    if react_char and react_char in self.character_visuals:
                        rc_vis = self.character_visuals.get(react_char, {})
                        rc_physical = rc_vis.get("physical", "") if isinstance(rc_vis, dict) else ""
                        rc_outfit   = rc_vis.get("outfit", "")   if isinstance(rc_vis, dict) else ""
                        rc_desc = ", ".join(filter(None, [rc_physical, f"wearing {rc_outfit}" if rc_outfit else ""])).strip(", ")
                        rc_desc_str = f", {rc_desc}" if rc_desc else ""
                        return (
                            f"High quality cinematic medium shot of {react_char}{rc_desc_str}, "
                            f"head turning, {emotion} expression, reacting to an off-screen voice, "
                            f"{location} in the background"
                        )
                    else:
                        # Absolute last resort: atmospheric environment shot
                        return (
                            f"High quality cinematic wide establishing shot of {location}, "
                            f"peaceful atmosphere, no characters"
                        )

            # Animal-as-speaker guard -----------------------------------------
            vis_data = self.character_visuals.get(speaker, {})
            if isinstance(vis_data, dict) and vis_data.get("is_animal"):
                animal_desc = vis_data.get("full_desc", f"{speaker}, animal")
                human_speaker = fallback_speaker
                if not human_speaker or human_speaker not in self.character_visuals:
                    human_speaker = next(
                        (c for c in active_cast
                         if c not in animal_chars
                         and not self.character_visuals.get(c, {}).get("is_animal")),
                        None
                    )
                if human_speaker:
                    hs_vis = self.character_visuals.get(human_speaker, {})
                    hs_physical = hs_vis.get("physical", "") if isinstance(hs_vis, dict) else ""
                    hs_outfit   = hs_vis.get("outfit", "")   if isinstance(hs_vis, dict) else ""
                    hs_desc = ", ".join(filter(None, [hs_physical, f"wearing {hs_outfit}" if hs_outfit else ""])).strip(", ")
                    hs_desc_str = f", {hs_desc}" if hs_desc else ""
                    return (
                        f"High quality cinematic close-up portrait of {human_speaker}{hs_desc_str}, "
                        f"{emotion} facial expression, reacting to {animal_desc}, "
                        f"{location} softly blurred in the background"
                    )
                else:
                    return (
                        f"High quality cinematic shot of a single {animal_desc}, "
                        f"{location} in the background"
                    )

            # Normal human speaker dialogue shot --------------------------------
            vis_data = self.character_visuals.get(speaker, {})
            if isinstance(vis_data, dict):
                physical = vis_data.get("physical", "")
                outfit   = vis_data.get("outfit", "")
                char_desc = ", ".join(filter(None, [physical, f"wearing {outfit}" if outfit else ""])).strip(", ")
            else:
                char_desc = str(vis_data)
            char_desc_str = f", {char_desc}" if char_desc else ""

            # mouth open is ALWAYS included for any dialogue beat
            return (
                f"High quality cinematic close-up portrait of {speaker}{char_desc_str}, "
                f"mouth slightly open mid-speech, {emotion} facial expression, "
                f"eyes reflecting emotion, "
                f"{location} softly blurred in the background"
            )

        # ------------------------------------------------------------------ #
        # -- NARRATION BEATS                                                -- #
        # ------------------------------------------------------------------ #
        else:
            lower_text = text.lower()

            # Step 1: Identify which characters are explicitly in this beat ---
            relevant_chars = [c for c in active_cast if c.lower() in lower_text]

            # Step 2: Fall back to semantic subject if no explicit name found -
            semantic_subject = beat_data.get('semantic', {}).get('subject')
            if semantic_subject and semantic_subject in active_cast and semantic_subject not in relevant_chars:
                relevant_chars.append(semantic_subject)

            # Step 3: Final fallback — use fallback_speaker or first cast member
            if not relevant_chars:
                if fallback_speaker and fallback_speaker in active_cast:
                    relevant_chars = [fallback_speaker]
                elif active_cast:
                    relevant_chars = active_cast[:1]

            # Step 4: Resolve pronouns in the beat text BEFORE sending to LLM -
            # Use the primary character (first in relevant_chars) as the antecedent.
            primary_char = relevant_chars[0] if relevant_chars else None
            # Derive secondary character for object-pronoun resolution:
            # pick the first known character that is NOT the primary.
            secondary_char = next(
                (c for c in self.character_visuals if c != primary_char),
                None
            )
            resolved_text = self._resolve_pronouns(text, primary_char, secondary_char)

            # Build chars_hint for the LLM prompt ----------------------------
            chars_in_beat = [c for c in active_cast if c.lower() in lower_text]
            chars_hint = ", ".join(chars_in_beat) if chars_in_beat else (
                primary_char if primary_char else "(none explicitly named)"
            )

            # Previous context line for scene continuity ----------------------
            context_line = f"Previous context: {context_text}" if context_text else ""

            # ---- Redesigned few-shot prompt ---------------------------------
            # IMPORTANT: The old Example A (dog running) was REMOVED because
            # Phi-2 was pattern-matching it onto unrelated scenes.
            # New examples focus on pronoun resolution and two-character scenes.
            narration_prompt = f"""Task: Summarize the Target Sentence into a highly descriptive visual prompt for image generation. Use comma-separated keywords (Noun, Action/Detail).

IMPORTANT RULES:
- Describe ONLY the action stated in the Target Sentence. Do NOT add props, animals, or events not mentioned.
- Only include characters listed under 'Characters present'. Do NOT invent others.
- Pronouns (he/she/they/him/her) have already been replaced with the character name — use the name as given.
- Characters in narration scenes face the scene or subject, NOT the camera.
- Do NOT add physical contact unless the sentence explicitly states it.
- Background: {location}
{context_line}

Example A:
Target Sentence: "Arjun threw a stone toward the lake."
Characters present: Arjun
Summary: Arjun throwing a stone toward the lake, side profile, gaze fixed on the water

Example B:
Target Sentence: "Meera sat beside Arjun, handing a cup sideways to Arjun."
Characters present: Meera, Arjun
Summary: Meera sitting next to Arjun, Meera handing a cup sideways to Arjun, both looking forward, not touching

Target Sentence: "{resolved_text}"
Characters present: {chars_hint}
Summary:"""

            out = self.llm(narration_prompt, max_tokens=50, stop=["\n", "Example", "Target:"], temperature=0.1)
            action = out["choices"][0]["text"].strip()
            action = re.sub(r"^(Narrator|Output|Summary|Description|Visual|Scene)[:\-\s]+", "", action, flags=re.IGNORECASE).strip()

            # Hard fallback if LLM output is too short -----------------------
            if len(action) < 5:
                semantic = beat_data.get('semantic', {})
                subj = semantic.get('subject') or primary_char
                verb = semantic.get('action')
                obj  = semantic.get('object')
                action = (
                    f"{subj} {verb}ing" + (f" {obj}" if obj else "")
                    if subj and verb else "quiet scene"
                )

            # ---- Build final_cast (names confirmed in beat OR action) ------
            beat_text_lower = text.lower()
            action_lower    = action.lower()
            final_cast = []
            for c_name in self.character_visuals.keys():
                name_lower = c_name.lower()
                in_beat   = name_lower in beat_text_lower
                in_action = bool(re.search(rf'\b{re.escape(c_name)}\b', action, re.IGNORECASE))
                if in_beat or in_action:
                    final_cast.append(c_name)

            # Preserve ordering — primary character first
            ordered = [c for c in relevant_chars if c in final_cast]
            for c in final_cast:
                if c not in ordered:
                    ordered.append(c)
            final_cast = ordered

            # ---- Build Character DNA appearance block ----------------------
            appearance_parts = []
            for c in final_cast:
                vis = self.character_visuals.get(c, {})
                if isinstance(vis, dict) and not vis.get("is_animal"):
                    physical = vis.get("physical", "")
                    outfit   = vis.get("outfit", "")
                    cdesc = ", ".join(filter(None, [physical, f"wearing {outfit}" if outfit else ""])).strip(", ")
                    if cdesc:
                        appearance_parts.append(f"{c}: {cdesc}")
                elif isinstance(vis, dict) and vis.get("is_animal"):
                    physical = vis.get("physical", c)
                    appearance_parts.append(f"{c}: {physical}")

            appearance_block = "  [" + " | ".join(appearance_parts) + "]" if appearance_parts else ""

            # Ensure background always has a lighting descriptor
            bg_anchor = location
            if "lighting" not in bg_anchor.lower():
                bg_anchor += ", natural lighting"

            return (
                f"High quality cinematic shot of {action}, "
                f"{bg_anchor} in the background{appearance_block}"
            )

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
            bgm_prompt = f"""Task: Determine a 3-5 word music style description for the background score.
If the scene does not need music, output "Silence".
Input Context: "The rain drummed on the roof. Julian sighed."
Emotion: "sadness"
Output: Melancholic acoustic guitar and rain
Input Context: "He picked up the cup."
Emotion: "neutral"
Output: Silence
Input Context: "{text}"
Emotion: "{emotion_label}"
Output:"""
            
            out_bgm = self.llm(bgm_prompt, max_tokens=15, stop=["\n", "Input Context:"], temperature=0.1)
            bgm_style = out_bgm["choices"][0]["text"].strip()
            
            # Clean up hallucinations (e.g. if the LLM repeats the story text)
            if len(bgm_style) < 3 or bgm_style.lower() == "none" or "sfx:" in bgm_style.lower():
                bgm_style = "Silence"
            elif len(bgm_style.split()) > 8 or '"' in bgm_style or 'voice' in bgm_style.lower() or 'spoke' in bgm_style.lower():
                bgm_style = "Silence"

            import random
            bgm_vol = round(random.uniform(0.30, 0.40), 2) if bgm_style != "Silence" else 0.0

            # --- SFX LOGIC ---
            # Strict few-shot prompt for Phi-2 preventing figurative sounds
            prompt = f"""Task: List perfectly literal, audible SFX. DO NOT output metaphorical sounds (e.g. if rain is "drumming", the sound is "Rain", NOT "Drums").
Input: "The rain drummed on the roof."
Output: SFX: Rain
Input: "He smelled the coffee."
Output: SFX: None
Input: "He sipped the tea."
Output: SFX: Sipping
Input: "Fingers dancing on the screen."
Output: SFX: Tapping
Input: "{text}"
Output:"""
            
            out = self.llm(prompt, max_tokens=20, stop=["Input:", "\n"], temperature=0.1)
            raw = out["choices"][0]["text"].strip()
            
            # Parse SFX
            if "SFX:" in raw:
                try:
                    raw_sfx = raw.split("SFX:")[1].split("\n")[0].strip()
                    if raw_sfx.lower() != "none" and raw_sfx.lower() != "no":
                        items = [x.strip() for x in raw_sfx.split(",")]
                        for item in items:
                            name = re.sub(r"\(.*?\)", "", item).strip()
                            # Filter out common LLM metaphorical hallucinations
                            if name and name.lower() not in ["none", "no sound", "drumming", "dancing"]:
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
        animal_chars = getattr(self, 'animal_characters', set())
        for char in characters:
            meta = self.character_metadata.get(char, {"gender": "Unknown", "style": "Standard"})
            vis = self.character_visuals.get(char, {})
            physical = vis.get("physical", "")

            if char in animal_chars:
                registry[char] = {
                    "type": "animal",
                    "archetype": profiles.get(char, "Animal"),
                    "species": meta.get("gender", "Unknown"),  # gender field repurposed as species
                    "visual_details": {
                        "physical": physical,
                        "is_animal": True
                    }
                }
            else:
                registry[char] = {
                    "type": "human",
                    "voice_model_id": f"en_us_generic_{char.lower()}",
                    "archetype": profiles.get(char, "Standard"),
                    "gender": meta.get("gender", "Unknown"),
                    "clothing_style": meta.get("style", "Standard"),
                    "visual_details": {
                        "physical": physical
                    }
                }
        return registry