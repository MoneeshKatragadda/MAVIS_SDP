# -*- coding: utf-8 -*-
import re
import spacy
import torch
from transformers import pipeline
from nltk.corpus import wordnet as wn
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# NLTK Downloads (Quietly)
try: nltk.data.find('vader_lexicon')
except LookupError: nltk.download('vader_lexicon', quiet=True)
try: nltk.data.find('wordnet')
except LookupError: nltk.download('wordnet', quiet=True)
try: nltk.data.find('omw-1.4')
except LookupError: nltk.download('omw-1.4', quiet=True)

class NLPExtractor:
    def __init__(self, config):
        self.nlp = spacy.load(config["models"]["spacy_model"])
        self.emotion_model = config["models"]["emotion_model"]
        self._emotion_pipe = None
        self.sia = SentimentIntensityAnalyzer()
        
        # Expanded sound lemmas to catch more varied SFX verbs
        self.sound_lemmas = {
            "drum", "hiss", "snap", "click", "scream", "whisper", "shout", 
            "thud", "bang", "ring", "crash", "creak", "rustle", "beep", "siren", 
            "slap", "gulp", "swallow", "breath", "step", "clatter", "buzz", "engine", "run",
            "sip", "drink", "tap", "type", "knock", "rattle", "sizzle", "hum", "whir", "laugh"
        }

        self.entity_blocklist = {
            "way", "time", "idea", "series", "security", "side", "line", 
            "business", "risk", "reward", "moment", "hair", "hairs", "grace",
            "name", "internet", "signal", "fence", "air", "tension", "rhythm",
            "aggression", "bucket", "life", "spreadsheet", "account", "glow",
            "door", "handle", "something", "anything", "nothing", "everything",
            "one", "two", "three", "four", "five", "first", "second", "third",
            "detail", "bit", "lot", "bunch", "kind", "part"
        }
        
        self.BODY_ACTION_MAP = {
            "voice": "speak", "eyes": "look", "gaze": "look", "hand": "gesture",
            "finger": "touch", "legs": "stand", "head": "turn", "face": "express"
        }
        
        self.NOUN_ACTION_MAP = {
            "breath": "breathe", "sigh": "sigh", "step": "walk",
            "look": "look", "glance": "look", "smile": "smile", "nod": "nod"
        }

    # ---------------- Emotion ----------------
    def load_emotion_model(self):
        if self._emotion_pipe is None:
            device = 0 if torch.cuda.is_available() else -1
            self._emotion_pipe = pipeline(
                "text-classification",
                model=self.emotion_model,
                top_k=None, 
                device=device
            )

    def get_emotion(self, text, beat_type):
        if not text.strip(): 
            return {"label": "neutral", "intensity": 0.0}
            
        results = self._emotion_pipe(text[:512])[0]
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        top = results[0]
        second = results[1]
        
        label = top["label"].lower()
        score = float(top["score"])

        if label == "neutral" and score < 0.85:
            if second["label"].lower() not in ["approval", "realization"]: 
                label = second["label"].lower()
                score = float(second["score"]) * 1.5 
                if score > 0.9: score = 0.9

        intensity = min(score, 0.99)
        return {"label": label, "intensity": round(intensity, 3)}

    # ---------------- SFX & Semantics ----------------
    def extract_sfx(self, text):
        doc = self.nlp(text)
        sfx = []
        
        # Explicit lists for POS-based filtering
        active_verbs = {
            "drum", "hiss", "snap", "click", "scream", "whisper", "shout", 
            "thud", "bang", "crash", "creak", "rustle", "beep", 
            "slap", "gulp", "swallow", "step", "clatter", "buzz", "run",
            "sip", "drink", "tap", "type", "knock", "rattle", "sizzle", "hum", 
            "whir", "laugh", "cry", "sob", "gasp", "groan", "choke", "breathe", "sigh"
        }
        
        # Nouns that inherently imply sound
        sound_nouns = {
            "rain", "thunder", "wind", "storm", "siren", "explosion", 
            "gunshot", "scream", "laughter", "applause", "silence", "noise", 
            "footsteps", "voice", "clamor", "uproar"
        }

        for t in doc:
            lemma = t.lemma_.lower()
            
            # Verbs: Must be an action
            if t.pos_ == "VERB" and lemma in active_verbs:
                sfx.append(lemma)
            
            # Nouns: Must be an explicit sound source
            elif t.pos_ == "NOUN" and (lemma in sound_nouns or lemma in active_verbs):
                # checking 'lemma in active_verbs' handles "a tap", "a click"
                sfx.append(lemma)
                
        return sorted(list(set(sfx)))

    def extract_svo(self, text, context_subject=None):
        doc = self.nlp(text)
        
        for t in doc:
            if t.dep_ == "nsubj":
                verb = t.head.lemma_
                if t.head.dep_ == "aux": verb = t.head.head.lemma_
                if verb == "be" and t.head.text.lower() in ["'s", "'re"]: verb = "is/are"

                subject = t.text
                obj = None
                for c in t.head.children:
                    if c.dep_ in {"dobj", "pobj", "attr", "acomp"}:
                        obj = c.text; break
                
                if subject.lower() in self.BODY_ACTION_MAP:
                    if context_subject:
                        subject = context_subject
                        verb = self.BODY_ACTION_MAP.get(t.text.lower(), verb)

                if t.pos_ == "PRON" and context_subject:
                    subject = context_subject

                return {"subject": subject, "action": verb, "object": obj}

        for t in doc:
            if t.pos_ == "NOUN" and t.lemma_ in self.NOUN_ACTION_MAP:
                action = self.NOUN_ACTION_MAP[t.lemma_]
                return {"subject": context_subject, "action": action, "object": None}

        return {"subject": context_subject, "action": "react", "object": None} if context_subject else {"subject": None, "action": None, "object": None}

    # ---------------- Parsing & Segmentation ----------------
    def parse_scene_structure(self, text, characters):
        beats = []
        parts = re.split(r'(".*?")', text)
        last_speaker = None
        last_narration_subject = None
        local_scene_cast = set()

        for p in parts:
            if not p.strip(): continue

            if p.startswith('"') and p.endswith('"'):
                clean_text = p.strip('"')
                speaker = self._resolve_speaker(text, p, characters, last_speaker, last_narration_subject, local_scene_cast)
                
                if speaker and speaker != "Unknown":
                    last_speaker = speaker
                    local_scene_cast.add(speaker)

                beats.append({
                    "type": "dialogue",
                    "speaker": speaker, 
                    "text": clean_text,
                    "duration": self._duration(clean_text, 2.5)
                })
            else:
                sub_parts = self._split_complex_sentences(p.strip(), characters)
                for sub_text in sub_parts:
                    if not sub_text.strip(): continue
                    subj = self._find_narration_subject(sub_text, characters)
                    if subj: 
                        last_narration_subject = subj
                        local_scene_cast.add(subj)
                    
                    beats.append({
                        "type": "narration",
                        "text": sub_text,
                        "duration": self._duration(sub_text, 2.0)
                    })

        return {"beats": beats, "active_chars": sorted(list(local_scene_cast))}

    def _split_complex_sentences(self, text, characters):
        doc = self.nlp(text)
        final_splits = []
        
        for sent in doc.sents:
            char_subjects = set()
            for t in sent:
                if t.dep_ == "nsubj" and t.text in characters:
                    char_subjects.add(t.text)
            
            # If multi-character interaction, use rigorous subject-based splitting
            if len(char_subjects) >= 2:
                sub_segments = self._segment_multi_char_sentence(sent, characters)
                final_splits.extend(sub_segments)
            else:
                # If single character (or none), still check for structural complexity (e.g., "where a man...", "while looking...")
                # This fixes the "Silas ... cook" issue where secondary entities are important
                structural_split = False
                tokens = list(sent)
                for i, t in enumerate(tokens):
                    # Look for structural splitters that aren't just normal conjunctions
                    if t.text.lower() in ["where", "while"] and i > 3 and (len(tokens) - i) > 4:
                        # Ensure we don't split "where are you?" or short phrases
                        
                        # Split!
                        seg1 = sent[:i].text.strip()
                        seg2 = sent[i:].text.strip()
                        
                        # Clean trailing commas from seg1
                        if seg1.endswith(","): seg1 = seg1[:-1].strip()
                        
                        final_splits.append(seg1)
                        final_splits.append(seg2)
                        structural_split = True
                        break
                
                if not structural_split:
                    final_splits.append(sent.text.strip())
            
        return [s.strip(" ,.") for s in final_splits if len(s.strip()) > 2]

    def _segment_multi_char_sentence(self, sent, characters):
        tokens = list(sent)
        subj_map = {} 
        for i, t in enumerate(tokens):
            if t.dep_ == "nsubj" and t.text in characters:
                subj_map[i] = t.text

        sorted_subj_indices = sorted(subj_map.keys())
        
        if len(sorted_subj_indices) < 2:
            return [sent.text]

        current_start = 0
        segments = []

        for i in range(len(sorted_subj_indices) - 1):
            idx_a = sorted_subj_indices[i]
            idx_b = sorted_subj_indices[i+1]
            char_a = subj_map[idx_a]
            char_b = subj_map[idx_b]

            # Split only if subjects differ (interaction) or explicit disconnect
            best_split = -1
            for k in range(idx_a + 1, idx_b):
                if tokens[k].text == "," or tokens[k].text == ";":
                    best_split = k + 1 
                    break
                elif tokens[k].pos_ == "CCONJ" or tokens[k].text.lower() in ["while", "as", "but"]:
                     best_split = k 
                     if k > 0 and tokens[k-1].text == ",":
                         best_split = k-1
                     break
            
            if best_split != -1:
                seg_text = sent[current_start:best_split].text.strip()
                if len(seg_text) > 2:
                    segments.append(seg_text)
                current_start = best_split 

        last_seg = sent[current_start:].text.strip()
        if len(last_seg) > 2:
            segments.append(last_seg)
            
        return segments if segments else [sent.text]

    def _find_narration_subject(self, text, characters):
        doc = self.nlp(text)
        for t in doc:
            if t.dep_ == "nsubj" and t.text in characters:
                return t.text
        return None

    def _resolve_speaker(self, full_text, quote, characters, last_speaker, last_narration_subject, local_scene_cast):
        doc = self.nlp(quote.strip('"'))
        addressed = [e.text for e in doc.ents if e.text in characters]
        
        idx = full_text.find(quote)
        candidate = None
        if idx != -1:
            pre = full_text[max(0, idx-100):idx]
            post = full_text[idx+len(quote):min(len(full_text), idx+len(quote)+100)]
            context_doc = self.nlp(pre + " ... " + post)
            
            speech_verbs = {
                "say", "ask", "whisper", "shout", "hiss", "mutter", "yell", 
                "interrupt", "reply", "warn", "command", "tell", "snap", "continue", "add"
            }
            
            for t in context_doc:
                if t.lemma_ in speech_verbs:
                    for child in t.children:
                        if child.dep_ == "nsubj" and child.text in characters:
                            candidate = child.text

        speaker = candidate or last_narration_subject or last_speaker or "Unknown"

        if speaker == "Unknown" and len(local_scene_cast) == 1:
            speaker = list(local_scene_cast)[0]

        if speaker in addressed and len(characters) > 1:
            alts = [c for c in characters if c != speaker]
            return alts[0] if alts else "Unknown"

        return speaker

    # ---------------- Entity Extraction ----------------
    def extract_scene_entities(self, text, characters):
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.text in characters: continue
            if ent.text.lower() in self.entity_blocklist: continue
            etype = "prop"
            if ent.label_ in {"GPE", "LOC", "FAC"}: etype = "location"
            elif ent.label_ == "ORG": etype = "organization" 
            entities.append({"name": ent.text, "type": etype, "role": "background"})

        for t in doc:
            if t.text.lower() in self.entity_blocklist: continue
            if len(t.text) < 3: continue
            if t.pos_ == "NOUN" and not t.ent_type_:
                if t.text.isdigit(): continue
                synsets = wn.synsets(t.lemma_, pos=wn.NOUN)
                is_phys = False
                is_abstract = False
                for s in synsets:
                    hypernyms = [p.name() for p in s.lowest_common_hypernyms(wn.synset('entity.n.01'))]
                    if "artifact.n.01" in hypernyms or "physical_entity.n.01" in hypernyms: is_phys = True
                    if "abstraction.n.06" in hypernyms: is_abstract = True
                if is_phys and not is_abstract:
                    entities.append({"name": t.text.lower(), "type": "prop", "role": "background"})

        unique = {}
        for e in entities: unique[e['name'].lower()] = e
        return list(unique.values())

    def extract_characters_from_text(self, text):
        doc = self.nlp(text)
        candidates = set()
        BLOCKLIST = {"Board", "Formica", "Teflon", "Monday", "Sunday", "Rusty Anchor", "Anchor", "Adam", "Street", "Station"}
        active_deps = {"nsubj", "nsubjpass", "dobj", "iobj", "vocative"}
        for t in doc:
            if t.pos_ == "PROPN":
                clean_name = t.text.strip().strip('"').strip("'")
                if clean_name in BLOCKLIST: continue
                if t.ent_type_ in {"ORG", "GPE", "LOC", "FAC", "PRODUCT", "DATE", "TIME", "CARDINAL", "ORDINAL"}: continue
                if t.dep_ in active_deps:
                    if len(clean_name) > 2 and clean_name[0].isupper():
                        candidates.add(clean_name)
                elif t.ent_type_ == "PERSON":
                    if t.dep_ not in {"poss", "compound"}:
                         if len(clean_name) > 2 and clean_name[0].isupper():
                            candidates.add(clean_name)
        return sorted(list(candidates))

    def build_audio_prompt(self, beat, emo):
        speaker = beat.get("speaker", "Narrator") or "Narrator"
        return f"{speaker}, tone={emo['label']}, intensity={emo['intensity']}"

    def strip_dialogue(self, text):
        return re.sub(r'".*?"', '', text).strip()

    def _duration(self, text, speed_factor):
        return round(max(1.0, len(text.split()) / speed_factor), 2)