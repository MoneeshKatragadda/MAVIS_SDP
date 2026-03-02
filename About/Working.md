# MAVIS Project In-Depth Documentation

This document provides a comprehensive technical breakdown of the MAVIS (Multimodal Audio-Visual Intelligent System) codebase. It explains the purpose, logic, and key components of every file in the project.

---

## 🏗️ Core Architecture Files

### 1. `main.py` (The Director Pipeline)
**Purpose**: This is the central brain of the project. It orchestrates the transformation of raw text into a structured "Movie Blueprint" (`events.json`).
**Key Functions**:
*   `run_director()`: The main entry point. 
    1.  **Reads Input**: ingest `story.txt`.
    2.  **Profiling**: Uses `LLMReasoner` to generate character archetypes and visual DNA (consistent looks).
    3.  **Timeline Creation**: Iterates through the story, parsing it into "Scenes".
    4.  **Beat Analysis**: Breaks scenes into "Beats" (Dialogue or Narration). For each beat, it determines:
        *   **Emotion**: Using NLP and LLM refinement.
        *   **Cinematography**: Decides shot types (Close-up, Wide, etc.).
        *   **Directing**: Generates visual prompts for the image generator.
    5.  **Output**: Saves the structured timeline to `output/events.json`.
*   `patch_sfx_only()`: A utility to update sound effects in an existing JSON without re-running the whole costly pipeline.

### 2. `run_pipeline.py` (The Executive Producer)
**Purpose**: A high-level wrapper that automates the entire production chain sequentially.
**Workflow**:
1.  **Phase 1 (Director)**: Calls `main.run_director()` to generate the blueprint.
2.  **Phase 2 (Audio Factory)**: Calls `generate_audio()` to produce all voices, music, and SFX.
3.  **Phase 3 (Image Factory)**: Calls `generate_images()` to render every visual frame. 
    *   *Note*: Runs sequentially to avoid crashing your GPU (Audio and Image models both demand VRAM).

### 3. `config.yaml`
**Purpose**: Central configuration file.
**Key Settings**:
*   `models`: Paths to local GGUF models (Phi-2) and HuggingFace IDs (Stable Diffusion, Parler-TTS).
*   `paths`: Input/Output directory locations.

---

## 🧠 Intelligence Modules (`src/`)

### 4. `src/extractor.py` (NLP Engine)
**Purpose**: Handles deterministic linguistic analysis using Spacy.
**Key Components**:
*   `NLPExtractor` Class:
    *   `extract_characters_from_text`: Identifies proper nouns acting as subjects to find cast members.
    *   `parse_scene_structure`: Splits text into dialogue (quoted) and narration. Attributes dialogue to speakers based on context (e.g., "Julian said").
    *   `get_emotion`: Uses a RoBERTa model to classify text into 28 emotions (Joy, Anger, Fear, etc.).
    *   `extract_sfx`: Looks for specific sound verbs (thud, crash, whisper) to suggest Foley effects.

### 5. `src/llm_reasoner.py` (The Creative Mind)
**Purpose**: Interfaces with the Phi-2 LLM to perform "fuzzy" creative tasks.
**Key Functions**:
*   `analyze_cast_visuals`: "Hallucinates" consistent details for characters (e.g., "Julian always wears a brown trench coat") to ensure they look the same in every frame.
*   `refine_dialogue_emotion`: Adds nuance to acting. Converts "I hate you" from just "Anger" to "Cold, Calculating Fury".
*   `determine_shot_type`: Acting as a cinematographer, decides if a line needs a specific camera angle (Close-up vs Wide) based on dramatic importance.
*   `generate_visual_prompt_v2`: writes the exact prompt sent to Stable Diffusion, ensuring it includes the character's "Visual DNA" and the current action.

---

## 🎨 Generative Modules

### 6. `generate_cast.py` (Character Foundry)
**Purpose**: Pre-generates "Reference Sheets" for each character to be used as style inputs.
**Key Technology**: 
*   **Model**: `segmind/SSD-1B` (SDXL Mini). A distilled, faster version of Stable Diffusion XL.
*   **Optimization**: 
    *   **FP16 + Fixed VAE**: Uses a float32-patched VAE to prevent color corruption ("fried images").
    *   **Memory Tiling**: Processes images in chunks to run 1024px generation on consumer GPUs.
**Workflow**:
1.  Reads `output/characters.json`.
2.  Generates Front, Back, and Side views for every character.
3.  Enforces strict framing (Waist-Up) via negative prompting to avoid hallucinations.

### 7. `generate_audio.py` (The Sound Stage)
**Purpose**: Produces all audio assets.
**Phases**:
*   **Phase 1 (Casting)**: Uses **Parler-TTS** to generate a "Master Reference" file for each character based on a text description (e.g., "A rough, gravelly male voice").
*   **Phase 2 (Acting)**: Uses **XTTS-v2** (Voice Conversion) to speak every line of dialogue, cloning the "Master Reference" voice but injecting the specific emotion of the line.
*   **Phase 3 (Scoring)**: Uses **MusicGen** to compose background music (BGM) based on the scene's mood (e.g., "Noir Jazz", "Suspense Drone").
*   **Phase 4 (Foley)**: Uses prompt engineering to generate sound effects (SFX) that match the text actions.

---

## 📊 Evaluation & Quality Control

### 8. `evaluate_character_consistency.py`
**Purpose**: Quantifies how consistent your character generation is.
**Logic**: 
*   Uses **CLIP (Contrastive Language-Image Pretraining)** to embed images into vector space.
*   Calculates the cosine similarity between the "Front", "Side", and "Back" views of a character.
*   A high score means the AI isn't accidentally changing the character's face or clothes between shots.

### 9. `evaluate_audio.py`
**Purpose**: Checks the quality and identity of generated voices.
**Logic**:
*   Extracts **MFCCs** (Mel-frequency cepstral coefficients) from the generated audio.
*   Uses **DTW (Dynamic Time Warping)** to compare the generated line against the "Master Voice".
*   A low distance score means the generated voice works and sounds like the target actor.

---

## 📄 Dependency Files

### 10. `requirements.txt`
**Purpose**: Lists all Python libraries required to run the project.
*   Includes `torch` (AI backend), `diffusers` (Image Gen), `transformers` (NLP/Audio), and `llama-cpp-python` (LLM).

### 11. `output/events.json`
**Purpose**: The central data artifact. It contains the entire movie in machine-readable format.
*   **Timeline**: List of scenes.
*   **Beats**: Individual moments (lines of dialogue) with all metadata (Audio path, Image Prompt, Emotion, Duration).
