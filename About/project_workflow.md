# MAVIS Project Workflow & Architecture

## 1. Project Overview
MAVIS (Multimodal AI Video generation System) is an automated pipeline designed to convert text stories into cinematic videos. It orchestrates multiple AI models to perform "Virtual Film Production," distinct phases mimicking real-world movie making: Scripting/Direction, Casting, Principal Photography (Image Gen), Sound Engineering, and Editing.

## 2. Pipeline Architecture

The workflow is divided into three main sequential phases, orchestrated by `run_pipeline.py`.

### Phase 1: The Director Engine (`main.py`)
**Goal:** Analyze the raw text story and break it down into a structured shooting script (`events.json`).

*   **Inputs:** Raw Text File (Story).
*   **Sub-components:**
    *   **NLP Extractor (`src/extractor.py`)**: Uses **Spacy** to parse text, identify subjects/verbs/objects, and split complex sentences into individual "beats" (shots). Handles scene segmentation (splitting shots when logic changes, e.g., "Silas looks..." vs "The cook flips...").
    *   **LLM Reasoner (`src/llm_reasoner.py`)**: Uses **Phi-2 (via llama.cpp)** to function as a creative director.
        *   Extracts Character Visual DNA (Appearance, Clothing).
        *   Determines Emotion, Lighting, and Shot Type (Medium, Close-up) for each beat.
        *   Generates "Visual Prompts" for Image Gen (Action + Setting + Lighting).
        *   Generates "Audio Prompts" for TTS (Mood, Intensity).
*   **Outputs:** `output/events.json` (The "Master Contract" containing the entire timeline, character registry, and prompts).

### Phase 2: Production Factories
This phase generates the raw assets. It runs in two sequential streams to optimize GPU VRAM usage.

#### Phase 2a: Audio Factory (`generate_audio.py`)
**Goal:** Create all voice lines, background music, and sound effects.

*   **Sub-parts:**
    *   **Casting (Parler TTS)**: Generates a unique "Master Voice Reference" for each character based on their text description (archetype).
    *   **Production (XTTS v2)**: Uses the Master Reference to clone the voice and speak the dialogue lines with specific emotions (e.g., whispering, shouting).
    *   **Scoring (MusicGen)**: Generates Background Music (BGM) tracks tailored to the scene's mood (e.g., "Dark Suspense Drone", "Rainy Noir Ambience").
    *   **Foley (MusicGen/AudioGen)**: Generates Sound Effects (SFX) extracted from the text actions (e.g., "Footsteps", "Sizzling").
*   **Tech Stack:** `Parler TTS`, `Coqui TTS (XTTS v2)`, `Facebook MusicGen`.

#### Phase 2b: Visual Factory (`generate_images.py`)
**Goal:** Generate consistent cinematic frames for every beat.

*   **Process:**
    *   **Casting (Master Images)**: Uses the extracted "Visual DNA" to generate a high-quality "Master Reference Image" for each character (via `generate_cast.py`).
    *   **Scene Generation**: Reads `events.json` for visual prompts.
    *   **Consistency Injection**: Uses the Character Reference Images (conceptually) and injects the Visual DNA text into the scene prompts to maintain likeness across the movie.
    *   **Generation**: Uses **Segmind SSD-1B** (a distilled, faster version of SDXL) to generate 1024x1024 images.
    *   **Refinement**: Applies negative prompts to prevent distortions and enforces specific camera angles.
*   **Tech Stack:** `Diffusers`, `Segmind SSD-1B`, `PyTorch` (with memory optimizations like CPU Offload, VAE Tiling).

### Phase 3: Assembly & Editing (`movie.py`)
**Goal:** Stitch all assets into the final video file.

*   **Process:**
    *   **Timeline Assembly**: Iterates through the timeline in `events.json`.
    *   **Compositing**: Matches Images (`output/images/*.png`) with their corresponding Audio (`output/audio/*.wav`).
    *   **Audio Mixing**: Layers Dialogue, BGM (looped/ducked), and SFX into a Composite Audio Track.
    *   **Subtitles**: Overlays text captions using `TextClip` (Arial font) at the bottom of the screen.
    *   **Transitions**: Applies "Cross Fade" transitions between clips for smooth flow.
    *   **Rendering**: Encodes the final sequence into MP4.
*   **Tech Stack:** `MoviePy v2`, `FFMPEG`, `Pillow`.

## 3. Technology Stack Summary

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core orchestration |
| **Director Intelligence** | **Phi-2** (GGUF via `llama-cpp-python`) | Reasoning, Prompt Generation, Nuance extraction |
| **NLP** | **Spacy** (`en_core_web_lg`) | Semantic parsing, Sentence segmentation, Entity Recognition |
| **Image Generation** | **SSD-1B** (via `Diffusers`) | High-speed, high-quality cinematic image generation |
| **Voice Cloning** | **XTTS v2** (`TTS`) | Multi-speaker, emotional speech generation |
| **Voice Design** | **Parler TTS** | Creating unique voice profiles from text descriptions |
| **Music/SFX** | **MusicGen** (`Transformers`) | Generating atmospheric BGM and specific sound effects |
| **Video Editing** | **MoviePy v2** | Video compositing, transitions, subtitles |
| **Dependencies** | `Torch`, `Numpy`, `FFMPEG` | Core ML and Media processing frameworks |
