# Research Paper: Methodology and Evaluation (Expanded)

## III. METHODOLOGY

The MAVIS (Multimodal Audio-Visual Intelligent System) framework implements a hierarchical, modular architecture designed to autonomously transform unstructured textual narratives into coherent, cinematic audio-visual sequences. The system mimics the workflow of a professional film production crew, employing specialized "agents" for scriptwriting, casting, directing, and post-production. The proposed architecture is divided into three distinct but interconnected processing stages: (A) Narrative Semantics & Event Extraction via Large Language Models (LLM), (B) Identity-Consistent Visual Synthesis via Diffusion Models, and (C) Affect-Conditioned Auditory Generation via Neural Audio Synthesis.

### A. Narrative Understanding and Event Extraction (The "Director" Engine)
The foundation of the pipeline is the **Narrative Intelligence Module**, which acts as the central orchestrator. Its primary objective is to translate raw literary text ($T_{raw}$) into a structured, machine-interpretable "Master Blueprint" ($\mathcal{B}$). We employ **Phi-2** (2.7B parameters) or **Llama-3**, chosen for their high reasoning capability relative to computational cost. The model is quantized to 4-bit (GGUF format) to enable local inference on consumer-grade hardware.

#### 1. Dynamic Character Serialization ("Visual DNA" Protocol)
A critical failure mode in generative video is "identity drift," where a character's appearance changes randomly between shots. MAVIS solves this via a dedicated **Identity Resolution Pass**.
*   **Algorithm:** The LLM scans the text to identify all entities $E = \{e_1, e_2, ..., e_n\}$. For each entity, it extracts immutable physical attributes (e.g., *age, ethnicity, hair style, facial structure, recurring outfit*) into a concise descriptor string, termed **"Visual DNA"**.
*   **Formulation:**
    $$DNA(C_i) = \text{Concat}(\text{Gender}, \text{Age}, \text{Physique}, \text{Hair}, \text{Apparel}_{default})$$
*   **Implementation:** This string is cached in a hash map `Map<CharacterName, DNA_String>` and is **strictly pre-pended** to the prompt of every visual beat involving that character. This acts as a semantic anchor, ensuring the diffusion model's cross-attention layers receive identical subject conditioning across widely divergent scenes.

#### 2. Hierarchical Temporal Segmentation (Beat Parsing)
The narrative is decomposed into a hierarchical tree structure of *Scenes* ($\mathcal{S}$) and *Beats* ($\mathcal{B}$).
*   **Scene Definition:** A contiguous block of time and space (e.g., "Julian's Office - Night").
*   **Beat Definition:** An atomic unit of storytelling, classified as either `Dialogue`, `Action`, or `Narration`.
*   **Extraction Tuple:** For each beat $b_j$, the extraction agent outputs a strictly typed JSON object:
    $$Beat(b_j) = \{ \text{ID, Type, Speaker, Dialogue, Action, Emotion, Location} \}$$
    This structure is validated against a Pydantic schema to eliminate "hallucinated" fields.

#### 3. Chain-of-Thought (CoT) Visual & Auditory Translation
Raw text often lacks explicit visual instructions. The Director Engine functions as a "translator" from *Diegetic Text* (story) to *Non-Diegetic Instructions* (production cues).
*   **Visual Translation:** "The room felt heavy" $\rightarrow$ "Dimly lit noir atmosphere, strong chiaroscuro lighting, long shadows, cigarette smoke."
*   **Auditory Translation:** "He felt a chill" $\rightarrow$ `Emotion="Fear"`, `BGM="Tense Industrial Drone"`, `SFX="Wind howling"`.

---

### B. Identity-Consistent Visual Synthesis (The "Visual Factory")
The visual module utilizes **Stable Diffusion v1.5** (or SDXL) via the **Hugging Face Diffusers** library. We deliberately avoid video generation models (e.g., SVD) in favor of high-fidelity frame generation to maximize controllability and resolution, prioritizing "Cinematic Stills" over "Warpy Video."

#### 1. Constructive Prompt Injection Algorithm
To guarantee consistency, we employ a deterministic prompt assembly algorithm. The final prompt $P_{final}$ sent to the CLIP Text Encoder is a concatenation of four semantic blocks:
$$P_{final} = \underbrace{\text{StylePrefix}}_{\text{"Cinematic shot..."}} \oplus \underbrace{DNA(C_i)}_{\text{"Julian, 30s..."}} \oplus \underbrace{\text{Action}(b_j)}_{\text{"looking at watch"}} \oplus \underbrace{\text{Env}(S_k)}_{\text{"rainy street"}} \oplus \underbrace{\text{Lighting}}_{\text{"neon signs"}} $$
*   **Priority Tokenization:** By placing the $DNA$ string at the *beginning* of the prompt, we effectively bias the standard Self-Attention mechanism of the U-Net. Tokens appearing earlier in the sequence generally exert greater influence on the final image structure.

#### 2. Negative Embedding Optimization
To counteract the model's tendency towards artifacts, we inject a static set of "Negative Embeddings" (learned vectors representing undesirable concepts) into the uncond (unconditional) path of the Classifier-Free Guidance (CFG).
*   $Neg = \{ \text{"monochrome", "low quality", "deformed", "mutated hands", "text", "watermark"} \}$

#### 3. Inference Optimization
To achieve practical synthesis speeds on research hardware (e.g., single NVIDIA GPU):
*   **Precision:** `float16` (Half Precision) is used to halve VRAM usage.
*   **Attention Slicing:** We enable `xformers` memory-efficient attention, reducing the complexity of the attention mechanism from $O(N^2)$ to $O(N)$, allowing for higher resolution (e.g., 512x768) batches.

---

### C. Affect-Conditioned Auditory Generation (The "Audio Factory")
MAVIS implements a novel "cascaded" audio pipeline that treats speech, music, and foley as separate layers that are mixed downstream.

#### 1. Expressive Speech Synthesis (Parler TTS & XTTS)
We employ a two-stage approach to achieve emotional nuance:
*   **Stage 1: Voice Design (Zero-Shot Casting):** Using **Parler TTS** (Mini v1), we generate a unique "Master Reference" constituent `.wav` file for each character based on their *Archetype* description (e.g., "A deep, gravelly voice for a 1940s detective").
*   **Stage 2: Performance Cloning:** For actual dialogue generation, we use **XTTS v2**. This model takes the "Master Reference" latent vector and clones its timbre while synthesizing new text. The *Emotion* tag from the Director Engine modulates the synthesis parameters:
    *   `Anger` $\rightarrow$ Pitch Shift (+), Speed (1.1x), Volume (+).
    *   `Whisper` $\rightarrow$ Speed (0.9x), Filtering (High-cut), Lower Amplitude.

#### 2. Context-Aware Music Generation (MusicGen)
Background Music (BGM) is generated using **Meta's MusicGen**. Instead of generic loops, we prompt the model with specific *Mood-Style* pairs derived from the scene's emotional arc.
*   *Prompt Strategy:* "A [Emotion] track in the style of [Genre], [Tempo], [Instrumentation]."
*   *Example:* "A suspenseful track in the style of Noir Jazz, slow tempo, solo saxophone and rain."

#### 3. Neural Foley Generation (AudioGen)
Discrete sound effects (SFX) are synthesized using **AudioGen**. The Director Engine identifies implicit sounds in the text (e.g., "He slammed the door" $\rightarrow$ `SFX="Door Slam"`) and schedules them on the timeline. This removes the need for manual stock library searching.

---

## IV. EVALUATION AND RESULTS

We conducted a quantitative evaluation of the generate pipeline using a pilot Noir narrative ("The Rusty Anchor"). The system processed 8 scenes and 16 beats.

### A. Audio Consistency and Expressiveness
We evaluated the audio generation using two key metrics: **Mel-Cepstral Distortion + Dynamic Time Warping (MCD-DTW)** to measure the "distance" between the generated voice and the master reference, and **Words Per Minute (WPM)** to analyze paging.

| Character | Avg MCD-DTW | Interpretation |
| :--- | :--- | :--- |
| **Silas** | **122.97** | **High Consistency.** The low variance indicates the model successfully locked the "Villain" archetype. |
| **Lena** | **126.66** | **High Consistency.** Consistent preservation of the female "Femme Fatale" timbre. |
| **Julian** | 137.69 | **High Expressiveness.** The higher deviation correctly correlates with the character's erratic emotional arc (Panic/Desperation). |
| **Narrator** | 144.33 | **Variable Tone.** The narrator shifted between "Neutral" and "Ominous" tones, explaining the wider acoustic variance. |

**Key Finding:** The system achieved a stable audio energy profile (Avg RMS: 0.11 ± 0.02), ensuring broadcast-safe levels without compression artifacts. However, the Narrator's average speaking rate (208 WPM max) was observed to be faster than standard audiobooks (150-160 WPM), suggesting a need for explicit pause token injection in future work.

### B. Visual Fidelity
Using the **Visual DNA** prompt injection method:
1.  **Identity Retention:** 92% of generated frames for the character "Julian" correctly maintained the key attribute *[Brown Trench Coat]*, reducing the "wardrobe hallucination" rate significantly compared to baseline Stable Diffusion (typically ~60% consistency).
2.  **Atmospheric Coherence:** 100% of scenes tagged `Lighting="Noir"` successfully generated low-key lighting effects, validating the Director Engine's translation logic.
