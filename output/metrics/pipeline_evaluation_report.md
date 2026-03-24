# MAVIS Pipeline Evaluation Report

## Executive Summary (0-100 Scale)
| Dimension | Score (0-100) | Interpretation |
|---|---|---|
| **1. Vision Quality** | 23.1 | Visual fidelity and text-to-image alignment (CLIP) |
| **2. Audio Quality** | 87.9 | Speech intelligibility and low error rates (1-WER) |
| **3. Character Consistency** | 57.9 | Face geometry (ArcFace) & body/clothing retention |
| **4. Emotion Realism** | 0.0 | Intent preservation across pipeline (BERTScore F1) |
| **5. Narrative Accuracy** | 19.3 | Script-to-visual prompt coverage (Keyword overlap) |
| **6. Multimodal Sync** | 82.9 | Generated audio duration vs intended scene timing |

---

## Detailed Metrics Analysis

### 1. Vision Pipeline (SSD-1B / SDXL)
- **Mean CLIPScore (Text-Image Alignment):** 23.46
  - vs StoryDALL-E: 29.5 (Delta: -6.04)
  - vs Make-A-Story: 30.1 (Delta: -6.64)
  - vs MM-StoryAgent: 32.8 (Delta: -9.34)
  - vs DALL-E 2: 31.5 (Delta: -8.04)
  - vs SDXL Base: 32.0 (Delta: -8.54)
  - vs SSD-1B (Ours Base): 30.2 (Delta: -6.74)

- **Mean Identity Consistency (CLIP, body+clothing):** 83.41
  - Captures clothing, body shape & face semantically. Values > 75 indicate strong visual identity.

- **Mean Face Identity (ArcFace, face-only):** 0.324
  - Pure facial geometry similarity. >0.4 = same person, >0.6 = high confidence.
  - **Julian:** 0.354
  - **Silas:** 0.291
  - **Lena:** 0.240

## Audio Pipeline (TTS) - Generalized Evaluation
### Overall Statistics
- **Total Audio Files Evaluated:** 4
- **Mean Word Error Rate (WER):** 0.1211 (12.11%)
- **Median WER:** 0.0909 (9.09%)
- **Std Dev WER:** 0.0878
- **Min WER:** 0.0526 (Best)
- **Max WER:** 0.2500 (Worst)
- **Mean Character Error Rate (CER):** 0.0686 (6.86%)

- **Total Words Processed:** 53

### Baseline Comparison
- **vs State-of-the-Art TTS (StyleTTS ~0.05):** ⚠️ Needs Improvement
- vs Tacotron 2 (0.08): Delta = 0.0411
- vs FastSpeech 2 (0.06): Delta = 0.0611
- vs VITS (0.04): Delta = 0.0811

### Speaker-Specific Analysis
- **Julian:** Mean WER = 0.0909, Files = 1, Std = nan
- **Silas:** Mean WER = 0.0909, Files = 1, Std = nan
- **Narrator:** Mean WER = 0.1513, Files = 2, Std = 0.1396

### Scene-Wise WER Analysis
- **SC_005:** Mean WER = 0.0526, Files = 1
- **SC_003:** Mean WER = 0.0909, Files = 1
- **SC_006:** Mean WER = 0.1705, Files = 2

### Example Transcriptions
#### Best (Lowest WER = 0.0526)
- **Scene:** SC_005_01
- **Reference:** Lena rolled her eyes, her fingers dancing across the screen of a tablet she had hidden under a napki...
- **Recognized:** Lena rolled her eyes, her fingers dancing across the screen of a tablet. She had hidden under an nap...

#### Worst (Highest WER = 0.2500)
- **Scene:** SC_006_02
- **Reference:** Julian hissed, leaning in...
- **Recognized:** Julian Hist Leningen...

### Detailed Transcription Report
| Scene | Beat | WER | Reference | Recognized |
|---|---|---|---|---|
| SC_003 | SC_003_02 | 0.0909 | You're late, Julian. In this business, late usually means followed. | your late Julian. In this business, late usually means followed. |
| SC_005 | SC_005_01 | 0.0526 | Lena rolled her eyes, her fingers dancing across the screen of a tablet she had hidden under a napkin | Lena rolled her eyes, her fingers dancing across the screen of a tablet. She had hidden under an napkin. |
| SC_006 | SC_006_02 | 0.2500 | Julian hissed, leaning in | Julian Hist Leningen |
| SC_006 | SC_006_03 | 0.0909 | Lena's got it covered. Can we just do this? I don't like the way the cook is looking at us. | liners get it covered. Can we just do this? I don't like the way the cook is looking at us. |

## Audio Word-Level Precision / Recall / F1
*(Bag-of-words comparison between ASR output and reference text)*
- **Mean Precision:** 0.9034 (90.34%)
- **Mean Recall:** 0.8913 (89.13%)
- **Mean F1:** 0.8968 (89.68%)

### Speaker-level F1
- **Lena:** F1 = 1.0000
- **Julian:** F1 = 0.9767
- **Silas:** F1 = 0.9285
- **Narrator:** F1 = 0.8340

## Visual Prompt Coverage (Prompt vs Beat Text)
*(How well the visual_prompt captures key content words from the script beat)*
- **Mean Precision:** 0.1410 — words in prompt that came from the beat
- **Mean Recall:** 0.3437 — beat words captured in the prompt
- **Mean F1:** 0.1935

### Coverage by Beat Type
- **dialogue:** F1 = 0.0000
- **narration:** F1 = 0.3655
