# MAVIS Pipeline Evaluation Report

## Vision Pipeline (SSD-1B / SDXL)
- **Mean CLIPScore (Text-Image Alignment):** 24.09
  - vs SSD-1B Baseline: 30.2 (Delta: -6.11)
  - vs SDXL Base Baseline: 32.0 (Delta: -7.91)
  - vs DALL-E 2: 31.5 (Delta: -7.41)

- **Mean Identity Consistency:** 39.76
  - Note: Values > 75 usually indicate recognizable identity preservation.

## Audio Pipeline (TTS) - Generalized Evaluation
### Overall Statistics
- **Total Audio Files Evaluated:** 6
- **Mean Word Error Rate (WER):** 0.1276 (12.76%)
- **Median WER:** 0.1151 (11.51%)
- **Std Dev WER:** 0.0838
- **Min WER:** 0.0400 (Best)
- **Max WER:** 0.2500 (Worst)
- **Mean Character Error Rate (CER):** 0.0558 (5.58%)

- **Total Words Processed:** 80

### Baseline Comparison
- **vs State-of-the-Art TTS (XTTS/Parler ~0.05):** ⚠️ Needs Improvement
- vs Tacotron 2 (0.08): Delta = 0.0476
- vs FastSpeech 2 (0.06): Delta = 0.0676
- vs VITS (0.04): Delta = 0.0876

### Speaker-Specific Analysis
- **Julian:** Mean WER = 0.0455, Files = 1, Std = nan
- **Narrator:** Mean WER = 0.1441, Files = 5, Std = 0.0822

### Scene-Wise WER Analysis
- **SC_001:** Mean WER = 0.0400, Files = 1
- **SC_005:** Mean WER = 0.1053, Files = 1
- **SC_004:** Mean WER = 0.1250, Files = 1
- **SC_006:** Mean WER = 0.1477, Files = 2
- **SC_007:** Mean WER = 0.2000, Files = 1

### Example Transcriptions
#### Best (Lowest WER = 0.0400)
- **Scene:** SC_001_01
- **Reference:** The rain drummed a relentless rhythm against the fogged windows of The Rusty Anchor, a small diner t...
- **Recognized:** The rain drummed a relentless rhythm against the fog windows of the rusty anchor, a small diner that...

#### Worst (Highest WER = 0.2500)
- **Scene:** SC_006_02
- **Reference:** Julian hissed, leaning in...
- **Recognized:** Julian Hist, leaning in....

### Detailed Transcription Report
| Scene | Beat | WER | Reference | Recognized |
|---|---|---|---|---|
| SC_001 | SC_001_01 | 0.0400 | The rain drummed a relentless rhythm against the fogged windows of The Rusty Anchor, a small diner that smelled of burnt coffee and old vinyl | The rain drummed a relentless rhythm against the fog windows of the rusty anchor, a small diner that smelled of burnt coffee and old vinyl. |
| SC_004 | SC_004_01 | 0.1250 | Julian swallowed hard, his Adam's apple bobbing | Julian swallowed heart his Adam's apple bobbing. |
| SC_005 | SC_005_01 | 0.1053 | Lena rolled her eyes, her fingers dancing across the screen of a tablet she had hidden under a napkin | Mina rolled her eyes, her fingers dancing across the screen of a tablet she had hidden under an napkin. |
| SC_006 | SC_006_02 | 0.2500 | Julian hissed, leaning in | Julian Hist, leaning in. |
| SC_006 | SC_006_03 | 0.0455 | Lena's got it covered. Can we just do this? I don't like the way the cook is looking at us. | Nina's got it covered. Can we just do this? I don't like the way the cook is looking at us. |
| SC_007 | SC_007_01 | 0.2000 | Silas glanced toward the counter | Silas glans toward the counter. |

## Audio Word-Level Precision / Recall / F1
*(Bag-of-words comparison between ASR output and reference text)*
- **Mean Precision:** 0.9550 (95.50%)
- **Mean Recall:** 0.9550 (95.50%)
- **Mean F1:** 0.9550 (95.50%)

### Speaker-level F1
- **Lena:** F1 = 1.0000
- **Silas:** F1 = 1.0000
- **Julian:** F1 = 0.9909
- **Narrator:** F1 = 0.9200

## Visual Prompt Coverage (Prompt vs Beat Text)
*(How well the visual_prompt captures key content words from the script beat)*
- **Mean Precision:** 0.1410 — words in prompt that came from the beat
- **Mean Recall:** 0.3437 — beat words captured in the prompt
- **Mean F1:** 0.1935

### Coverage by Beat Type
- **dialogue:** F1 = 0.0000
- **narration:** F1 = 0.3655
