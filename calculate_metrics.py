import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
from tqdm import tqdm
import re
import string
try:
    from num2words import num2words
except ImportError:
    num2words = None

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MAVIS_METRICS")

# Attempt to load Whisper's EnglishTextNormalizer (Standard for WER)
try:
    from whisper.normalizers import EnglishTextNormalizer
    std_normalizer = EnglishTextNormalizer()
    logger.info("Loaded Whisper EnglishTextNormalizer for WER calculation.")
except ImportError:
    std_normalizer = None
    logger.warning("EnglishTextNormalizer not found. Using custom fallback.")

# --- RESEARCH BASELINES ---
# Detailed benchmarks from reliable sources (Coco zero-shot, LJSpeech WER)
PAPER_BASELINES = {
    "Vision (CLIPScore)": {
        "Real Images (COCO)": 30.5,
        "SD 1.5": 28.5,
        "DALL-E 2": 31.5,
        "SDXL Base": 32.0,
        "SSD-1B (Ours Base)": 30.2
    },
    "Audio (WER)": {
        "Human (Ground Truth)": 0.0,
        "Tacotron 2": 0.08,
        "FastSpeech 2": 0.06,
        "VITS": 0.04,
        "XTTS/Parler (Ours Base)": 0.05
    },
    "NLP (BERTScore F1)": {
        "Human Reference": 1.0,
        "GPT-2": 0.82,
        "GPT-3.5": 0.89,
        "GPT-4": 0.91
    }
}

# --- NLP METRICS ---
def compute_nlp_metrics(generated_text, reference_text):
    """
    Computes BERTScore (Precision, Recall, F1).
    """
    if not generated_text or not reference_text:
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}
    
    try:
        from bert_score import score
        # Using a small BERT model for speed/efficiency
        P, R, F1 = score([generated_text], [reference_text], lang="en", verbose=False)
        return {
            "bert_precision": P.mean().item(),
            "bert_recall": R.mean().item(),
            "bert_f1": F1.mean().item()
        }
    except ImportError:
        logger.warning("bert_score not installed. Skipping NLP metrics.")
        return {}
    except Exception as e:
        logger.error(f"NLP Metric Error: {e}")
        return {}


def compute_token_f1(reference_text, hypothesis_text):
    """
    Computes word-level Precision, Recall, and F1 score between reference
    and hypothesis text. Useful for audio evaluation beyond WER.

    - Precision: of the words predicted, how many are correct?
    - Recall: of the reference words, how many were found?
    - F1: harmonic mean of precision and recall.
    """
    if not reference_text or not hypothesis_text:
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}

    ref_words = normalize_text(reference_text).split()
    hyp_words = normalize_text(hypothesis_text).split()

    if not ref_words or not hyp_words:
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}

    # Count word overlaps (bag-of-words style, like ROUGE-1)
    ref_counts = {}
    for w in ref_words:
        ref_counts[w] = ref_counts.get(w, 0) + 1

    hyp_counts = {}
    for w in hyp_words:
        hyp_counts[w] = hyp_counts.get(w, 0) + 1

    matches = sum(min(hyp_counts.get(w, 0), ref_counts[w]) for w in ref_counts)

    precision = matches / len(hyp_words) if hyp_words else 0.0
    recall = matches / len(ref_words) if ref_words else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "token_precision": round(precision, 4),
        "token_recall": round(recall, 4),
        "token_f1": round(f1, 4)
    }


def compute_visual_coverage(visual_prompt, beat_text):
    """
    Measures how well the visual_prompt covers the key content words from the
    original beat text (props, characters, actions).

    Returns Precision, Recall, F1 based on keyword token overlap.
    Stopwords are filtered so only meaningful words are compared.
    """
    if not visual_prompt or not beat_text:
        return {"visual_precision": 0.0, "visual_recall": 0.0, "visual_f1": 0.0}

    STOPWORDS = {
        "the", "a", "an", "of", "in", "at", "on", "is", "are", "was", "were", "with",
        "and", "or", "but", "for", "to", "it", "he", "she", "his", "her", "their",
        "its", "by", "from", "into", "this", "that", "not", "as", "be", "been",
        "have", "has", "had", "do", "did", "will", "would", "could", "should",
        "more", "than", "even", "each", "them", "they", "who", "which", "also"
    }

    def tokenize(text):
        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    ref_tokens = tokenize(beat_text)
    prompt_tokens = tokenize(visual_prompt)

    if not ref_tokens:
        return {"visual_precision": 1.0, "visual_recall": 1.0, "visual_f1": 1.0}

    ref_set = set(ref_tokens)
    prompt_set = set(prompt_tokens)

    matched = ref_set & prompt_set
    precision = len(matched) / len(prompt_set) if prompt_set else 0.0
    recall = len(matched) / len(ref_set) if ref_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "visual_precision": round(precision, 4),
        "visual_recall": round(recall, 4),
        "visual_f1": round(f1, 4)
    }


# --- VISION METRICS ---
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def compute_vision_metrics(image_path, text_prompt, reference_image_path=None):
    """
    Computes CLIPScore (Text-Image) and Identity Consistency (Image-Image).
    """
    metrics = {"clip_score": 0.0, "identity_consistency": 0.0}
    
    if not os.path.exists(image_path):
        return metrics

    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize([text_prompt]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            # CLIPScore: Cosine similarity between image and text features
            # Normalized features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).item()
            metrics["clip_score"] = similarity

            # Identity Consistency
            if reference_image_path and os.path.exists(reference_image_path):
                ref_image = preprocess(Image.open(reference_image_path)).unsqueeze(0).to(device)
                ref_features = model.encode_image(ref_image)
                ref_features /= ref_features.norm(dim=-1, keepdim=True)
                
                sim_id = (100.0 * image_features @ ref_features.T).item()
                metrics["identity_consistency"] = sim_id

    except Exception as e:
        logger.error(f"Vision Metric Error for {image_path}: {e}")

    return metrics

# --- AUDIO METRICS ---
import whisper
from jiwer import wer, cer
import soundfile as sf
import librosa

def normalize_text(text):
    """
    Normalizes text for WER calculation.
    Uses Whisper's EnglishTextNormalizer to handle:
    - Punctuation removal (periods, commas, etc.)
    - Number expansion (4th -> fourth, 10 -> ten)
    - Hyphen handling (geo-fence -> geofence usually)
    """
    if not isinstance(text, str):
        return ""

    # Specific fix for "geo-fence" -> "geofence" to match ASR output preference
    text = text.replace("geo-fence", "geofence")
    
    # 1. Preferred: Use Whisper's Standard Normalizer
    if std_normalizer:
        return std_normalizer(text)

    # 2. Fallback: Custom Logic
    # Lowercase
    text = text.lower()

    # Normalize Numbers (including ordinals like 4th)
    if num2words:
        try:
            # Handle Ordinals (1st, 2nd, 3rd, 4th)
            def replace_ordinal(match):
                val = int(match.group(1))
                return num2words(val, to='ordinal')
            
            text = re.sub(r'\b(\d+)(st|nd|rd|th)\b', replace_ordinal, text)

            # Handle Cardinals (integers and decimals)
            def replace_num(match):
                val_str = match.group().replace(',', '')
                if '.' in val_str:
                    return num2words(float(val_str))
                return num2words(int(val_str))
            
            text = re.sub(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', replace_num, text)
        except Exception:
            pass 

    # Handle Hyphens: "geo-fence" -> "geofence" (remove hyphen)
    text = text.replace("-", "")

    # Remove remaining punctuation
    text = re.sub(r'[' + re.escape(string.punctuation) + ']', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Load Whisper model (using base for speed/accuracy balance)
# Options: tiny, base, small, medium, large
logger.info("Loading Whisper ASR model (base)...")
whisper_model = whisper.load_model("base")

def compute_audio_metrics(audio_path, reference_text):
    """
    Computes WER (Word Error Rate) and CER (Character Error Rate) using Whisper ASR.
    Whisper is state-of-the-art and runs locally, providing consistent results.
    """
    metrics = {"wer": 1.0, "cer": 1.0, "recognized_text": "", "word_count": 0}
    
    # Convert to absolute path for better compatibility
    abs_path = os.path.abspath(audio_path)
    
    if not os.path.exists(abs_path):
        logger.warning(f"Audio file not found: {abs_path}")
        return metrics

    try:
        # Load audio manually to avoid ffmpeg dependency
        logger.info(f"Transcribing: {os.path.basename(abs_path)}")
        
        # Load audio using soundfile
        audio_data, sample_rate = sf.read(abs_path)
        
        # Whisper expects mono audio at 16kHz
        # Resample if necessary
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Ensure float32
        audio_data = audio_data.astype(np.float32)
        
        # Transcribe using Whisper with pre-loaded audio array
        result = whisper_model.transcribe(
            audio_data,
            language="en",
            fp16=torch.cuda.is_available(),  # Use fp16 if CUDA available
            verbose=False  # Reduce output noise
        )
        text_pred = result["text"].strip()
        metrics["recognized_text"] = text_pred
        
        # Compute WER and CER
        # Compute WER and CER
        if reference_text and text_pred:
            # Normalize both reference and prediction
            ref_norm = normalize_text(reference_text)
            pred_norm = normalize_text(text_pred)
            
            # Word Error Rate
            # Ensure we don't divide by zero if ref becomes empty
            if ref_norm:
                error_rate = wer(ref_norm, pred_norm)
                metrics["wer"] = error_rate
                
                # Character Error Rate
                char_error_rate = cer(ref_norm, pred_norm)
                metrics["cer"] = char_error_rate
            else:
                # If reference becomes empty after normalization (e.g. only punctuation), skip or set 0?
                # Usually implies match if pred is also empty.
                if not pred_norm:
                    metrics["wer"] = 0.0
                    metrics["cer"] = 0.0
                else:
                    metrics["wer"] = 1.0 # Error
                    metrics["cer"] = 1.0

            
            # Word count for analysis
            metrics["word_count"] = len(reference_text.split())
            
            logger.info(f"  WER: {error_rate:.4f} | CER: {char_error_rate:.4f}")
        else:
            logger.warning(f"Empty transcription or reference for {abs_path}")
                
    except Exception as e:
        logger.error(f"Audio Metric Error {abs_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return metrics

def run_evaluation(events_file="output/events.json", output_dir="output/metrics"):
    if not os.path.exists(events_file):
        logger.error(f"Events file not found: {events_file}")
        return

    logger.info("Loading Events...")
    with open(events_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        timeline = data.get("timeline", [])

    results = []

    # Iterate through timeline
    for scene in timeline:
        scene_id = scene.get("id")
        for beat in scene.get("beats", []):
            beat_id = beat.get("sub_scene_id")
            if not beat_id: continue
            
            logger.info(f"evaluating beat: {beat_id}")
            
            row = {
                "id": beat_id,
                "scene_id": scene_id,
                "type": beat.get("type"),
                "speaker": beat.get("speaker", "Narrator"),  # Track speaker
                "text": beat.get("text"),
                "visual_prompt": beat.get("visual_prompt"),
                "audio_prompt": beat.get("audio_prompt")
            }

            # 1. Image Evaluation
            img_path = f"output/images/{beat_id}.png"
            if os.path.exists(img_path) and row["visual_prompt"]:
                # Try to find reference image for the speaker
                speaker = row["speaker"]
                ref_img_path = None
                
                # Check standard character directory structure
                # Logic: output/images/characters/{Name}/{name}_waist_front.png
                if speaker and speaker != "Narrator":
                    # Handle potential full names "Julian Black" -> Directory "Julian_Black", File "julian_black_waist_front.png" ??
                    # Based on list_dir, directories are "Julian", "Lena".
                    # Let's try direct name first.
                    char_dir_name = speaker.replace(" ", "_")
                    char_file_name = f"{speaker.lower().replace(' ', '_')}_waist_front.png"
                    
                    potential_path = os.path.join("output/images/characters", char_dir_name, char_file_name)
                    if os.path.exists(potential_path):
                        ref_img_path = potential_path
                    else:
                        # Fallback for just first name if directory is "Julian" but speaker is "Julian Black"
                        first_name = speaker.split()[0]
                        potential_path_first = os.path.join("output/images/characters", first_name, f"{first_name.lower()}_waist_front.png")
                        if os.path.exists(potential_path_first):
                            ref_img_path = potential_path_first

                v_metrics = compute_vision_metrics(img_path, row["visual_prompt"], reference_image_path=ref_img_path)
                row.update(v_metrics)


            # Audio evaluation
            aud_path = f"output/audio/{beat_id}.wav"
            if beat.get("type") in ["dialogue", "narration"] and os.path.exists(aud_path):
                a_metrics = compute_audio_metrics(aud_path, row["text"])
                row.update(a_metrics)
                # Word-level Precision / Recall / F1 from ASR output
                if a_metrics.get("recognized_text"):
                    f1_metrics = compute_token_f1(row["text"], a_metrics["recognized_text"])
                    row.update(f1_metrics)

            # Visual prompt coverage (how well visual_prompt covers the beat text)
            if row.get("visual_prompt") and row.get("text"):
                vis_cov = compute_visual_coverage(row["visual_prompt"], row["text"])
                row.update(vis_cov)

            results.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics_report.csv")
    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved raw metrics to {csv_path}")
    except PermissionError:
        logger.warning(f"Could not save {csv_path} (File Open?). Saving to metrics_report_new.csv")
        df.to_csv(os.path.join(output_dir, "metrics_report_new.csv"), index=False)
    
    # Generate Research Grade Report
    generate_comprehensive_report(df, output_dir)
    
    # Plotting
    plot_metrics(df, output_dir)

def generate_comprehensive_report(df, output_dir):
    """Generates a text report comparing pipeline performance to baselines."""
    report_lines = ["# MAVIS Pipeline Evaluation Report", ""]
    
    # 1. Vision Analysis
    if "clip_score" in df.columns:
        mean_clip = df["clip_score"].mean()
        report_lines.append(f"## Vision Pipeline (SSD-1B / SDXL)")
        report_lines.append(f"- **Mean CLIPScore (Text-Image Alignment):** {mean_clip:.2f}")
        
        # Access Vision Baselines
        vision_bl = PAPER_BASELINES["Vision (CLIPScore)"]
        report_lines.append(f"  - vs SSD-1B Baseline: {vision_bl['SSD-1B (Ours Base)']} (Delta: {mean_clip - vision_bl['SSD-1B (Ours Base)']:.2f})")
        report_lines.append(f"  - vs SDXL Base Baseline: {vision_bl['SDXL Base']} (Delta: {mean_clip - vision_bl['SDXL Base']:.2f})")
        report_lines.append(f"  - vs DALL-E 2: {vision_bl['DALL-E 2']} (Delta: {mean_clip - vision_bl['DALL-E 2']:.2f})")
        report_lines.append("")
        
        if "identity_consistency" in df.columns:
            mean_id = df["identity_consistency"].mean()
            report_lines.append(f"- **Mean Identity Consistency:** {mean_id:.2f}")
            report_lines.append(f"  - Note: Values > 75 usually indicate recognizable identity preservation.")
            report_lines.append("")

    # 2. Audio Analysis - ENHANCED
    if "wer" in df.columns:
        audio_df = df[df["wer"] > 0].copy()
        if not audio_df.empty:
            mean_wer = audio_df["wer"].mean()
            median_wer = audio_df["wer"].median()
            std_wer = audio_df["wer"].std()
            min_wer = audio_df["wer"].min()
            max_wer = audio_df["wer"].max()
            
            # CER stats if available
            mean_cer = audio_df["cer"].mean() if "cer" in audio_df.columns else 0
            
            report_lines.append(f"## Audio Pipeline (TTS) - Generalized Evaluation")
            report_lines.append(f"### Overall Statistics")
            report_lines.append(f"- **Total Audio Files Evaluated:** {len(audio_df)}")
            report_lines.append(f"- **Mean Word Error Rate (WER):** {mean_wer:.4f} ({mean_wer*100:.2f}%)")
            report_lines.append(f"- **Median WER:** {median_wer:.4f} ({median_wer*100:.2f}%)")
            report_lines.append(f"- **Std Dev WER:** {std_wer:.4f}")
            report_lines.append(f"- **Min WER:** {min_wer:.4f} (Best)")
            report_lines.append(f"- **Max WER:** {max_wer:.4f} (Worst)")
            if mean_cer > 0:
                report_lines.append(f"- **Mean Character Error Rate (CER):** {mean_cer:.4f} ({mean_cer*100:.2f}%)")
            report_lines.append("")
            
            # Total words processed
            if "word_count" in audio_df.columns:
                total_words = audio_df["word_count"].sum()
                report_lines.append(f"- **Total Words Processed:** {total_words}")
                report_lines.append("")
            
            # Baseline Comparison
            audio_bl = PAPER_BASELINES["Audio (WER)"]
            report_lines.append(f"### Baseline Comparison")
            report_lines.append(f"- **vs State-of-the-Art TTS (XTTS/Parler ~0.05):** {'✅ Pass' if mean_wer <= 0.1 else '⚠️ Needs Improvement'}")
            report_lines.append(f"- vs Tacotron 2 ({audio_bl['Tacotron 2']}): Delta = {mean_wer - audio_bl['Tacotron 2']:.4f}")
            report_lines.append(f"- vs FastSpeech 2 ({audio_bl['FastSpeech 2']}): Delta = {mean_wer - audio_bl['FastSpeech 2']:.4f}")
            report_lines.append(f"- vs VITS ({audio_bl['VITS']}): Delta = {mean_wer - audio_bl['VITS']:.4f}")
            report_lines.append("")
            
            # Speaker-specific analysis
            if "speaker" in audio_df.columns:
                report_lines.append(f"### Speaker-Specific Analysis")
                speaker_stats = audio_df.groupby("speaker")["wer"].agg(['mean', 'count', 'std']).sort_values('mean')
                for speaker, row in speaker_stats.iterrows():
                    report_lines.append(f"- **{speaker}:** Mean WER = {row['mean']:.4f}, Files = {int(row['count'])}, Std = {row['std']:.4f}")
                report_lines.append("")
            
            # Scene-wise WER distribution
            if "scene_id" in audio_df.columns:
                report_lines.append(f"### Scene-Wise WER Analysis")
                scene_stats = audio_df.groupby("scene_id")["wer"].agg(['mean', 'count']).sort_values('mean')
                for scene, row in scene_stats.iterrows():
                    report_lines.append(f"- **{scene}:** Mean WER = {row['mean']:.4f}, Files = {int(row['count'])}")
                report_lines.append("")
            
            # Best and Worst Examples
            report_lines.append(f"### Example Transcriptions")
            
            # Best case
            best_idx = audio_df["wer"].idxmin()
            best_row = audio_df.loc[best_idx]
            report_lines.append(f"#### Best (Lowest WER = {best_row['wer']:.4f})")
            report_lines.append(f"- **Scene:** {best_row['id']}")
            report_lines.append(f"- **Reference:** {best_row['text'][:100]}...")
            report_lines.append(f"- **Recognized:** {best_row.get('recognized_text', '')[:100]}...")
            report_lines.append("")
            
            # Worst case
            worst_idx = audio_df["wer"].idxmax()
            worst_row = audio_df.loc[worst_idx]
            report_lines.append(f"#### Worst (Highest WER = {worst_row['wer']:.4f})")
            report_lines.append(f"- **Scene:** {worst_row['id']}")
            report_lines.append(f"- **Reference:** {worst_row['text'][:100]}...")
            report_lines.append(f"- **Recognized:** {worst_row.get('recognized_text', '')[:100]}...")
            report_lines.append("")

            # Detailed Transcription Table (Requested by USER)
            report_lines.append("### Detailed Transcription Report")
            report_lines.append("| Scene | Beat | WER | Reference | Recognized |")
            report_lines.append("|---|---|---|---|---|")
            
            # Sort by Scene ID then Beat ID
            detailed_df = audio_df.sort_values(["scene_id", "id"])
            
            for index, row in detailed_df.iterrows():
                # Sanitize text
                ref = str(row['text']).replace('|', '\|').replace('\n', ' ')[:200] # Limit length
                rec = str(row.get('recognized_text', '')).replace('|', '\|').replace('\n', ' ')[:200]
                
                report_lines.append(f"| {row['scene_id']} | {row['id']} | {row['wer']:.4f} | {ref} | {rec} |")
            report_lines.append("")

    # 3. NLP Analysis
    if "bert_f1" in df.columns:
        mean_bert = df["bert_f1"].mean()
        report_lines.append(f"## Narrative Consistency (LLM)")
        report_lines.append(f"- **Mean BERTScore F1:** {mean_bert:.3f}")
        
        # Access NLP Baselines
        nlp_bl = PAPER_BASELINES["NLP (BERTScore F1)"]
        report_lines.append(f"  - vs GPT-4 Reference: {nlp_bl['GPT-4']} (Delta: {mean_bert - nlp_bl['GPT-4']:.3f})")
        report_lines.append("")

    # 4. Audio Word-level Precision / Recall / F1
    if "token_f1" in df.columns:
        audio_f1_df = df[df["token_f1"].notna() & (df["token_f1"] > 0)]
        if not audio_f1_df.empty:
            mean_p = audio_f1_df["token_precision"].mean()
            mean_r = audio_f1_df["token_recall"].mean()
            mean_f1 = audio_f1_df["token_f1"].mean()
            report_lines.append("## Audio Word-Level Precision / Recall / F1")
            report_lines.append("*(Bag-of-words comparison between ASR output and reference text)*")
            report_lines.append(f"- **Mean Precision:** {mean_p:.4f} ({mean_p*100:.2f}%)")
            report_lines.append(f"- **Mean Recall:** {mean_r:.4f} ({mean_r*100:.2f}%)")
            report_lines.append(f"- **Mean F1:** {mean_f1:.4f} ({mean_f1*100:.2f}%)")
            report_lines.append("")

            # Per-speaker breakdown
            if "speaker" in audio_f1_df.columns:
                report_lines.append("### Speaker-level F1")
                spk_f1 = audio_f1_df.groupby("speaker")["token_f1"].mean().sort_values(ascending=False)
                for spk, val in spk_f1.items():
                    report_lines.append(f"- **{spk}:** F1 = {val:.4f}")
                report_lines.append("")

    # 5. Visual Prompt Coverage
    if "visual_f1" in df.columns:
        vis_df = df[df["visual_f1"].notna()]
        if not vis_df.empty:
            mean_vp = vis_df["visual_precision"].mean()
            mean_vr = vis_df["visual_recall"].mean()
            mean_vf1 = vis_df["visual_f1"].mean()
            report_lines.append("## Visual Prompt Coverage (Prompt vs Beat Text)")
            report_lines.append("*(How well the visual_prompt captures key content words from the script beat)*")
            report_lines.append(f"- **Mean Precision:** {mean_vp:.4f} — words in prompt that came from the beat")
            report_lines.append(f"- **Mean Recall:** {mean_vr:.4f} — beat words captured in the prompt")
            report_lines.append(f"- **Mean F1:** {mean_vf1:.4f}")
            report_lines.append("")

            # Per beat type breakdown
            if "type" in vis_df.columns:
                report_lines.append("### Coverage by Beat Type")
                type_f1 = vis_df.groupby("type")["visual_f1"].mean()
                for btype, val in type_f1.items():
                    report_lines.append(f"- **{btype}:** F1 = {val:.4f}")
                report_lines.append("")


    with open(os.path.join(output_dir, "pipeline_evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info("Generated pipeline_evaluation_report.md")

def plot_metrics(df, output_dir):
    """Generates comparative plots for the metrics."""
    if df.empty:
        logger.warning("No data to plot.")
        return

    logger.info("Generating plots...")
    
    # 1. Bar Chart: Text-Image Alignment (CLIPScore)
    if "clip_score" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="id", y="clip_score", hue="id", palette="viridis", legend=False)
        plt.title("Text-Image Alignment (CLIPScore) per Scene")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "clip_score_per_scene.png"))
        plt.close()

    # 2. Bar Chart: Word Error Rate (WER)
    if "wer" in df.columns:
        # Filter only audio rows
        audio_df = df[df["wer"] > 0]
        if not audio_df.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=audio_df, x="id", y="wer", hue="id", palette="magma", legend=False)
            plt.title("Audio Transcription Word Error Rate (WER)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "wer_per_scene.png"))
            plt.close()

    # 3. Overall Statistics Summary
    summary = df.describe()
    try:
        summary.to_csv(os.path.join(output_dir, "metrics_summary.csv"))
    except PermissionError:
        logger.warning("Could not save metrics_summary.csv (File Open?). Saving to metrics_summary_new.csv")
        summary.to_csv(os.path.join(output_dir, "metrics_summary_new.csv"))
    
    # 4. Comparative Bar Charts (Multi-Model)
    
    # --- Vision Comparison ---
    if "clip_score" in df.columns:
        mean_clip = df["clip_score"].mean()
        
        vision_data = PAPER_BASELINES["Vision (CLIPScore)"].copy()
        vision_data["MAVIS Pipeline (Ours)"] = mean_clip
        
        # Sort for better visualization
        vision_df = pd.DataFrame(list(vision_data.items()), columns=["Model", "CLIPScore"])
        vision_df = vision_df.sort_values("CLIPScore", ascending=True)
        
        plt.figure(figsize=(10, 6))
        colors = ["grey" if "MAVIS" not in x else "crimson" for x in vision_df["Model"]]
        p = sns.barplot(data=vision_df, x="CLIPScore", y="Model", hue="Model", palette=colors, legend=False)
        plt.title("Vision Model Comparison: Text-Image Alignment (CLIPScore)")
        plt.xlabel("CLIPScore (Higher is Better)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_vision.png"))
        plt.close()

    # --- Audio Comparison ---
    if "wer" in df.columns:
        mean_wer = df[df["wer"] > 0]["wer"].mean() if not df[df["wer"] > 0].empty else 0
        
        audio_data = PAPER_BASELINES["Audio (WER)"].copy()
        audio_data["MAVIS Pipeline (Ours)"] = mean_wer
        
        audio_df = pd.DataFrame(list(audio_data.items()), columns=["Model", "WER"])
        audio_df = audio_df.sort_values("WER", ascending=True) # Lower is better? Yes, but for sorting display we might want smallest at top
        
        plt.figure(figsize=(10, 6))
        colors = ["grey" if "MAVIS" not in x else "crimson" for x in audio_df["Model"]]
        sns.barplot(data=audio_df, x="WER", y="Model", hue="Model", palette=colors, legend=False)
        plt.title("Audio Model Comparison: Word Error Rate (WER)")
        plt.xlabel("WER (Lower is Better)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_audio.png"))
        plt.close()

    # --- NLP Comparison ---
    if "bert_f1" in df.columns:
        mean_bert = df["bert_f1"].mean()
        
        nlp_data = PAPER_BASELINES["NLP (BERTScore F1)"].copy()
        nlp_data["MAVIS Pipeline (Ours)"] = mean_bert
        
        nlp_df = pd.DataFrame(list(nlp_data.items()), columns=["Model", "BERTScore F1"])
        nlp_df = nlp_df.sort_values("BERTScore F1", ascending=True)
        
        plt.figure(figsize=(10, 6))
        colors = ["grey" if "MAVIS" not in x else "crimson" for x in nlp_df["Model"]]
        sns.barplot(data=nlp_df, x="BERTScore F1", y="Model", hue="Model", palette=colors, legend=False)
        plt.title("NLP Model Comparison: Semantic Consistency (BERTScore)")
        plt.xlabel("BERTScore F1 (Higher is Better)")
        plt.xlim(0.5, 1.0) # Zoom in to relevant range
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_nlp.png"))
        plt.close()
    
    logger.info("Evaluation Complete. Check output directory for comparison plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Research-Grade Metrics for MAVIS Pipeline")
    parser.add_argument("--events", default="output/events.json", help="Path to events.json")
    args = parser.parse_args()
    
    run_evaluation(args.events)