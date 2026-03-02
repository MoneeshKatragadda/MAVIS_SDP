
import os
import argparse
import itertools
import torch
import clip
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MAVIS_CONSISTENCY")


def load_clip(device):
    logger.info(f"Loading CLIP (ViT-B/32) on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def encode_image(model, preprocess, image_path, device):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]


def compute_consistency(char_dir, model, preprocess, device):
    results = {}

    for char_name in sorted(os.listdir(char_dir)):
        char_path = os.path.join(char_dir, char_name)
        if not os.path.isdir(char_path):
            continue

        image_files = [
            os.path.join(char_path, f)
            for f in sorted(os.listdir(char_path))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
               and os.path.getsize(os.path.join(char_path, f)) > 0
        ]

        if len(image_files) < 2:
            logger.warning(f"  {char_name}: Only {len(image_files)} image(s) found — need at least 2 to compare.")
            continue

        logger.info(f"  Processing '{char_name}' — {len(image_files)} views found")

        # Encode all views
        embeddings = {}
        for img_path in image_files:
            view_name = os.path.splitext(os.path.basename(img_path))[0]
            try:
                embeddings[view_name] = encode_image(model, preprocess, img_path, device)
            except Exception as e:
                logger.warning(f"    Skipping {view_name}: {e}")

        if len(embeddings) < 2:
            logger.warning(f"  {char_name}: Not enough encodable images.")
            continue

        # Pairwise cosine similarity
        pairs = list(itertools.combinations(embeddings.keys(), 2))
        scores = []
        pair_details = []

        for v1, v2 in pairs:
            e1, e2 = embeddings[v1], embeddings[v2]
            sim = float(np.dot(e1, e2) * 100.0)  # CLIP features already L2-normalised
            scores.append(sim)
            pair_details.append((v1, v2, sim))

        mean_score = float(np.mean(scores))
        min_score  = float(np.min(scores))
        max_score  = float(np.max(scores))
        std_score  = float(np.std(scores))

        results[char_name] = {
            "num_views": len(embeddings),
            "mean_consistency": round(mean_score, 2),
            "min_consistency":  round(min_score,  2),
            "max_consistency":  round(max_score,  2),
            "std_consistency":  round(std_score,  2),
            "pair_details": sorted(pair_details, key=lambda x: x[2])   # worst first
        }

    return results


def print_report(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    report_lines = [
        "# Character Consistency Report",
        "(CLIP ViT-B/32 pairwise cosine similarity × 100)",
        "",
        "Interpretation:",
        "  > 80  — Excellent identity consistency",
        "  70-80 — Good consistency",
        "  60-70 — Moderate (some drift between views)",
        "  < 60  — Poor consistency",
        "",
    ]

    print("\n" + "=" * 60)
    print("  CHARACTER CONSISTENCY REPORT")
    print("  (CLIP pairwise cosine similarity × 100)")
    print("=" * 60)

    for char_name, data in results.items():
        status = (
            "✅ Excellent" if data["mean_consistency"] > 80 else
            "✅ Good"      if data["mean_consistency"] > 70 else
            "⚠️  Moderate" if data["mean_consistency"] > 60 else
            "❌ Poor"
        )

        print(f"\n  [{char_name}]  {status}")
        print(f"    Views:        {data['num_views']}")
        print(f"    Mean Score:   {data['mean_consistency']:.2f}")
        print(f"    Range:        {data['min_consistency']:.2f} – {data['max_consistency']:.2f}")
        print(f"    Std Dev:      {data['std_consistency']:.2f}")
        print(f"    --- Worst pairs (lowest consistency) ---")
        for v1, v2, sim in data["pair_details"][:3]:
            print(f"      {v1}  ↔  {v2}:  {sim:.2f}")
        print(f"    --- Best pairs (highest consistency) ---")
        for v1, v2, sim in data["pair_details"][-3:]:
            print(f"      {v1}  ↔  {v2}:  {sim:.2f}")

        # Build report lines
        report_lines.append(f"## {char_name}  —  {status}")
        report_lines.append(f"- **Views evaluated:** {data['num_views']}")
        report_lines.append(f"- **Mean Consistency Score:** {data['mean_consistency']:.2f}")
        report_lines.append(f"- **Range:** {data['min_consistency']:.2f} – {data['max_consistency']:.2f}  |  Std: {data['std_consistency']:.2f}")
        report_lines.append("")
        report_lines.append("| View A | View B | Score |")
        report_lines.append("|--------|--------|-------|")
        for v1, v2, sim in sorted(data["pair_details"], key=lambda x: x[2]):
            report_lines.append(f"| {v1} | {v2} | {sim:.2f} |")
        report_lines.append("")

    print("\n" + "=" * 60)

    report_path = os.path.join(out_dir, "character_consistency_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute character consistency scores using CLIP.")
    parser.add_argument("--char_dir", default="output/images/characters",
                        help="Root directory of character image folders")
    parser.add_argument("--out", default="output/metrics",
                        help="Output directory for the report")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip(device)

    logger.info(f"Scanning characters in: {args.char_dir}")
    results = compute_consistency(args.char_dir, model, preprocess, device)

    if not results:
        logger.error("No character data found. Check --char_dir path.")
        return

    print_report(results, args.out)


if __name__ == "__main__":
    main()
