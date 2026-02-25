"""Batch testing CLI for the YouTube Thumbnail Generator.

Usage:
  python batch_test.py batch prompts.txt [--output report.json]
  python batch_test.py generalize [--output report.json]
  python batch_test.py stability "prompt text" [--n 3] [--output report.json]
  python batch_test.py compare "prompt text" [--output report.json]
"""

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from agent import generate_thumbnail

load_dotenv()


# ---------------------------------------------------------------------------
# Built-in generalization prompt suite
# ---------------------------------------------------------------------------

GENERALIZATION_PROMPTS = [
    {"category": "Tech",      "prompt": "How to build a REST API in Python"},
    {"category": "Gaming",    "prompt": "Top 10 Minecraft survival tips for beginners"},
    {"category": "Cooking",   "prompt": "30-minute dinner recipes anyone can make"},
    {"category": "Finance",   "prompt": "How to invest $1000 for beginners in 2024"},
    {"category": "Fitness",   "prompt": "Full body workout at home with no equipment"},
    {"category": "Education", "prompt": "Learn machine learning from scratch in 30 days"},
    {"category": "Travel",    "prompt": "Budget backpacking tips for Southeast Asia"},
    {"category": "Business",  "prompt": "How to start a dropshipping business with $0"},
    {"category": "Science",   "prompt": "Why black holes are stranger than you think"},
    {"category": "Lifestyle", "prompt": "Morning routine that changed my life forever"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_similarity(bytes1: bytes, bytes2: bytes) -> float:
    """Pixel-level similarity: 1.0 = identical images, 0.0 = completely different.

    Resizes both images to 320×180 grayscale and returns 1 - mean_absolute_error.
    This gives an intuitive score: same-prompt re-runs from AI generators typically
    score 0.40–0.70 (high variance is expected); identical images score 1.0.
    """
    def _load(b: bytes) -> "np.ndarray":
        return (
            np.array(
                Image.open(io.BytesIO(b)).convert("L").resize((320, 180), Image.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
        )

    img1, img2 = _load(bytes1), _load(bytes2)
    mae = float(np.mean(np.abs(img1 - img2)))
    return round(1.0 - mae, 3)


def _extract_metrics(result: dict) -> dict:
    """Pull quality metrics from a generate_thumbnail result dict."""
    m = result.get("validation_metrics", {})
    return {
        "validation":     result.get("validation_result", "N/A"),
        "retries":        result.get("retry_count", 0),
        "overall_quality": m.get("overall_quality"),
        "ocr":            m.get("ocr_accuracy", {}).get("score"),
        "contrast":       m.get("contrast", {}).get("score"),
        "contrast_grade": m.get("contrast", {}).get("grade", "N/A"),
        "artifacts":      m.get("artifacts", {}).get("score"),
        "layout":         m.get("layout_stability", {}).get("score"),
        "output_path":    result.get("output_path", ""),
        "error":          result.get("error", ""),
    }


def _fmt(val, digits: int = 3) -> str:
    return f"{val:.{digits}f}" if val is not None else "N/A"


def _print_table(rows, headers):
    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep      = "  ".join("-" * w for w in col_widths)
    print(row_fmt.format(*headers))
    print(sep)
    for row in rows:
        print(row_fmt.format(*row))


def _pass_label(validation: str) -> str:
    return "PASS" if str(validation).startswith("PASS") else "FAIL"


def _save_report(path, data: dict):
    if not path:
        return
    data["generated_at"] = datetime.now().isoformat()
    out = Path(path)
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nReport saved to: {out}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_batch(args):
    """Run generate_thumbnail for each prompt in a text file (one per line)."""
    src = Path(args.file)
    if not src.exists():
        print(f"Error: {src} not found")
        sys.exit(1)

    prompts = [
        line.strip()
        for line in src.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not prompts:
        print("No prompts found in file (blank lines and # comments are ignored)")
        sys.exit(1)

    print(f"Batch test — {len(prompts)} prompt(s) from '{src}'\n")

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(prompts)}] {prompt}")
        print("=" * 60)
        result = generate_thumbnail(prompt)
        m = _extract_metrics(result)
        results.append({"prompt": prompt, **m})

    _print_batch_summary(results)
    _save_report(args.output, {"mode": "batch", "file": str(src), "results": results})


def cmd_generalize(args):
    """Run the built-in 10-category prompt suite to test topic generalization."""
    n = len(GENERALIZATION_PROMPTS)
    print(f"Generalization test — {n} prompts across {n} categories\n")

    results = []
    for i, item in enumerate(GENERALIZATION_PROMPTS, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{n}] [{item['category']}] {item['prompt']}")
        print("=" * 60)
        result = generate_thumbnail(item["prompt"])
        m = _extract_metrics(result)
        results.append({"category": item["category"], "prompt": item["prompt"], **m})

    print("\n\n=== GENERALIZATION REPORT ===\n")
    rows = [
        (
            r["category"],
            (r["prompt"][:40] + "...") if len(r["prompt"]) > 40 else r["prompt"],
            _fmt(r["overall_quality"]),
            _fmt(r["ocr"]),
            _fmt(r["contrast"]),
            r["contrast_grade"],
            _fmt(r["artifacts"]),
            _fmt(r["layout"]),
            str(r["retries"]),
            _pass_label(r["validation"]),
        )
        for r in results
    ]
    _print_table(
        rows,
        ["Category", "Prompt", "Quality", "OCR", "Contrast", "Grade", "Artifacts", "Layout", "Retries", "Status"],
    )

    passed    = sum(1 for r in results if _pass_label(r["validation"]) == "PASS")
    qualities = [r["overall_quality"] for r in results if r["overall_quality"] is not None]
    print(f"\nPass rate:    {passed}/{n}  ({100 * passed // n}%)")
    if qualities:
        print(f"Avg quality:  {_fmt(sum(qualities) / len(qualities))}")

    _save_report(args.output, {"mode": "generalize", "results": results})


def cmd_stability(args):
    """Generate the same prompt N times and measure consistency of outputs."""
    prompt = args.prompt
    n      = args.n
    print(f"Stability test — '{prompt}' × {n} runs\n")

    runs = []
    for i in range(1, n + 1):
        print(f"\n{'='*60}")
        print(f"Run {i}/{n}")
        print("=" * 60)
        result = generate_thumbnail(prompt)
        m      = _extract_metrics(result)
        runs.append({
            "run":         i,
            "image_bytes": result.get("image_bytes", b""),
            **m,
        })

    # Pairwise image similarity across all runs
    all_bytes    = [r["image_bytes"] for r in runs if r["image_bytes"]]
    similarities = [
        _image_similarity(all_bytes[i], all_bytes[j])
        for i in range(len(all_bytes))
        for j in range(i + 1, len(all_bytes))
    ]

    print("\n\n=== STABILITY REPORT ===\n")
    rows = [
        (
            str(r["run"]),
            _fmt(r["overall_quality"]),
            _fmt(r["contrast"]),
            r["contrast_grade"],
            _fmt(r["artifacts"]),
            _fmt(r["layout"]),
            _pass_label(r["validation"]),
        )
        for r in runs
    ]
    _print_table(rows, ["Run", "Quality", "Contrast", "Grade", "Artifacts", "Layout", "Status"])

    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        print(f"\nImage similarity (1=identical, 0=different):")
        print(f"  Mean={_fmt(avg_sim)}  Min={_fmt(min(similarities))}  Max={_fmt(max(similarities))}")
        print(f"  (AI generators typically score 0.40–0.70 for same-prompt re-runs)")

    qualities = [r["overall_quality"] for r in runs if r["overall_quality"] is not None]
    if qualities:
        avg_q = sum(qualities) / len(qualities)
        std_q = (sum((q - avg_q) ** 2 for q in qualities) / len(qualities)) ** 0.5
        print(f"\nQuality score:  Mean={_fmt(avg_q)}  StdDev={_fmt(std_q)}")

    report_runs = [{k: v for k, v in r.items() if k != "image_bytes"} for r in runs]
    _save_report(
        args.output,
        {"mode": "stability", "prompt": prompt, "n": n, "similarities": similarities, "runs": report_runs},
    )


def cmd_compare(args):
    """Compare Strategy A (Gemini-enriched) vs Strategy B (direct Imagen prompt)."""
    prompt = args.prompt
    print(f"Strategy comparison — '{prompt}'\n")

    strategies = [
        ("enriched", "Strategy A: Gemini-Enriched"),
        ("direct",   "Strategy B: Direct Imagen"),
    ]

    results = {}
    for key, label in strategies:
        print(f"\n{'='*60}")
        print(label)
        print("=" * 60)
        result = generate_thumbnail(prompt, strategy=key)
        results[key] = {
            "label":       label,
            "metrics":     _extract_metrics(result),
            "image_bytes": result.get("image_bytes", b""),
            "image_prompt": result.get("image_prompt", ""),
        }

    # Similarity between the two strategy outputs
    ba = results["enriched"]["image_bytes"]
    bd = results["direct"]["image_bytes"]
    sim = _image_similarity(ba, bd) if ba and bd else None

    print("\n\n=== STRATEGY COMPARISON REPORT ===\n")
    rows = [
        (
            results[k]["label"],
            _fmt(results[k]["metrics"]["overall_quality"]),
            _fmt(results[k]["metrics"]["ocr"]),
            _fmt(results[k]["metrics"]["contrast"]),
            results[k]["metrics"]["contrast_grade"],
            _fmt(results[k]["metrics"]["artifacts"]),
            _fmt(results[k]["metrics"]["layout"]),
            str(results[k]["metrics"]["retries"]),
            _pass_label(results[k]["metrics"]["validation"]),
        )
        for k in ("enriched", "direct")
    ]
    _print_table(
        rows,
        ["Strategy", "Quality", "OCR", "Contrast", "Grade", "Artifacts", "Layout", "Retries", "Status"],
    )

    if sim is not None:
        print(f"\nVisual similarity between strategies: {_fmt(sim)}")
        if sim > 0.85:
            print("  → Very similar — enrichment has minimal visual impact for this prompt")
        elif sim > 0.65:
            print("  → Moderately different — enrichment noticeably changes the composition")
        else:
            print("  → Very different — enrichment significantly alters the visual result")

    report = {
        "mode":       "compare",
        "prompt":     prompt,
        "similarity": sim,
        "strategies": {
            k: {"label": v["label"], "metrics": v["metrics"], "image_prompt": v["image_prompt"]}
            for k, v in results.items()
        },
    }
    _save_report(args.output, report)


# ---------------------------------------------------------------------------
# Shared summary printer
# ---------------------------------------------------------------------------

def _print_batch_summary(results):
    print("\n\n=== BATCH SUMMARY ===\n")
    rows = [
        (
            (r["prompt"][:45] + "...") if len(r["prompt"]) > 45 else r["prompt"],
            _fmt(r["overall_quality"]),
            _fmt(r["ocr"]),
            _fmt(r["contrast"]),
            r["contrast_grade"],
            _fmt(r["artifacts"]),
            _fmt(r["layout"]),
            str(r["retries"]),
            _pass_label(r["validation"]),
        )
        for r in results
    ]
    _print_table(
        rows,
        ["Prompt", "Quality", "OCR", "Contrast", "Grade", "Artifacts", "Layout", "Retries", "Status"],
    )
    passed    = sum(1 for r in results if _pass_label(r["validation"]) == "PASS")
    qualities = [r["overall_quality"] for r in results if r["overall_quality"] is not None]
    print(f"\nPass rate:   {passed}/{len(results)}")
    if qualities:
        print(f"Avg quality: {_fmt(sum(qualities) / len(qualities))}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch testing CLI for the YouTube Thumbnail Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_test.py batch prompts.txt --output batch_report.json
  python batch_test.py generalize --output generalize_report.json
  python batch_test.py stability "Python tips" --n 3 --output stability_report.json
  python batch_test.py compare "Python tips" --output compare_report.json
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_batch = sub.add_parser("batch", help="Run prompts from a text file (one per line)")
    p_batch.add_argument("file", help="Path to text file with one prompt per line")
    p_batch.add_argument("--output", "-o", metavar="FILE", help="Save JSON report")

    p_gen = sub.add_parser("generalize", help="Run built-in 10-category prompt suite")
    p_gen.add_argument("--output", "-o", metavar="FILE", help="Save JSON report")

    p_stab = sub.add_parser("stability", help="Generate same prompt N times, measure consistency")
    p_stab.add_argument("prompt", help="The prompt to repeat")
    p_stab.add_argument("--n", type=int, default=3, metavar="N", help="Number of runs (default: 3)")
    p_stab.add_argument("--output", "-o", metavar="FILE", help="Save JSON report")

    p_cmp = sub.add_parser("compare", help="Compare enriched vs direct generation strategies")
    p_cmp.add_argument("prompt", help="The prompt to test both strategies on")
    p_cmp.add_argument("--output", "-o", metavar="FILE", help="Save JSON report")

    args = parser.parse_args()
    {"batch": cmd_batch, "generalize": cmd_generalize, "stability": cmd_stability, "compare": cmd_compare}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
