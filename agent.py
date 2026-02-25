import base64
import io
import json
import os
import re
import uuid
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv

import numpy as np

try:
    import pytesseract

    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False

from PIL import Image, ImageFilter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from google import genai
from google.genai import types

load_dotenv()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ThumbnailState(TypedDict):
    user_prompt: str  # Original user input
    topic_analysis: str  # Gemini's structured topic breakdown
    thumbnail_text: str  # Short punchy title to overlay on image (2–6 words)
    image_prompt: str  # Optimized Imagen prompt
    image_bytes: bytes  # Raw PNG bytes from Imagen
    # --- LLM-decided text styling (from vision call on the real image) ---
    output_path: str  # Local path to saved thumbnail
    validator_feedback: str  # Actionable feedback written by the validator agent
    retry_count: int  # Number of times the validator has triggered a retry
    validation_result: str  # Validator verdict ("PASSED: ..." or "FAILED: ...")
    validation_metrics: (
        dict  # Quantitative quality scores (OCR, contrast, artifacts, layout)
    )
    error: str  # Non-empty if a node fails (non-retryable)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


def _gemini_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key,
    )


def _imagen_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Node 1: Analyze topic
# ---------------------------------------------------------------------------


def analyze_topic(state: ThumbnailState) -> ThumbnailState:
    """Gemini breaks down the topic and produces a short thumbnail title.
    On retries, validator feedback is injected so the agent can correct issues."""
    retry = state.get("retry_count", 0)
    if retry > 0:
        print(f"[1/5] Re-analyzing topic (retry {retry}/2) with Gemini...")
    else:
        print("[1/5] Analyzing topic with Gemini...")

    # Append validator feedback when retrying so the agent can fix the issues
    user_content = state["user_prompt"]
    if state.get("validator_feedback"):
        user_content += (
            "\n\n--- FEEDBACK FROM VALIDATOR AGENT (previous attempt was rejected) ---\n"
            + state["validator_feedback"]
            + "\nPlease fix the issues above in your revised thumbnail text and visual strategy."
        )

    messages = [
        SystemMessage(
            content=(
                "You are a YouTube thumbnail strategist. "
                "Given a video topic, respond in exactly this format:\n\n"
                "THUMBNAIL TEXT: <2–6 word punchy, all-caps title for the thumbnail>\n\n"
                "ANALYSIS:\n"
                "- Core subject (1-2 sentences)\n"
                "- Recommended visual style\n"
                "- Target audience\n"
                "- Emotional tone\n"
                "- Key visual elements\n\n"
                "The THUMBNAIL TEXT must be short, bold, and impactful — "
                "like a viral YouTube headline."
            )
        ),
        HumanMessage(content=user_content),
    ]
    response = _gemini_llm().invoke(messages)
    raw = response.content

    # Parse THUMBNAIL TEXT and ANALYSIS from the structured response
    thumbnail_text = ""
    analysis = raw
    for line in raw.splitlines():
        if line.upper().startswith("THUMBNAIL TEXT:"):
            thumbnail_text = line.split(":", 1)[1].strip()
            break
    if "ANALYSIS:" in raw:
        analysis = raw[raw.index("ANALYSIS:") + len("ANALYSIS:") :].strip()

    return {**state, "topic_analysis": analysis, "thumbnail_text": thumbnail_text}


# ---------------------------------------------------------------------------
# Node 2: Generate image prompt
# ---------------------------------------------------------------------------


def generate_image_prompt(state: ThumbnailState) -> ThumbnailState:
    """Gemini writes an optimized Imagen prompt for the thumbnail."""
    print("[2/5] Crafting image prompt with Gemini...")

    messages = [
        SystemMessage(
            content=(
                "You are an expert prompt engineer for Google Imagen 3 image generation. "
                "Write a single detailed prompt that produces a stunning YouTube thumbnail.\n\n"
                "Rules:\n"
                "- Target aspect ratio: 16:9 (wide cinematic format)\n"
                "- Bold, high-contrast, visually striking composition\n"
                "- Specify: lighting, color palette, camera angle, art style\n"
                "- Add quality boosters: 'highly detailed', 'professional photography', "
                "'4K resolution', 'cinematic lighting', 'sharp focus'\n"
                "- Do NOT include readable text or words in the scene\n"
                "- Output ONLY the prompt text, nothing else"
            )
        ),
        HumanMessage(
            content=(
                f"User's request:\n{state['user_prompt']}\n\n"
                f"Topic analysis:\n{state['topic_analysis']}"
            )
        ),
    ]
    response = _gemini_llm().invoke(messages)
    return {**state, "image_prompt": response.content.strip()}


def create_thumbnail_image(state: ThumbnailState) -> ThumbnailState:
    """
    Generates a thumbnail with the caption embedded via the image prompt.
    Gemini vision is used to enrich the prompt with style guidance beforehand.
    """

    # ------------------------------------------------------------------
    # Step 1: Enrich the image prompt with caption + style via Gemini
    # ------------------------------------------------------------------
    print("[3/5] Planning caption-integrated prompt with Gemini...")
    thum_nail = state[
        "thumbnail_text"
    ]  # to test validator failure modes, you can set this to something irrelevant like "BANANA" and see if the validator catches it
    style_messages = [
        SystemMessage(
            content=(
                "You are an expert YouTube thumbnail designer. "
                "Given a video topic and a short caption, write a single, detailed image generation prompt "
                "that instructs the model to render the caption TEXT visually embedded in the scene — "
                "e.g. as bold overlaid title text, a sign, a banner, or a stylized typographic element — "
                "not as an afterthought but as a designed part of the composition.\n\n"
                "Guidelines:\n"
                "- Describe the text style explicitly: font weight, color, placement (top/bottom), "
                "  outline or glow, contrast against background.\n"
                "- Keep the full prompt under 300 words.\n"
                "- Output ONLY the final prompt string, no explanation."
            )
        ),
        HumanMessage(
            content=(
                f"Video topic: {state['user_prompt']}\n"
                f'Caption to embed: "{thum_nail}"\n'
                f"Base image concept: {state['image_prompt']}"
            )
        ),
    ]

    try:
        enriched_response = _gemini_llm().invoke(style_messages)
        enriched_prompt = enriched_response.content.strip()
    except Exception as e:
        print(f"Prompt enrichment failed, falling back to base prompt: {e}")
        enriched_prompt = (
            f"{state['image_prompt']} "
            f"With bold overlaid title text reading \"{state['thumbnail_text']}\" "
            f"in large white letters with a dark outline at the bottom of the image."
        )

    # ------------------------------------------------------------------
    # Step 2: Generate image with caption baked in
    # ------------------------------------------------------------------
    print("[3/5] Generating thumbnail with Imagen (caption embedded)...")
    try:
        client = _imagen_client()
        response = client.models.generate_images(
            model="imagen-4.0-fast-generate-001",
            prompt=enriched_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",
                output_mime_type="image/png",
            ),
        )
        raw_bytes = response.generated_images[0].image.image_bytes

        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img = img.resize((1280, 720), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        return {**state, "image_bytes": image_bytes}
    except Exception as e:
        print(e)
        return {**state, "error": f"Imagen generation failed: {e}"}


def create_thumbnail_image_direct(state: ThumbnailState) -> ThumbnailState:
    """Strategy B: Generate image with a simple prompt — no Gemini enrichment step."""
    print("[3/5] Generating thumbnail with Imagen (direct prompt, no enrichment)...")
    prompt = (
        f"{state['image_prompt']} "
        f"With bold overlaid title text reading \"{state['thumbnail_text']}\" "
        "in large white letters with a dark shadow outline at the bottom of the image."
    )
    try:
        client = _imagen_client()
        response = client.models.generate_images(
            model="imagen-4.0-fast-generate-001",
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",  # ✅ allowed
                output_mime_type="image/png",
            ),
        )
        raw_bytes = response.generated_images[0].image.image_bytes

        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img = img.resize((1280, 720), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        return {**state, "image_bytes": image_bytes}
    except Exception as e:
        return {**state, "error": f"Imagen generation failed: {e}"}


MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Quantitative validation metric helpers
# ---------------------------------------------------------------------------


def _metric_ocr_accuracy(img: Image.Image, expected_text: str) -> dict:
    """
    Metric 1 — OCR exact-match accuracy (token-level F1).

    Justification:
      A thumbnail is only effective if viewers can actually read the intended title.
      Exact character matching is too strict for image-rendered text (font aliasing,
      partial occlusion by visual elements, decorative glyphs), so we use token-level
      F1 overlap between the Tesseract OCR output and the expected overlay text.
      The image is upscaled 2× before OCR to improve recognition on small glyphs.

    Threshold guidance:
      >= 0.80  excellent  — all key words legible
      0.50–0.79 acceptable — most words legible, some stylistic loss
      < 0.50  poor        — text is likely illegible or missing from the image
    """
    if not _HAS_TESSERACT:
        return {
            "score": None,
            "reason": "pytesseract not installed (pip install pytesseract + system tesseract-ocr)",
            "ocr_text": "",
            "expected_text": expected_text,
        }

    try:
        w, h = img.size
        # 2× upscale: small thumbnail text benefits greatly from higher resolution OCR
        upscaled = img.resize((w * 2, h * 2), Image.LANCZOS)
        ocr_raw = pytesseract.image_to_string(upscaled, config="--psm 3").strip()

        expected_tokens = set(expected_text.upper().split())
        ocr_tokens = set(ocr_raw.upper().split())

        if not expected_tokens:
            return {
                "score": 1.0,
                "ocr_text": ocr_raw,
                "expected_text": expected_text,
                "method": "token_F1",
            }

        common = expected_tokens & ocr_tokens
        precision = len(common) / len(ocr_tokens) if ocr_tokens else 0.0
        recall = len(common) / len(expected_tokens)
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "score": round(f1, 3),
            "ocr_text": ocr_raw,
            "expected_text": expected_text,
            "matched_tokens": sorted(common),
            "method": "token_F1",
        }
    except Exception as exc:
        return {
            "score": None,
            "reason": str(exc),
            "ocr_text": "",
            "expected_text": expected_text,
        }


def _metric_contrast_score(img: Image.Image) -> dict:
    """
    Metric 2 — Text contrast / legibility score (WCAG 2.1 contrast ratio).

    Justification:
      Low contrast between text and background is the single most common legibility
      failure in AI-generated thumbnails.  We compute the WCAG 2.1 relative-luminance
      contrast ratio using the 5th-percentile (darkest representative region) and
      95th-percentile (lightest representative region) of the image luminance
      distribution.  Using percentiles instead of absolute min/max prevents isolated
      bright specks or dark corners from inflating the score.

    WCAG thresholds (large/bold text applies for thumbnail headings):
      ratio >= 7.0  → AAA  (enhanced accessibility — excellent legibility)
      ratio >= 4.5  → AA   (minimum accessibility — acceptable legibility)
      ratio <  4.5  → fail (poor contrast — text likely unreadable at a glance)
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

    # sRGB → linear light (IEC 61966-2-1)
    def _linearize(c: np.ndarray) -> np.ndarray:
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    r = _linearize(arr[:, :, 0])
    g = _linearize(arr[:, :, 1])
    b = _linearize(arr[:, :, 2])
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b  # WCAG relative luminance

    p5 = float(np.percentile(luminance, 5))  # representative dark value
    p95 = float(np.percentile(luminance, 95))  # representative light value
    L1, L2 = max(p5, p95), min(p5, p95)
    ratio = (L1 + 0.05) / (L2 + 0.05)

    if ratio >= 7.0:
        grade = "AAA"
    elif ratio >= 4.5:
        grade = "AA"
    else:
        grade = "fail"

    return {
        "score": round(ratio, 2),
        "grade": grade,
        "dark_luminance": round(p5, 4),
        "light_luminance": round(p95, 4),
        "threshold_AA": 4.5,
        "threshold_AAA": 7.0,
    }


def _metric_artifact_score(img: Image.Image) -> dict:
    """
    Metric 3 — Artifact detection heuristics (composite cleanliness score).

    Justification:
      AI image generators occasionally produce images with visible artifacts that
      degrade thumbnail quality.  We measure three independent signals:

      a) Sharpness (Laplacian variance via FIND_EDGES filter):
         Low variance indicates blurry or soft-focus output.
         Normalized to [0, 1] where 1 = very sharp.  Threshold: raw variance >= 100.

      b) Noise level (fraction of pixels > 2.5 σ from global mean on a 320×180
         downsampled grayscale):
         High noise fraction indicates model hallucination or visible grain.
         Threshold: noise_level <= 0.05 (≤5% noisy pixels is acceptable).

      c) Banding (fraction of near-identical adjacent rows in grayscale):
         If > 40% of consecutive row pairs have luminance difference < 0.5 on a
         [0, 255] scale, horizontal banding artifacts are present.

    Weights: sharpness 40 %, noise 40 %, banding 20 % → overall score 0–1.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)

    # a) Sharpness — Laplacian variance
    edges = np.array(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    sharpness = float(np.var(edges))
    sharpness_score = min(1.0, sharpness / 200.0)  # 200+ = excellent sharpness

    # b) Noise — pixel deviation from global mean on downsampled image
    small = np.array(
        img.convert("L").resize((320, 180), Image.BILINEAR), dtype=np.float32
    )
    mean_val = float(np.mean(small))
    std_val = float(np.std(small)) + 1e-6
    noisy_px = int(np.sum(np.abs(small - mean_val) > 2.5 * std_val))
    noise_level = noisy_px / small.size
    noise_score = max(0.0, 1.0 - noise_level * 10.0)  # 10 % noisy pixels → score 0

    # c) Banding — consecutive-row luminance similarity
    row_means = np.mean(gray, axis=1)
    row_diffs = np.abs(np.diff(row_means))
    banding_detected = bool(np.sum(row_diffs < 0.5) > 0.40 * len(row_diffs))
    banding_score = 0.5 if banding_detected else 1.0

    overall = round(
        0.40 * sharpness_score + 0.40 * noise_score + 0.20 * banding_score, 3
    )

    return {
        "score": overall,
        "sharpness_variance": round(sharpness, 1),
        "sharpness_score": round(sharpness_score, 3),
        "noise_level": round(noise_level, 4),
        "noise_score": round(noise_score, 3),
        "banding_detected": banding_detected,
        "banding_score": banding_score,
    }


def _metric_layout_stability(img: Image.Image) -> dict:
    """
    Metric 4 — Layout stability (single-seed proxy metrics).

    Justification:
      True layout stability requires generating multiple images with the same prompt
      under different random seeds and measuring structural similarity (e.g., SSIM on
      edge maps).  Since we generate one image per pipeline run, we instead measure
      three single-image proxies that strongly correlate with stable, intentional
      composition:

      a) Rule-of-thirds adherence (weight 35 %):
         Measures edge density in ±5 % bands around the horizontal/vertical ⅓ and ⅔
         grid lines using Pillow's FIND_EDGES filter.  Well-composed thumbnails place
         key visual elements near these lines; randomly drifting layouts do not.
         Normalized to [0, 1]; score of 1 means strong edges at every grid line.

      b) Horizontal symmetry (weight 30 %):
         Compares left half vs. mirrored right half via mean absolute pixel difference.
         Intentional asymmetry (deliberate composition) should score ~0.5–0.8;
         very low scores indicate chaotic layout drift.

      c) Visual centroid balance (weight 35 %):
         Computes the brightness-weighted centroid of the image and measures its
         distance from the optical center (0.50, 0.45 — slightly above geometric
         center, matching natural focal tendency).  Severely off-center centroids
         indicate unintended layout drift.

    Note: these are single-image proxies.  For rigorous stability measurement,
    generate N images with the same prompt and compute pairwise structural similarity.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape

    # a) Rule-of-thirds — edge density near ⅓/⅔ grid lines
    edges = np.array(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    band_px = max(1, int(h * 0.05))
    thirds_h = [h // 3, 2 * h // 3]
    thirds_w = [w // 3, 2 * w // 3]

    grid_density = 0.0
    for r in thirds_h:
        grid_density += float(np.mean(edges[max(0, r - band_px) : r + band_px, :]))
    for c in thirds_w:
        grid_density += float(np.mean(edges[:, max(0, c - band_px) : c + band_px]))
    grid_density /= 4  # average over the four grid lines
    thirds_score = min(1.0, grid_density / 30.0)  # 30 = strong edge density

    # b) Horizontal symmetry — left half vs. mirrored right half
    left = gray[:, : w // 2]
    right = np.fliplr(gray[:, w - w // 2 :])
    min_w = min(left.shape[1], right.shape[1])
    sym_diff = np.abs(left[:, :min_w] - right[:, :min_w])
    symmetry_score = max(0.0, 1.0 - float(np.mean(sym_diff)) / 128.0)

    # c) Visual centroid vs. optical center
    total_weight = float(np.sum(gray)) + 1e-6
    cy = float(np.sum(gray * np.arange(h)[:, None])) / total_weight / h  # 0–1
    cx = float(np.sum(gray * np.arange(w)[None, :])) / total_weight / w  # 0–1
    centroid_dist = ((cx - 0.50) ** 2 + (cy - 0.45) ** 2) ** 0.5
    centroid_score = max(0.0, 1.0 - centroid_dist * 3.0)  # dist > 0.33 → 0

    overall = round(
        0.35 * thirds_score + 0.30 * symmetry_score + 0.35 * centroid_score, 3
    )

    return {
        "score": overall,
        "rule_of_thirds_score": round(thirds_score, 3),
        "symmetry_score": round(symmetry_score, 3),
        "centroid_x": round(cx, 3),
        "centroid_y": round(cy, 3),
        "centroid_balance_score": round(centroid_score, 3),
        "note": (
            "Single-image proxy metrics — true stability requires multi-seed "
            "SSIM comparison across repeated generations of the same prompt."
        ),
    }


def validate_output(state: ThumbnailState) -> ThumbnailState:
    """
    Validator agent: uses Gemini vision to check whether the thumbnail image
    and overlay text are both relevant to the user's original prompt.

    On failure it writes specific, actionable feedback into state["validator_feedback"]
    and increments state["retry_count"] so the graph can loop back to analyze_topic
    for a corrected attempt. On success it clears the feedback and approves.
    """
    retry = state.get("retry_count", 0)
    print(
        f"[4/5] Validator agent reviewing thumbnail (attempt {retry + 1}/{MAX_RETRIES + 1})..."
    )

    img_pil = Image.open(io.BytesIO(state["image_bytes"])).convert("RGB")
    img_b64 = base64.b64encode(state["image_bytes"]).decode("utf-8")

    # ------------------------------------------------------------------
    # Quantitative metrics (run before the LLM call so results are always
    # captured even if the vision call raises an exception)
    # ------------------------------------------------------------------
    ocr_m = _metric_ocr_accuracy(img_pil, state.get("thumbnail_text", ""))
    contrast_m = _metric_contrast_score(img_pil)
    artifact_m = _metric_artifact_score(img_pil)
    layout_m = _metric_layout_stability(img_pil)

    # Weighted overall quality (equal contribution from the four pillars)
    ocr_score = ocr_m.get("score") or 0.0  # None when tesseract missing → 0
    overall_q = round(
        0.25 * ocr_score
        + 0.25 * min(1.0, contrast_m["score"] / 7.0)  # normalise to 0–1 (AAA = 1.0)
        + 0.25 * artifact_m["score"]
        + 0.25 * layout_m["score"],
        3,
    )

    metrics = {
        "ocr_accuracy": ocr_m,
        "contrast": contrast_m,
        "artifacts": artifact_m,
        "layout_stability": layout_m,
        "overall_quality": overall_q,
    }

    print(
        f"  [metrics] OCR={ocr_m.get('score', 'N/A')}  "
        f"Contrast={contrast_m['score']}({contrast_m['grade']})  "
        f"Artifacts={artifact_m['score']}  "
        f"Layout={layout_m['score']}  "
        f"Overall={overall_q}"
    )

    messages = [
        SystemMessage(
            content=(
                "You are a strict YouTube thumbnail quality validator agent.\n"
                "You inspect a thumbnail (image + overlay text) against the user's topic "
                "and write clear, actionable feedback for the generator agents when issues exist.\n\n"
                "Evaluate three things:\n"
                "  1. IMAGE RELEVANCE  — does the visual content clearly depict the topic?\n"
                "  2. TEXT RELEVANCE   — does the overlay text accurately represent the topic?\n"
                "  3. CONSISTENCY      — do the image and text work together for this topic?\n\n"
                "  4. check like a human check for other issues also like spelling mistakes, spacess between letters, weird artifacts, low contrast, bad layout etc. and include all these in your feedback\n\n"
                "if something fails make sure fails the thumbnail and also include specific instructions on how to fix the issues in the feedback so the generator agents can improve in the next attempt. Be specific and actionable in your feedback.\n\n"
                "Respond with ONLY valid JSON — no markdown fences, no extra keys:\n"
                "{\n"
                '  "passed": true or false,\n'
                '  "image_relevant": true or false,\n'
                '  "text_relevant": true or false,\n'
                '  "reason": "one-sentence verdict",\n'
                '  "feedback": "specific instructions telling the generator agents exactly what to change — empty string if passed"\n',
                "}\n\n"
                "Approve (passed=true) when the thumbnail is clearly on-topic.\n"
                "Reject (passed=false) only if image or text clearly contradicts or is "
                "entirely unrelated to the topic. Minor creative liberties are fine."
                f"metrics  : {metrics}",
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        f"User's original prompt: {state['user_prompt']}\n"
                        f"Overlay text on the thumbnail: {state['thumbnail_text']}\n\n"
                        "Does this thumbnail match the user's topic?"
                    ),
                },
            ]
        ),
    ]

    try:
        response = _gemini_llm().invoke(messages)
        raw = response.content.strip()

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)

        result = json.loads(raw)
        passed = bool(result.get("passed", True))
        print(f"-----------------> {result}")
        reason = result.get("reason", "")
        feedback = result.get("feedback", "")
        img_ok = result.get("image_relevant", True)
        txt_ok = result.get("text_relevant", True)

    except Exception as e:
        # Transient validator error — let the thumbnail through rather than blocking it
        print(f"  [validator] Error during validation, skipping: {e}")
        return {
            **state,
            "validation_result": f"SKIPPED (validator error): {e}",
            "validation_metrics": metrics,
        }

    if passed:
        print(f"  [validator] PASSED — {reason}")
        return {
            **state,
            "validation_result": f"PASSED: {reason}",
            "validator_feedback": "",  # clear any previous feedback
            "validation_metrics": metrics,
        }

    # Build a detailed feedback message for the generator agents
    issues = []
    if not img_ok:
        issues.append("the image does not visually match the topic")
    if not txt_ok:
        issues.append("the overlay text is not relevant to the topic")
    issue_summary = (
        "; ".join(issues) if issues else "the thumbnail does not match the topic"
    )

    full_feedback = f"Issues: {issue_summary}.\n"
    if feedback:
        full_feedback += f"Instructions: {feedback}"

    print(f"  [validator] FAILED (retry {retry + 1}/{MAX_RETRIES}) — {reason}")

    return {
        **state,
        "validation_result": f"FAILED: {reason}",
        "validator_feedback": full_feedback,
        "retry_count": retry + 1,
        "validation_metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Node 7: Save image to disk
# ---------------------------------------------------------------------------


def save_image(state: ThumbnailState) -> ThumbnailState:
    """Saves the final PNG bytes to a local file."""
    print("[5/5] Saving thumbnail to disk...")

    output_dir = Path(os.getenv("OUTPUT_DIR", "./thumbnails"))
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"thumbnail_{uuid.uuid4().hex[:8]}.png"
    output_path = output_dir / filename

    with open(output_path, "wb") as f:
        f.write(state["image_bytes"])

    return {**state, "output_path": str(output_path)}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------


def route_after_create(state: ThumbnailState) -> str:
    """Abort immediately if Imagen failed (non-retryable), otherwise validate."""
    return "abort" if state.get("error") else "validate"


def route_after_validation(state: ThumbnailState) -> str:
    """
    After the validator agent runs:
      - PASSED or SKIPPED → save the thumbnail
      - FAILED + retries remaining → loop back to analyze_topic
      - FAILED + retries exhausted → save the best attempt anyway
    """
    result = state.get("validation_result", "")
    if result.startswith("FAILED") and state.get("retry_count", 0) <= MAX_RETRIES:
        return "retry"
    return "not_processable" if result.startswith("FAILED") else "save"


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------


def mark_not_processable(state: ThumbnailState):
    print("❌ Thumbnail NOT PROCESSABLE — retries exhausted")
    return {
        **state,
        "status": "NOT_PROCESSABLE",
        "error": "Thumbnail failed validation and retries exhausted",
    }


def build_graph(strategy: str = "enriched"):
    graph = StateGraph(ThumbnailState)

    graph.add_node("analyze_topic", analyze_topic)
    graph.add_node("generate_image_prompt", generate_image_prompt)
    create_node = (
        create_thumbnail_image_direct
        if strategy == "direct"
        else create_thumbnail_image
    )
    graph.add_node("create_thumbnail_image", create_node)
    graph.add_node("validate_output", validate_output)
    graph.add_node("save_image", save_image)
    graph.add_node("not_processable", mark_not_processable)

    graph.add_edge(START, "analyze_topic")
    graph.add_edge("analyze_topic", "generate_image_prompt")
    graph.add_edge("generate_image_prompt", "create_thumbnail_image")
    graph.add_edge("create_thumbnail_image", "validate_output")

    graph.add_conditional_edges(
        "validate_output",
        route_after_validation,
        {
            "retry": "analyze_topic",
            "save": "save_image",
            "not_processable": "not_processable",
        },
    )

    graph.add_edge("save_image", END)
    graph.add_edge("not_processable", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_thumbnail(prompt: str, strategy: str = "enriched") -> dict:
    """
    Run the full thumbnail generation pipeline.

    Args:
        prompt:   Description of the YouTube video or desired thumbnail.
        strategy: "enriched" (default) uses Gemini to enrich the image prompt before
                  calling Imagen; "direct" calls Imagen with the base prompt only.

    Returns:
        State dict with keys:
          output_path        – local path to the saved PNG
          image_prompt       – the Imagen prompt that was used
          topic_analysis     – Gemini's topic breakdown
          validation_result  – validator verdict ("PASSED/FAILED/SKIPPED: ...")
          validation_metrics – quantitative scores: ocr_accuracy, contrast, artifacts,
                               layout_stability, overall_quality (0–1)
          retry_count        – number of validator-triggered retries
          error              – non-empty string if something failed (non-retryable)
    """
    app = build_graph(strategy=strategy)
    initial_state: ThumbnailState = {
        "user_prompt": prompt,
        "topic_analysis": "",
        "thumbnail_text": "",
        "image_prompt": "",
        "image_bytes": b"",
        "output_path": "",
        "validator_feedback": "",
        "retry_count": 0,
        "validation_result": "",
        "validation_metrics": {},
        "error": "",
    }
    return app.invoke(initial_state)
