"""End-to-end PDF -> LaTeX pipeline driver.

Usage:
    python -m pipeline.pdf_to_latex INPUT.pdf [--out OUT.tex] ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from .bedrock_client import BedrockClaudeClient, ConverseResult
from .figure_extractor import CroppedFigure, crop_figures, parse_pass1_json
from .prompts import PASS1_SYSTEM, PASS2_SYSTEM, pass1_user_intro, pass2_user

logger = logging.getLogger("pdf_to_latex")


# ---------------------------------------------------------------------------
# .env loading (no python-dotenv dependency)
# ---------------------------------------------------------------------------


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
    # Bedrock long-term API keys authenticate via this specific env var.
    if "BEDROCK_API_KEY" in os.environ and "AWS_BEARER_TOKEN_BEDROCK" not in os.environ:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = os.environ["BEDROCK_API_KEY"]


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


def rasterize_pdf(
    pdf_path: Path, work_dir: Path, dpi: int, max_side: int
) -> list[Path]:
    """Convert each PDF page to a downscaled PNG. Returns ordered list of paths."""
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info("rasterizing %s at %d dpi -> %s", pdf_path.name, dpi, work_dir)
    pil_pages = convert_from_path(str(pdf_path), dpi=dpi, fmt="png")

    out_paths: list[Path] = []
    for i, img in enumerate(pil_pages, start=1):
        img = img.convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
        out_path = work_dir / f"page_{i:03d}.png"
        img.save(out_path, format="PNG", optimize=True)
        size_kb = out_path.stat().st_size / 1024
        logger.info("  page %2d  %dx%d  %.0f KB", i, *img.size, size_kb)
        out_paths.append(out_path)
    return out_paths


# ---------------------------------------------------------------------------
# Pass orchestration
# ---------------------------------------------------------------------------


def _log_usage(label: str, res: ConverseResult) -> None:
    logger.info(
        "%s usage: input=%d  output=%d  total=%d  thinking_chars=%d",
        label,
        res.input_tokens,
        res.output_tokens,
        res.total_tokens,
        len(res.thinking),
    )


def _model_output_cap(model_id: str) -> int:
    """Per-model maximum output tokens. Conservative defaults."""
    mid = model_id.lower()
    if "anthropic" in mid:
        return 16000
    if "amazon.nova" in mid:
        return 9500           # Nova caps around 10k
    if "meta.llama4" in mid:
        return 8000
    if "mistral.pixtral" in mid:
        return 8000
    return 4000


def _batch_pages(n_pages: int, batch_size: int) -> list[tuple[int, int]]:
    out = []
    for start in range(1, n_pages + 1, batch_size):
        out.append((start, min(start + batch_size - 1, n_pages)))
    return out


def run_pass1(
    client: BedrockClaudeClient,
    page_image_paths: list[Path],
    work_dir: Path,
    thinking: bool = True,
    batch_size: int = 4,
) -> dict:
    n = len(page_image_paths)
    cap = _model_output_cap(client.model_id)
    batches = _batch_pages(n, batch_size)
    logger.info("pass 1: %d pages -> %d batch(es) of up to %d, model=%s, max_out=%d, thinking=%s",
                n, len(batches), batch_size, client.model_id, cap, thinking)

    merged: dict = {
        "title": "",
        "course": "",
        "lecture_number": "",
        "pages": [],
        "figures": [],
        "structure_hints": [],
    }

    for bi, (start, end) in enumerate(batches, start=1):
        intro = pass1_user_intro(start, end, n)
        blocks: list = [BedrockClaudeClient.text_block(intro)]
        for p in page_image_paths[start - 1:end]:
            blocks.append(BedrockClaudeClient.image_block(p))

        logger.info("  batch %d/%d: pages %d..%d", bi, len(batches), start, end)
        try:
            result = client.converse(
                user_blocks=blocks,
                system=PASS1_SYSTEM,
                max_tokens=cap,
                thinking=thinking,
            )
        except Exception as e:
            logger.warning("  batch %d failed: %s -- inserting placeholder", bi, e)
            for p_idx in range(start, end + 1):
                merged["pages"].append({
                    "page": p_idx,
                    "transcript": "[transcription failed for this page]",
                })
            continue

        _log_usage(f"  pass1.b{bi}", result)

        (work_dir / f"pass1_batch{bi:02d}_response.txt").write_text(
            result.text, encoding="utf-8"
        )
        if result.thinking:
            (work_dir / f"pass1_batch{bi:02d}_thinking.txt").write_text(
                result.thinking, encoding="utf-8"
            )

        try:
            part = parse_pass1_json(result.text)
        except ValueError as e:
            logger.warning(
                "  batch %d returned unparseable output (%s); inserting placeholder",
                bi, e,
            )
            for p_idx in range(start, end + 1):
                merged["pages"].append({
                    "page": p_idx,
                    "transcript": "[unparseable model output for this page]",
                })
            continue

        if bi == 1:
            for k in ("title", "course", "lecture_number"):
                merged[k] = part.get(k, "") or ""
        for p in part.get("pages", []) or []:
            merged["pages"].append(p)
        for f in part.get("figures", []) or []:
            merged["figures"].append(f)
        for h in part.get("structure_hints", []) or []:
            merged["structure_hints"].append(h)

    merged["pages"].sort(key=lambda p: int(p.get("page", 0) or 0))
    (work_dir / "manifest.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return merged


def run_pass2(
    client: BedrockClaudeClient,
    manifest: dict,
    cropped: list[CroppedFigure],
    work_dir: Path,
    thinking: bool = True,
) -> str:
    enriched = json.loads(json.dumps(manifest))  # deep copy
    by_id = {c.id: c for c in cropped}
    for fig in enriched.get("figures", []) or []:
        fid = re.sub(r"[^A-Za-z0-9_-]", "_", str(fig.get("id", "")).strip())
        c = by_id.get(fid)
        if c is None:
            continue
        fig["id"] = fid
        fig["render"] = c.render
        if c.file:
            fig["file"] = c.file

    available_files = sorted(c.file for c in cropped if c.file)
    user_text = pass2_user(json.dumps(enriched, indent=2, ensure_ascii=False), available_files)

    cap = _model_output_cap(client.model_id)
    logger.info("pass 2: requesting final .tex from %s (thinking=%s, max_out=%d)",
                client.model_id, thinking, cap)
    result = client.converse(
        user_blocks=[BedrockClaudeClient.text_block(user_text)],
        system=PASS2_SYSTEM,
        max_tokens=cap,
        thinking=thinking,
    )
    _log_usage("pass2", result)

    (work_dir / "pass2_response.txt").write_text(result.text, encoding="utf-8")
    if result.thinking:
        (work_dir / "pass2_thinking.txt").write_text(result.thinking, encoding="utf-8")

    return _extract_latex(result.text)


def _extract_latex(text: str) -> str:
    fence = re.search(r"```(?:latex|tex)?\s*\n(.*?)```", text, re.DOTALL)
    body = fence.group(1) if fence else text
    return body.strip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="input PDF path")
    parser.add_argument("--out", type=Path, default=None, help="output .tex path")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--work-dir", type=Path, default=Path(".work"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-side", type=int, default=1800)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="pages per pass-1 batch (smaller = safer for tight output token caps)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="disable extended thinking (useful if billing isn't enabled for it)",
    )
    parser.add_argument("--model", default=None, help="override BEDROCK_MODEL_ID")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("botocore", "boto3", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not args.pdf.exists():
        logger.error("PDF not found: %s", args.pdf)
        return 2

    out_tex = args.out or args.pdf.with_suffix(".tex")
    work_dir = args.work_dir
    figures_dir = args.figures_dir

    client = BedrockClaudeClient(model_id=args.model)
    use_thinking = not args.no_thinking
    logger.info("model=%s region=%s thinking=%s thinking_budget=%d",
                client.model_id, client.region, use_thinking, client.thinking_budget)

    pages = rasterize_pdf(args.pdf, work_dir / "pages", dpi=args.dpi, max_side=args.max_side)

    manifest = run_pass1(
        client, pages, work_dir,
        thinking=use_thinking,
        batch_size=args.batch_size,
    )
    logger.info("pass 1 returned: %d pages, %d figures",
                len(manifest.get("pages", []) or []),
                len(manifest.get("figures", []) or []))

    cropped = crop_figures(manifest, pages, figures_dir)
    logger.info("cropped %d figures (%d png_crop, %d tikz)",
                len(cropped),
                sum(1 for c in cropped if c.render == "png_crop"),
                sum(1 for c in cropped if c.render == "tikz"))

    tex = run_pass2(client, manifest, cropped, work_dir, thinking=use_thinking)
    out_tex.write_text(tex, encoding="utf-8")
    logger.info("wrote %s (%d bytes)", out_tex, out_tex.stat().st_size)

    # quick sanity checks
    if "\\begin{document}" not in tex or "\\end{document}" not in tex:
        logger.warning("output is missing \\begin{document}/\\end{document} pair")
    referenced = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    have = {c.file.removesuffix(".png") for c in cropped if c.file}
    missing = {r for r in referenced if r not in have and f"{r}.png" not in {c.file for c in cropped if c.file}}
    if missing:
        logger.warning("references to missing figures: %s", sorted(missing))

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nDone.\n  LaTeX:   {out_tex}\n  Figures: {figures_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
