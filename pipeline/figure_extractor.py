"""Crop figure regions out of rasterized page images and parse pass-1 output."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CroppedFigure:
    id: str
    page: int
    render: str
    caption: str
    description: str
    file: str | None  # filename in figures/ or None for tikz
    tikz: str


_META_RE = re.compile(
    r"-{3,}\s*META\s*-{3,}(.*?)-{3,}\s*/\s*META\s*-{3,}",
    re.DOTALL | re.IGNORECASE,
)
_PAGE_RE = re.compile(
    r"-{3,}\s*PAGE\s+(\d+)\s*-{3,}\n?(.*?)\n?-{3,}\s*/\s*PAGE\s*-{3,}",
    re.DOTALL | re.IGNORECASE,
)
_FIGURE_RE = re.compile(
    r"-{3,}\s*FIGURE\s+([A-Za-z0-9_-]+)\s*-{3,}\n?(.*?)\n?-{3,}\s*/\s*FIGURE\s*-{3,}",
    re.DOTALL | re.IGNORECASE,
)


def _parse_kv_block(block: str) -> tuple[dict[str, str], str]:
    """Split `KEY: value` lines from leading section, return (kv, rest)."""
    kv: dict[str, str] = {}
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^\s*([A-Z][A-Z0-9_]*)\s*:\s*(.*)$", line)
        if not m:
            break
        key, value = m.group(1).upper(), m.group(2).rstrip()
        if key == "TIKZ":
            # everything after this line until end-of-block belongs to TIKZ.
            tikz_lines = lines[i + 1:]
            kv["TIKZ"] = "\n".join(tikz_lines).strip()
            return kv, ""
        kv[key] = value
        i += 1
    return kv, "\n".join(lines[i:]).strip()


def parse_pass1_json(text: str) -> dict[str, Any]:
    """Parse the delimited plaintext manifest produced by pass 1.

    Function name retained for backwards compatibility with the orchestrator.
    Returns the same dict shape used by the rest of the pipeline.
    """
    text = text.strip()
    if text.startswith("```"):
        m = re.match(r"```[A-Za-z]*\s*\n?(.*?)(?:```|$)", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

    meta: dict[str, str] = {}
    m = _META_RE.search(text)
    if m:
        meta, _ = _parse_kv_block(m.group(1).strip())

    pages: list[dict[str, Any]] = []
    for pm in _PAGE_RE.finditer(text):
        try:
            page_num = int(pm.group(1))
        except ValueError:
            continue
        transcript = pm.group(2).strip()
        pages.append({"page": page_num, "transcript": transcript})

    figures: list[dict[str, Any]] = []
    for fm in _FIGURE_RE.finditer(text):
        fid = fm.group(1)
        kv, _rest = _parse_kv_block(fm.group(2).strip())
        try:
            page = int(kv.get("PAGE", "0"))
        except ValueError:
            page = 0
        bbox: list[float] = []
        bbox_str = kv.get("BBOX", "").strip()
        if bbox_str:
            parts = re.split(r"[\s,]+", bbox_str)
            try:
                bbox = [float(x) for x in parts if x][:4]
            except ValueError:
                bbox = []
        if len(bbox) != 4:
            bbox = [0.0, 0.0, 0.0, 0.0]
        render = (kv.get("RENDER") or "png_crop").strip().lower()
        if render not in ("tikz", "png_crop"):
            render = "png_crop"
        figures.append({
            "id": fid,
            "page": page,
            "bbox": bbox,
            "render": render,
            "caption": kv.get("CAPTION", "").strip(),
            "description": kv.get("DESCRIPTION", "").strip(),
            "tikz": kv.get("TIKZ", "").strip(),
        })

    if not pages and not figures and not meta:
        raise ValueError("no recognizable PAGE/FIGURE/META blocks in response")

    return {
        "title": meta.get("TITLE", "").strip(),
        "course": meta.get("COURSE", "").strip(),
        "lecture_number": meta.get("LECTURE", "").strip(),
        "pages": pages,
        "figures": figures,
        "structure_hints": [],
    }


def _norm_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if x1 - x0 < 0.01 or y1 - y0 < 0.01:
        return None
    return (x0, y0, x1, y1)


def crop_figures(
    manifest: dict[str, Any],
    page_image_paths: list[Path],
    figures_dir: Path,
    pad_frac: float = 0.01,
) -> list[CroppedFigure]:
    """Crop every `png_crop` figure to `figures_dir/fig_<id>.png`.

    `tikz` figures are returned as well (with `file=None`) so the caller can pass
    the full list to pass 2.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    out: list[CroppedFigure] = []

    for fig in manifest.get("figures", []) or []:
        fid = str(fig.get("id") or "").strip() or f"f{len(out) + 1}"
        # sanitize id for filename use
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", fid)
        page = int(fig.get("page", 0) or 0)
        render = (fig.get("render") or "png_crop").strip().lower()
        caption = (fig.get("caption") or "").strip()
        description = (fig.get("description") or "").strip()
        tikz = fig.get("tikz") or ""

        if render == "tikz" and tikz.strip():
            out.append(
                CroppedFigure(
                    id=safe_id,
                    page=page,
                    render="tikz",
                    caption=caption,
                    description=description,
                    file=None,
                    tikz=tikz,
                )
            )
            continue

        # png_crop path
        if not (1 <= page <= len(page_image_paths)):
            logger.warning("figure %s: page %s out of range, skipping", fid, page)
            continue
        bbox = _norm_bbox(fig.get("bbox"))
        if bbox is None:
            logger.warning("figure %s: invalid bbox, skipping", fid)
            continue

        page_img = Image.open(page_image_paths[page - 1]).convert("RGB")
        w, h = page_img.size
        x0, y0, x1, y1 = bbox
        # add small padding
        pad_x = pad_frac * (x1 - x0)
        pad_y = pad_frac * (y1 - y0)
        px0 = max(0, int((x0 - pad_x) * w))
        py0 = max(0, int((y0 - pad_y) * h))
        px1 = min(w, int((x1 + pad_x) * w))
        py1 = min(h, int((y1 + pad_y) * h))

        crop = page_img.crop((px0, py0, px1, py1))
        out_name = f"fig_{safe_id}.png"
        crop.save(figures_dir / out_name, format="PNG", optimize=True)
        out.append(
            CroppedFigure(
                id=safe_id,
                page=page,
                render="png_crop",
                caption=caption,
                description=description,
                file=out_name,
                tikz="",
            )
        )

    return out
