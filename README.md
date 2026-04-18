# notes-generator

A small pipeline that turns an image-only PDF (e.g. scanned/scribbled lecture notes) into a clean, compilable LaTeX file using a vision-capable model on **Amazon Bedrock**.

Default model: **Llama 4 Maverick** (`us.meta.llama4-maverick-17b-instruct-v1:0`) — works without an AWS Marketplace payment instrument. Switch to **Claude 4.6 Opus with extended thinking** (`us.anthropic.claude-opus-4-6-v1`) once your account has billing set up — extended thinking is then enabled automatically.

The pipeline:

1. Rasterizes the PDF pages to PNGs (downscaled for token efficiency).
2. **Pass 1** — sends each page image to the model (one page per call by default) and asks for a delimited plaintext manifest (META + per-page transcript + figure metadata + TikZ code where appropriate). Per-page batching keeps each call within Bedrock's per-model output-token caps and bounds the blast-radius of any single bad page.
3. Crops the `png_crop` figures from the page images deterministically with Pillow using normalized bboxes.
4. **Pass 2** — sends the merged manifest + cropped figure list back to the model (text-only) which emits a single polished `.tex` file.

You compile the `.tex` yourself (e.g. on Overleaf).

## Prerequisites

- Python 3.10+
- `poppler-utils` (system package, used by `pdf2image`)
  - Arch/Manjaro: `sudo pacman -S poppler`
  - Debian/Ubuntu: `sudo apt install poppler-utils`
  - macOS: `brew install poppler`
- AWS credentials configured (env vars, `~/.aws/credentials`, or an instance role)
- Bedrock model access for Claude Opus 4.x in your AWS region

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit values
```

## Configuration

Set via environment (or `.env` — exported manually with `set -a; source .env; set +a`):

| Var | Default | Description |
| --- | --- | --- |
| `AWS_REGION` | `us-east-1` | Bedrock region |
| `BEDROCK_MODEL_ID` | `us.meta.llama4-maverick-17b-instruct-v1:0` | Inference-profile / model ID enabled in your account. Switch to `us.anthropic.claude-opus-4-6-v1` once you've added a payment instrument to your AWS account. |
| `THINKING_BUDGET_TOKENS` | `12000` | Extended-thinking budget per call (ignored unless model is Claude) |
| `BEDROCK_API_KEY` | _(none)_ | Optional Bedrock long-term API key starting with `ABSK...`; auto-mapped to `AWS_BEARER_TOKEN_BEDROCK` |

## Usage

```bash
python -m pipeline.pdf_to_latex mtl107_lec23.pdf
```

Optional flags:

```
--out mtl107_lec23.tex      # output .tex path
--figures-dir figures       # where cropped figures go
--work-dir .work            # rasterized pages + intermediate manifest
--dpi 300                   # rasterization DPI
--max-side 1800             # downscale long-edge of each page (px)
--keep-work                 # don't delete the work dir on success
--batch-size 1              # pages per pass-1 call (raise to merge calls)
--model <id>                # override BEDROCK_MODEL_ID
--no-thinking               # disable extended thinking even on Claude
-v                          # verbose logging
```

Outputs:

- `mtl107_lec23.tex` — the final LaTeX file
- `figures/fig_*.png` — cropped figure images referenced via `\includegraphics`

Upload both to Overleaf and compile with pdfLaTeX.

## Notes

- Pass 1 output is a delimited plaintext manifest, not JSON. JSON+LaTeX is fragile because models tend to emit single backslashes; the delimiter format avoids that entirely.
- Extended thinking (`additionalModelRequestFields.thinking`) is Anthropic-Claude-specific on Bedrock. The client auto-disables it for Nova / Llama / Mistral / Pixtral models so you don't have to think about it.
- When thinking is on, `temperature=1.0` is used (Anthropic requirement); otherwise `temperature=0.2`.
- Token usage (input / output / thinking) is printed at the end of each pass for transparency.
- Per-page batching (`--batch-size 1`) keeps each call well below per-model output caps (Nova ~10k, Llama 4 ~8k). Bump it up if you switch to Claude Opus.
- For figures the model flags as `png_crop`, the crops use normalized bounding boxes from the model with a small padding margin.

## Troubleshooting

**`AccessDeniedException ... INVALID_PAYMENT_INSTRUMENT`**
Your AWS account has no payment method configured (or your Marketplace
subscription for the requested Anthropic model isn't fully provisioned). Add a
payment instrument in the AWS Billing console, wait ~2 minutes, then retry.
Newer Anthropic models (Opus 4.5/4.6/4.7, Sonnet 4.5/4.6, Haiku 4.5) and any
extended-thinking invocation are billed and require this; trial-tier text-only
calls on older models may still succeed.

**`Invocation of model ID ... with on-demand throughput isn't supported`**
You used the bare model ID. Use the cross-region inference profile instead
(prefix `us.` or `global.`), e.g. `us.anthropic.claude-opus-4-6-v1`.

**`The provided model identifier is invalid`**
The model ID isn't enabled in the chosen `AWS_REGION`. List what's available with:

```bash
aws bedrock list-inference-profiles --region us-east-1 \
  --query 'inferenceProfileSummaries[?contains(inferenceProfileId, `anthropic`)].inferenceProfileId'
```

**Pass 1 returned non-JSON**
Re-run with `--keep-work` and inspect `.work/pass1_response.txt` — the model
sometimes adds prose around the fenced JSON block. The parser handles fenced
```json blocks; if Claude returned bare JSON it should still parse.
