# notes-generator

A small pipeline that turns an image-only PDF (e.g. scanned/scribbled lecture notes) into a clean, compilable LaTeX file using a vision-capable LLM.

Supports five providers out of the box: **Gemini**, **Anthropic** (Claude), **OpenAI**, **OpenRouter**, and **Amazon Bedrock**. Default is `gemini-2.5-pro` (free tier, no credit card required, native vision + thinking).

The pipeline:

1. Rasterizes the PDF pages to PNGs (downscaled for token efficiency).
2. **Pass 1** - sends each page image to the model (one page per call by default) and asks for a delimited plaintext manifest (META + per-page transcript + figure metadata + TikZ code where appropriate). Per-page batching keeps each call within per-model output-token caps and bounds the blast-radius of any single bad page.
3. Crops the `png_crop` figures from the page images deterministically with Pillow using normalized bboxes.
4. **Pass 2** - sends the merged manifest + cropped figure list back to the model (text-only); it emits a single polished `.tex` file.

You compile the `.tex` yourself (e.g. on Overleaf).

## Prerequisites

- Python 3.10+
- `poppler-utils` (system package, used by `pdf2image`)
  - Arch/Manjaro: `sudo pacman -S poppler`
  - Debian/Ubuntu: `sudo apt install poppler-utils`
  - macOS: `brew install poppler`
- An API key for at least one of the supported providers

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit values
```

## Provider quickstart

Pick one. All API keys go in `.env`.

| Provider | API key from | Recommended model | Notes |
| --- | --- | --- | --- |
| **Gemini** (default) | https://aistudio.google.com/app/apikey | `gemini-2.5-pro` | free tier, no card, native thinking |
| **Anthropic** | https://console.anthropic.com/settings/keys | `claude-opus-4-5-20251101` | best math quality, extended thinking |
| **OpenAI** | https://platform.openai.com/api-keys | `gpt-5` | reasoning via `reasoning_effort` |
| **OpenRouter** | https://openrouter.ai/keys | `google/gemini-2.5-pro` | one key for many models incl. free ones like `meta-llama/llama-4-maverick:free` |
| **Bedrock** | AWS console / Bedrock API key | `us.meta.llama4-maverick-17b-instruct-v1:0` | works on AWS credits without a card; Anthropic-on-Bedrock requires payment instrument |

Provider/model is resolved in this order:

1. `--provider` and `--model` CLI flags
2. `LLM_PROVIDER` and `LLM_MODEL` env vars
3. Slash-prefixed model id (`gemini/...`, `anthropic/...`, `openai/...`, `openrouter/...`, `bedrock/...`)
4. Bedrock-style id (`us.*`, `global.*`, `*.claude-*`, etc.) -> bedrock provider
5. Default: `gemini` / `gemini-2.5-pro`

## Configuration

| Var | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini` | one of `gemini`, `anthropic`, `openai`, `openrouter`, `bedrock` |
| `LLM_MODEL` | `gemini-2.5-pro` | model id (or use slash prefix to set both at once) |
| `THINKING_BUDGET_TOKENS` | `12000` | extended-thinking budget per call (ignored if model doesn't support thinking) |
| `GEMINI_API_KEY` | _(none)_ | for `gemini` provider |
| `ANTHROPIC_API_KEY` | _(none)_ | for `anthropic` provider |
| `OPENAI_API_KEY` | _(none)_ | for `openai` provider |
| `OPENROUTER_API_KEY` | _(none)_ | for `openrouter` provider |
| `AWS_REGION` | `us-east-1` | for `bedrock` provider |
| `BEDROCK_API_KEY` | _(none)_ | optional Bedrock long-term API key (`ABSK...`); auto-mapped to `AWS_BEARER_TOKEN_BEDROCK` |
| `BEDROCK_MODEL_ID` | _(none)_ | back-compat fallback for `LLM_MODEL` on bedrock provider |

## Usage

```bash
# default: Gemini 2.5 Pro
python -m pipeline.pdf_to_latex mtl107_lec23.pdf

# pick another provider
python -m pipeline.pdf_to_latex mtl107_lec23.pdf --provider anthropic --model claude-opus-4-5-20251101
python -m pipeline.pdf_to_latex mtl107_lec23.pdf --provider openai    --model gpt-5
python -m pipeline.pdf_to_latex mtl107_lec23.pdf --provider openrouter --model anthropic/claude-opus-4.6
python -m pipeline.pdf_to_latex mtl107_lec23.pdf --provider bedrock   --model us.meta.llama4-maverick-17b-instruct-v1:0

# slash-prefixed shortcut (provider auto-detected)
python -m pipeline.pdf_to_latex mtl107_lec23.pdf --model gemini/gemini-2.5-flash
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
--no-thinking               # disable extended thinking even on capable models
-v                          # verbose logging
```

Outputs:

- `mtl107_lec23.tex` - the final LaTeX file
- `figures/fig_*.png` - cropped figure images referenced via `\includegraphics`

Upload both to Overleaf and compile with pdfLaTeX.

## Architecture

```
pipeline/
  pdf_to_latex.py       # CLI + orchestration (rasterize, batched pass 1, pass 2)
  prompts.py            # PASS1_SYSTEM, PASS2_SYSTEM, helper user-message builders
  figure_extractor.py   # delimited-format parser + Pillow bbox cropping
  providers/
    base.py             # LLMClient ABC + LLMResult dataclass
    __init__.py         # make_client(provider, model) factory + provider auto-detection
    gemini.py           # google-genai SDK
    anthropic.py        # anthropic SDK
    openai_chat.py      # openai SDK (OpenAI + OpenRouter via base_url override)
    bedrock.py          # boto3 bedrock-runtime Converse API
  bedrock_client.py     # back-compat re-export shim
```

Adding a new provider is ~80 lines: subclass `LLMClient`, implement `text_block`, `image_block`, `converse`, and (optionally) `supports_thinking` / `default_max_output_tokens`.

## Notes

- Pass 1 output is a delimited plaintext manifest, not JSON. JSON+LaTeX is fragile because models tend to emit single backslashes; the delimiter format avoids that entirely.
- Extended thinking is auto-disabled for models that don't support it (e.g. Bedrock Llama / Mistral), so you can safely leave it on.
- When thinking is on, `temperature=1.0` is used (Anthropic requirement); otherwise `temperature=0.2`.
- Token usage (input / output / thinking) is printed at the end of each pass for transparency.
- Per-page batching (`--batch-size 1`) keeps each call well below per-model output caps. Bump it up if you switch to a model with a larger output window (Claude Opus, Gemini 2.5 Pro, GPT-5).

## Troubleshooting

**`AccessDeniedException ... INVALID_PAYMENT_INSTRUMENT` (Bedrock)**
Your AWS account has no payment method, or your Marketplace subscription for the
requested Anthropic model isn't fully provisioned. Add a payment instrument in
the AWS Billing console, wait ~2 minutes, then retry. AWS Activate credits do
NOT cover AWS Marketplace charges (which is what Anthropic-on-Bedrock is). Use
a non-Anthropic Bedrock model (Llama / Nova / Pixtral) or switch providers.

**`Invocation of model ID ... with on-demand throughput isn't supported` (Bedrock)**
You used the bare model ID. Use the cross-region inference profile instead
(prefix `us.` or `global.`), e.g. `us.anthropic.claude-opus-4-6-v1`.

**`The provided model identifier is invalid` (Bedrock)**
The model isn't enabled in your `AWS_REGION`. List available IDs:

```bash
aws bedrock list-inference-profiles --region us-east-1 \
  --query 'inferenceProfileSummaries[?contains(inferenceProfileId, `anthropic`)].inferenceProfileId'
```

**`<PROVIDER>_API_KEY is not set`**
Add the matching key to `.env` or export it in your shell. See the Configuration table above for which key each provider uses.

**Pass 1 returned no recognizable PAGE/FIGURE blocks**
Re-run with `--keep-work` and inspect `.work/pass1_batch*_response.txt`. The
parser tolerates an optional code fence and case variations, but the model
needs to emit the `----- PAGE n -----` / `----- /PAGE -----` markers. If a
single batch fails the pipeline inserts a placeholder and continues with the rest.

**Repetition loops on a page**
Per-page batching means only that one page is affected. Inspect
`.work/pass1_batch<N>_response.txt`, then either lower the page DPI, re-run, or
switch models. Amazon Nova Pro in particular tends to fall into
`\forall x \geq 0` style loops on dense handwritten math; Gemini 2.5 Pro,
Claude, and Llama 4 Maverick handle it cleanly.
