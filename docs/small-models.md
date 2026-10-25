# Working with a small model

The default model is Qwen2.5-3B-Instruct, served by Ollama as `qwen2.5:3b`.
It fits in about 8 GB of RAM and runs on CPU. It is still a small model.
The pipeline is shaped around that.

## What 3B does well

Short summaries, pulling bullets out of a chunk, and scoring a chunk against
a plainly stated goal. One constrained format at a time is enough structure.

## Where it struggles

Nuanced argument, dense math, long chains of reasoning, and languages other
than English. It will also invent a plausible command or a version number if
you ask for "practical steps" in the same breath as a summary. Quality drops
further when the prompt is truncated, which looks like the model "ignoring"
the end of the document.

## How this repo compensates

- One job per call. Layers are not packed into a single JSON object.
- Chunks default to about 1,500 tokens with about 150 tokens of overlap.
- If the chunk summaries do not fit in the next prompt, they are reduced in
  rounds until they do. The final layer call is a separate pass.
- Word counts, code blocks, and stray numbers are checked in Python.
- Temperature defaults to 0.2 and `top_p` to 0.9.
- Concurrency defaults to 2, because a 3B model on CPU does not speed up when
  you flood it with parallel calls.

## Context window

Ollama's OpenAI-compatible endpoint does not reliably honor a per-request
context size. The daemon default is small. Oversized prompts are then cut off
with no error, and the summary quietly misses the last part of the document.

Build a derived model once:

```bash
ollama pull qwen2.5:3b
make model
```

On Windows, without Make:

```powershell
ollama create tldr-qwen -f ollama/Modelfile
```

Then set `LLM_MODEL=tldr-qwen`. The Modelfile sets `num_ctx 8192`.

The other fix is to start the daemon with `OLLAMA_CONTEXT_LENGTH=8192` before
`ollama serve`, and leave `LLM_MODEL=qwen2.5:3b`. The app warns at startup
when the configured model is not `tldr-qwen` and the base URL looks like
Ollama. It also warns when `/api/show` reports a `num_ctx` below
`LLM_MAX_CONTEXT_TOKENS`.

## Scaling chunks

Keep `CHUNK_TOKENS` near `min(4000, LLM_MAX_CONTEXT_TOKENS * 0.2)`. At 8192,
1500 is the default. A 32k window can use 3000–4000. Leave room for the
prompt template, the output cap, and the 256-token safety margin.

## Upgrading the model

Larger models are a configuration change, not a code change.

| Model | Rough memory | Notes |
| --- | --- | --- |
| `qwen2.5:3b` | ~8 GB, CPU is fine | Default. Build `tldr-qwen` so context is 8192. |
| `llama3.2:3b` | ~8 GB | Another small option. Same chunk settings. |
| `qwen2.5:7b-instruct` | ~16 GB recommended | Clearer explanations. You can raise `CHUNK_TOKENS`. |

Expect roughly 20–45 seconds per map call on CPU for the 3B model, and
roughly 5–15 seconds on a small GPU. Those figures are approximate. The
progress line switches to the latency it actually measures.

## Other servers

LM Studio, default local server:

```bash
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=qwen2.5-3b-instruct
```

vLLM:

```bash
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=EMPTY
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
```

OpenRouter (this sends the document off the machine):

```bash
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=qwen/qwen-2.5-7b-instruct
```

Put the key in `.env`, not in the shell history you commit. `.env` is gitignored.

## Licenses

This repository is MIT. Model weights are not.

Qwen2.5-3B is released under the Qwen Research License, not Apache 2.0.
Read the current model card before you use the weights commercially.
Qwen2.5-7B-Instruct is Apache 2.0. Llama weights have their own license.
Verify the terms on the model card you actually pull. They change, and this
paragraph is not legal advice.
