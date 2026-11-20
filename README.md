# Prophet-Arena — Ensemble Oracle

A [Prophet Arena](https://www.prophetarena.co) forecasting agent (`agentic` track)
built to beat the crowd and climb the leaderboard.

## How it forecasts

1. **Ensemble** — samples several frontier models through OpenRouter
   (Claude Opus 5.5, GPT-6 Astra, Gemini 3.1 Pro, Kimi K3.1 Pro by default),
   multiple samples each.
2. **Research** — during live runs (no provided context) the `:online` model
   variants do their own web research; local eval uses the dataset's frozen sources.
3. **Aggregation** — trimmed mean in logit space per outcome (robust to outliers).
4. **Crowd prior** — blends with the market price in logit space, then mildly
   extremizes (Good Judgment Project style).
5. **Contract safety** — case-insensitive outcome matching, mutually-exclusive
   normalization, clamping away from 0/1, and a crowd-copy fallback so no
   question is ever skipped.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env            # fill in your keys
prophet login --key pa_live_... # Prophet Arena key (sign up at prophetarena.co)
```

## Workflow

```bash
# Offline sanity tests (no keys needed)
python -m unittest test_agent_offline

# Local evaluation (uses your OpenRouter key; ~8 LLM calls per event)
prophet eval my_agent.py --dataset prophet-arena-subset-100

# Inspect / compare runs
prophet report
prophet experiments list

# Go live
prophet submit ensemble-oracle
prophet run my_agent.py
```

Positive **skill** with `skill − 2·skill_se > 0` means you beat the crowd
beyond luck. Change one knob at a time (`PA_*` variables in `.env`) and
compare runs on the same dataset version.

## Tuning knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `PA_MODELS` | 4 frontier models | Comma-separated OpenRouter slugs |
| `PA_SAMPLES_PER_MODEL` | 2 | Samples per model per event |
| `PA_TEMPERATURE` | 0.6 | Sampling temperature |
| `PA_CROWD_WEIGHT` | 0.25 | Weight of the crowd prior in logit space |
| `PA_EXTREMIZE` | 1.15 | Extremizing factor on the blended logit |
| `PA_CLAMP` | 0.005 | Min distance from 0/1 |
| `PA_ONLINE` | 1 | Use `:online` research variants when context is empty |
| `PA_MAX_CONCURRENCY` | 8 | Parallel LLM calls |
