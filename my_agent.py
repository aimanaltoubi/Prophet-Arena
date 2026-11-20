"""Ensemble Oracle — a Prophet Arena agent.

Strategy
--------
1. Query several frontier models through OpenRouter, several samples each.
   During live runs (``event.context`` is empty) the ``:online`` model
   variants do their own web research; during local eval the dataset's
   frozen sources are used instead.
2. Aggregate samples per outcome with a trimmed mean in logit space.
3. Blend with the crowd price as a prior, then mildly extremize
   (Good Judgment Project style).
4. Normalize mutually exclusive events toward a coherent distribution,
   clamp away from 0/1, and cover every outcome exactly once.
5. Never fail: any error falls back to the crowd (0 skill beats a
   missing forecast — skipped weekly questions can be charged 0.25).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys

import arena
from openai import AsyncOpenAI

# ----------------------------------------------------------------- config

OPENROUTER_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

MODELS = [
    m.strip()
    for m in os.environ.get(
        "PA_MODELS",
        "anthropic/claude-opus-5.5,openai/gpt-6-astra,"
        "google/gemini-3.1-pro-preview,moonshotai/kimi-k3.1-pro",
    ).split(",")
    if m.strip()
]
SAMPLES_PER_MODEL = int(os.environ.get("PA_SAMPLES_PER_MODEL", "2"))
TEMPERATURE = float(os.environ.get("PA_TEMPERATURE", "0.6"))
CROWD_WEIGHT = float(os.environ.get("PA_CROWD_WEIGHT", "0.25"))
EXTREMIZE = float(os.environ.get("PA_EXTREMIZE", "1.15"))
CLAMP = float(os.environ.get("PA_CLAMP", "0.005"))
MAX_CONCURRENCY = int(os.environ.get("PA_MAX_CONCURRENCY", "8"))
ONLINE_RESEARCH = os.environ.get("PA_ONLINE", "1") == "1"
TRIM_FRACTION = 0.2  # drop top/bottom 20% of samples per outcome

SYSTEM = """You are a superforecaster. Estimate the probability of EACH listed \
outcome independently, each in [0, 1].

Method:
- Start from the base rate / reference class for this kind of question.
- Treat the crowd price (when given) as a strong prior; deviate only with \
specific evidence, and say what that evidence is.
- Weigh how much time remains until the question closes.
- Beware headlines: prefer primary data, official schedules, and track records.
- For mutually exclusive outcomes the probabilities should sum to about 1.

Finish with ONLY a JSON object on the last line:
{"probabilities": {"<outcome name>": <float>, ...}, "rationale": "<2 sentences>"}
Use the outcome names EXACTLY as listed. Include every outcome."""


# ------------------------------------------------------------- math utils

def _logit(p: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _trimmed_mean(values: list[float], trim: float = TRIM_FRACTION) -> float:
    vs = sorted(values)
    k = int(len(vs) * trim)
    core = vs[k : len(vs) - k] or vs
    return sum(core) / len(core)


def _extract_json(text: str) -> dict:
    """Parse the last JSON object in a model reply."""
    matches = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.DOTALL)
    for raw in reversed(matches):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "probabilities" in parsed:
            return parsed
    raise ValueError("no probabilities JSON found in model reply")


# ----------------------------------------------------------------- agent

class EnsembleOracle(arena.Agent):
    name = "ensemble-oracle"
    display_name = "Ensemble Oracle"
    track = "agentic"
    description = (
        "Multi-model OpenRouter ensemble with web research, logit-space "
        "aggregation, crowd-prior blending, and calibrated extremizing."
    )
    card = {
        "models": MODELS,
        "tools": ["web-search"],
        "methodology": (
            "Multiple frontier models sampled several times each via OpenRouter "
            "(online variants for live research), trimmed-mean aggregation in "
            "logit space, crowd price blended as a prior, mild extremizing, "
            "mutually-exclusive normalization, crowd fallback on failure."
        ),
    }

    def __init__(self) -> None:
        self._llm: AsyncOpenAI | None = None
        self._sem = asyncio.Semaphore(MAX_CONCURRENCY)

    def setup(self) -> None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._llm = AsyncOpenAI(base_url=OPENROUTER_URL, api_key=key, timeout=120.0)

    def teardown(self) -> None:
        self._llm = None

    # ------------------------------------------------------------ prompt

    def _prompt(self, event: arena.Event) -> str:
        rules = event.criteria or "\n".join(
            f"- {o.name}: {o.criteria}" for o in event.outcomes if o.criteria
        ) or "(no explicit resolution criteria provided)"
        parts = [
            f"Question: {event.title}",
            f"Category: {event.category}",
            f"Resolution rules:\n{rules}",
            f"Question closes: {event.close_at:%Y-%m-%d %H:%M} UTC",
            f"Outcomes: {json.dumps(event.outcome_names)}",
            f"Mutually exclusive: {event.mutually_exclusive}",
        ]
        if event.crowd:
            parts.append(f"Crowd prices (prior): {json.dumps(event.crowd)}")
        if event.context:
            evidence = "\n".join(
                f"- {s.title}: {s.summary}" for s in event.context[:15]
            )
            parts.append(f"Evidence (frozen at forecast time):\n{evidence}")
        else:
            parts.append(
                "No evidence provided. Research the latest relevant news and "
                "data yourself before estimating."
            )
        return "\n\n".join(parts)

    # ----------------------------------------------------------- samples

    async def _sample(self, model: str, prompt: str, event: arena.Event) -> dict[str, float] | None:
        """One model call -> {canonical outcome name: probability} or None."""
        assert self._llm is not None
        use_online = ONLINE_RESEARCH and not event.context and ":online" not in model
        slug = f"{model}:online" if use_online else model
        try:
            async with self._sem:
                resp = await self._llm.chat.completions.create(
                    model=slug,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                )
            parsed = _extract_json(resp.choices[0].message.content or "")
            raw = {
                str(k).strip().casefold(): float(v)
                for k, v in parsed["probabilities"].items()
            }
            out: dict[str, float] = {}
            for o in event.outcomes:
                p = raw.get(o.name.strip().casefold())
                if p is None:
                    return None  # sample did not cover every outcome
                out[o.name] = min(1.0, max(0.0, p))
            return out
        except Exception as exc:  # noqa: BLE001 — one bad sample never kills the forecast
            print(f"[{self.name}] sample failed ({slug}, {event.slug}): {exc}", file=sys.stderr)
            return None

    # ---------------------------------------------------------- forecast

    async def forecast_async(self, event: arena.Event) -> arena.Forecast:
        if self._llm is None:
            self.setup()
        prompt = self._prompt(event)

        tasks = [
            self._sample(model, prompt, event)
            for model in MODELS
            for _ in range(SAMPLES_PER_MODEL)
        ]
        samples = [s for s in await asyncio.gather(*tasks) if s is not None]
        if not samples:
            return self._fallback(event)

        # Trimmed mean in logit space per outcome.
        probs: dict[str, float] = {}
        for o in event.outcomes:
            agg = _trimmed_mean([_logit(s[o.name]) for s in samples])
            # Blend with the crowd prior, then extremize the result.
            crowd_p = event.crowd.get(o.name) if event.crowd else None
            if crowd_p is not None:
                agg = (1.0 - CROWD_WEIGHT) * agg + CROWD_WEIGHT * _logit(crowd_p)
            probs[o.name] = _sigmoid(EXTREMIZE * agg)

        if event.mutually_exclusive and len(probs) >= 2:
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}

        probs = {k: min(1.0 - CLAMP, max(CLAMP, v)) for k, v in probs.items()}
        rationale = (
            f"Trimmed-mean logit ensemble of {len(samples)} samples across "
            f"{len(MODELS)} models, crowd-blended and extremized."
        )
        forecast = arena.Forecast(probabilities=probs, rationale=rationale)
        forecast.validate_for(event)
        return forecast

    # ---------------------------------------------------------- fallback

    def _fallback(self, event: arena.Event) -> arena.Forecast:
        """Crowd copy (0 skill) or uniform — never skip a question."""
        if event.crowd:
            probs = {
                o.name: min(1.0 - CLAMP, max(CLAMP, event.crowd.get(o.name, 0.5)))
                for o in event.outcomes
            }
        else:
            p = 1.0 / len(event.outcomes) if event.mutually_exclusive else 0.5
            probs = {o.name: p for o in event.outcomes}
        return arena.Forecast(probabilities=probs, rationale="fallback: crowd prior")

    def on_error(self, event: arena.Event, exc: Exception) -> None:
        print(f"[{self.name}] forecast failed for {event.slug}: {exc}", file=sys.stderr)


agent = EnsembleOracle()
