"""Offline tests for EnsembleOracle — no API keys or network needed."""

from __future__ import annotations

import asyncio
import datetime as dt
import unittest
from unittest.mock import patch

import arena

import my_agent
from my_agent import EnsembleOracle


def make_event(**overrides) -> arena.Event:
    data = {
        "slug": "test-cpi",
        "title": "Will US CPI YoY exceed 3.0%?",
        "category": "Economics",
        "criteria": "Resolves Yes if CPI-U YoY exceeds 3.0%.",
        "close_at": dt.datetime(2026, 12, 1, tzinfo=dt.timezone.utc),
        "outcomes": [{"name": "Yes"}, {"name": "No"}],
        "crowd": {"Yes": 0.4, "No": 0.6},
        "context": [],
        "mutually_exclusive": True,
    }
    data.update(overrides)
    return arena.Event.model_validate(data)


class ForecastPipelineTest(unittest.TestCase):
    def setUp(self):
        self.agent = EnsembleOracle()
        self.agent._llm = object()  # skip setup()

    def _forecast(self, event, sample_return):
        async def fake_sample(model, prompt, ev):
            return sample_return

        with patch.object(EnsembleOracle, "_sample", side_effect=fake_sample):
            return asyncio.run(self.agent.forecast_async(event))

    def test_valid_forecast_covers_all_outcomes(self):
        event = make_event()
        fc = self._forecast(event, {"Yes": 0.7, "No": 0.3})
        fc.validate_for(event)
        self.assertEqual(set(fc.probabilities), {"Yes", "No"})
        for p in fc.probabilities.values():
            self.assertGreaterEqual(p, my_agent.CLAMP)
            self.assertLessEqual(p, 1 - my_agent.CLAMP)

    def test_mutually_exclusive_sums_to_one(self):
        event = make_event(
            outcomes=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            crowd={},
        )
        fc = self._forecast(event, {"A": 0.9, "B": 0.9, "C": 0.9})
        self.assertAlmostEqual(sum(fc.probabilities.values()), 1.0, places=5)

    def test_crowd_blend_pulls_toward_crowd(self):
        event = make_event(mutually_exclusive=False)
        fc = self._forecast(event, {"Yes": 0.9, "No": 0.1})
        # Crowd says 0.4 for Yes, so the blend must land strictly below the raw 0.9.
        self.assertLess(fc.probabilities["Yes"], 0.9)
        self.assertGreater(fc.probabilities["Yes"], 0.4)

    def test_fallback_to_crowd_when_all_samples_fail(self):
        event = make_event()
        fc = self._forecast(event, None)
        fc.validate_for(event)
        self.assertAlmostEqual(fc.probabilities["Yes"], 0.4)
        self.assertAlmostEqual(fc.probabilities["No"], 0.6)

    def test_fallback_uniform_without_crowd(self):
        event = make_event(crowd={}, outcomes=[{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}])
        fc = self._forecast(event, None)
        for p in fc.probabilities.values():
            self.assertAlmostEqual(p, 0.25)

    def test_sample_matches_outcome_names_case_insensitively(self):
        event = make_event()
        agent = EnsembleOracle()
        agent._llm = FakeLLM('{"probabilities": {"YES": 0.8, "no": 0.2}, "rationale": "r"}')
        sample = asyncio.run(agent._sample("fake/model", "prompt", event))
        self.assertEqual(sample, {"Yes": 0.8, "No": 0.2})

    def test_sample_rejects_missing_outcome(self):
        event = make_event()
        agent = EnsembleOracle()
        agent._llm = FakeLLM('{"probabilities": {"Yes": 0.8}}')
        sample = asyncio.run(agent._sample("fake/model", "prompt", event))
        self.assertIsNone(sample)

    def test_json_extracted_from_chatty_reply(self):
        event = make_event()
        reply = (
            'Reasoning: inflation is {"sticky": true} lately...\n'
            'Final answer:\n{"probabilities": {"Yes": 0.35, "No": 0.65}, "rationale": "cooling"}'
        )
        agent = EnsembleOracle()
        agent._llm = FakeLLM(reply)
        sample = asyncio.run(agent._sample("fake/model", "prompt", event))
        self.assertEqual(sample, {"Yes": 0.35, "No": 0.65})


class FakeLLM:
    """Minimal stand-in for AsyncOpenAI chat completions."""

    def __init__(self, content: str):
        class _Completions:
            async def create(inner, **kwargs):
                class Msg:
                    pass

                msg = Msg()
                msg.content = content
                choice = Msg()
                choice.message = msg
                resp = Msg()
                resp.choices = [choice]
                return resp

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


if __name__ == "__main__":
    unittest.main()
