import pytest

from myllmet.metrics import Faithfulness


@pytest.fixture
def claim_extractor_stub_factory():

    class Dummy:
        def __init__(self, return_claims):
            self.return_claims = return_claims

        def invoke(self, question, answer):
            return {"claims": self.return_claims}

        @property
        def instruction(self):
            return "instruction"

        @property
        def fewshot_examples(self):
            return []

    return Dummy

@pytest.fixture
def faithfulness_judge_stub_factory():

    class Dummy:
        def __init__(self, return_verdicts):
            self.return_verdicts = return_verdicts

        def invoke(self, context, claims):
            verdicts = [{"claim": c, "verdict": v, "reason": "r"} for c, v in zip(claims, self.return_verdicts)]
            return {"verdicts": verdicts}

        @property
        def instruction(self):
            return "instruction"

        @property
        def fewshot_examples(self):
            return []

    return Dummy

@pytest.fixture
def tracker_stub():

    class DummyTracker:
        def __init__(self):
            self.logged = None

        def log(self, data):
            self.logged = data

    return DummyTracker()

def test_score_valid(
    claim_extractor_stub_factory,
    faithfulness_judge_stub_factory
):
    ce = claim_extractor_stub_factory(return_claims=["c1", "c2", "c3", "c4"])
    fj = faithfulness_judge_stub_factory(return_verdicts=[1, 0, 1, 1])
    metrics = Faithfulness(ce, fj)

    expected = 0.75
    actual = metrics.score(question="q", answer="a", context="ctx")
    assert actual == expected


def test_score_invalid_no_context(
    claim_extractor_stub_factory,
    faithfulness_judge_stub_factory
):
    ce = claim_extractor_stub_factory(return_claims=["c1"])
    fj = faithfulness_judge_stub_factory(return_verdicts=[1])
    metrics = Faithfulness(ce, fj)

    with pytest.raises(ValueError):
        metrics.score(question="q", answer="a", context=None)


def test_score_invalid_incorresponding(
    claim_extractor_stub_factory,
    faithfulness_judge_stub_factory
):
    ce = claim_extractor_stub_factory(return_claims=["c1", "c2"])
    fj = faithfulness_judge_stub_factory(return_verdicts=[1])
    metrics = Faithfulness(ce, fj)

    with pytest.raises(ValueError):
        metrics.score(question="q", answer="a", context="ctx")


def test_score_logs_to_tracker(
    claim_extractor_stub_factory,
    faithfulness_judge_stub_factory,
    tracker_stub
):
    ce = claim_extractor_stub_factory(return_claims=["c1", "c2"])
    fj = faithfulness_judge_stub_factory(return_verdicts=[1, 0])
    metrics = Faithfulness(ce, fj)

    tracker = tracker_stub
    metrics.set_tracker(tracker)

    expected_logged_question = "q"
    expected_logged_answer = "a"
    expected_logged_context = "ctx"
    expected_logged_score = 0.5
    expected_logged_intermediates = {
        "claims": ["c1", "c2"],
        "verdicts": [
            {"claim": "c1", "verdict": 1, "reason": "r"},
            {"claim": "c2", "verdict": 0, "reason": "r"}
        ]
    }
    expected_logged_promts = {
        "claim_extractor": {
            "instruction": ce.instruction,
            "fewshot_examples": ce.fewshot_examples
        },
        "faithfulness_judge": {
            "instruction": fj.instruction,
            "fewshot_examples": fj.fewshot_examples
        }
    }

    metrics.score(question="q", answer="a", context="ctx")

    logged = tracker.logged
    assert logged["question"] == expected_logged_question
    assert logged["answer"] == expected_logged_answer
    assert logged["context"] == expected_logged_context
    assert logged["score"] == expected_logged_score
    assert logged["intermediates"] == expected_logged_intermediates
    assert logged["prompts"] == expected_logged_promts
