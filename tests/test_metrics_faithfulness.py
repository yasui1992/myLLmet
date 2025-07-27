import pytest
from unittest.mock import MagicMock
import json

from myllmet.metrics import Faithfulness


@pytest.fixture
def mock_clients():
    mock_claim_extractor = MagicMock()
    mock_faithfulness_judge = MagicMock()

    mock_claim_extractor.chat.return_value = json.dumps({"claims": ["c1", "c2"]})
    mock_faithfulness_judge.chat.return_value = json.dumps({
        "verdicts": [
            {"claim": "c1", "verdict": "1", "reason": "r1"},
            {"claim": "c2", "verdict": "0", "reason": "r2"}
        ]
    })

    return mock_claim_extractor, mock_faithfulness_judge


def test_score_returns_success(mock_clients):
    claim_client, judge_client = mock_clients
    faithfulness = Faithfulness(
        claim_extractor_client=claim_client,
        faithfulness_judge_client=judge_client
    )

    score = faithfulness.score(
        question="q",
        answer="a",
        context="rc1\nrc2"
    )

    assert score == 0.5


def test_missing_context_failed(mock_clients):
    claim_client, judge_client = mock_clients
    faithfulness = Faithfulness(
        claim_extractor_client=claim_client,
        faithfulness_judge_client=judge_client
    )

    with pytest.raises(ValueError):
        faithfulness.score(
            question="q",
            answer="a"
        )


def test_mismatched_claims_and_verdicts_failed(mock_clients):
    claim_client, judge_client = mock_clients
    claim_client.chat.return_value = json.dumps({"claims": ["c1", "c2", "c3"]})
    judge_client.chat.return_value = json.dumps({"verdicts": [1, 0]})

    faithfulness = Faithfulness(
        claim_extractor_client=claim_client,
        faithfulness_judge_client=judge_client
    )

    with pytest.raises(ValueError):
        faithfulness.score(
            question="q",
            answer="a",
            context="rc1\nrc2"
        )
