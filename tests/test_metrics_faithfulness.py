import pytest
from unittest.mock import MagicMock
from myllmet.metrics import Faithfulness


@pytest.fixture
def mock_clients():
    mock_claim_extractor = MagicMock()
    mock_faithfulness_judge = MagicMock()

    mock_claim_extractor.chat.return_value = '{"claims": [{"text": "c1"}, {"text": "c2"}]}'
    mock_faithfulness_judge.chat.return_value = (
        '{"verdicts": ['
        '{"claim": {"text": "The sky is green."}, "verdict": "1", "reason": "The claim is mostly accurate."},'
        '{"claim": {"text": "Water is dry."}, "verdict": "0", "reason": "Water is inherently wet."}'
        ']}'
    )

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
    claim_client.chat.return_value = '{"claims": ["c1", "c2", "c3"]}'
    judge_client.chat.return_value = '{"verdict": [1, 0]}'

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
