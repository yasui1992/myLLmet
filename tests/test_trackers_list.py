import pytest

from myllmet.trackers import ListTracker


@pytest.fixture
def tracker():
    return ListTracker()


@pytest.fixture
def record():
    record = {
        "question": "q",
        "answer": "a",
        "context": "ctx",
        "ground_truth": "gt",
        "score": 1.0,
        "prompts": {"system": "s"},
        "intermediates": {"int": "i"}
    }
    return record


def test_log_and_to_pandas_standard(tracker, record):
    tracker.log(record)
    df = tracker.to_pandas("standard")

    assert len(df) == 1
    assert isinstance(df.iloc[0]["id"], str)
    assert df.iloc[0]["question"] == "q"
    assert df.iloc[0]["answer"] == "a"
    assert df.iloc[0]["context"] == "ctx"
    assert df.iloc[0]["ground_truth"] == "gt"
    assert df.iloc[0]["score"] == 1.0



def test_log_and_to_pandas_prompts(tracker, record):
    tracker.log(record)
    df = tracker.to_pandas("prompts")

    assert len(df) == 1
    assert isinstance(df.iloc[0]["id"], str)
    assert df.iloc[0]["system"] == "s"


def test_log_and_to_pandas_intermediates(tracker, record):
    tracker.log(record)
    df = tracker.to_pandas("intermediates")

    assert len(df) == 1
    assert isinstance(df.iloc[0]["id"], str)
    assert df.iloc[0]["int"] == "i"


def test_invalid_kind_raises(tracker):
    with pytest.raises(ValueError):
        tracker.to_pandas("invalid_kind")
