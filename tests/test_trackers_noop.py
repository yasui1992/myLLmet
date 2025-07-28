from myllmet.trackers import NoOPTracker


def test_log_accepts_record():
    tracker = NoOPTracker()

    record = {
        "question": "q",
        "answer": "a",
        "context": "c",
        "ground_truth": "gt",
        "score": 0.5,
    }

    tracker.log(record)
    assert True

