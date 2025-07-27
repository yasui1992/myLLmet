from myllmet.trackers import NoOPTracker


def test_nooptracker_log_accepts_record():
    tracker = NoOPTracker()

    record = {
        "question": "q",
        "answer": "a",
        "context": "c",
        "ground_truth": "gt",
        "score": 0.5,
        "extra": None,
    }

    tracker.log(record)
    assert True

