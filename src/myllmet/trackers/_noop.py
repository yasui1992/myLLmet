from typing import Any, Dict

from myllmet.metrics.interface import TrackerInterface


class NoOPTracker(TrackerInterface):
    def log(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str,
        score: float,
        intermediates: Dict[str, Any],
        prompts: Dict[str, Any],
    ) -> None:
        pass
