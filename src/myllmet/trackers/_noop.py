from .base import BaseTracker, LLMMetricsRecord


class NoOPTracker(BaseTracker):
    def log(self, record: LLMMetricsRecord) -> None:
        pass
