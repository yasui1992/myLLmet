from myllmet.metrics.interface import LLMMetricsRecord, TrackerInterface


class NoOPTracker(TrackerInterface):
    def log(self, record: LLMMetricsRecord) -> None:
        pass
