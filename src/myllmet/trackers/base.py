from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypedDict


class LLMMetricsRecord(TypedDict):
    question: str
    answer: str
    context: str
    ground_truth: str
    score: float
    prompts: Optional[Dict[str, Any]]
    intermediates: Optional[Dict[str, Any]]


class BaseTracker(ABC):
    @abstractmethod
    def log(self, record: LLMMetricsRecord) -> None:
        pass
