from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from .base import BaseTracker, LLMMetricsRecord

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import]


class ListTracker(BaseTracker):
    def __init__(self):
        self._standard_records = []
        self._prompt_records = []
        self._intermediate_records = []

    def log(self, record: LLMMetricsRecord) -> None:
        id_ = str(uuid4())
        self._standard_records.append({
            "id": id_,
            "question": record["question"],
            "answer": record["answer"],
            "context": record.get("context", ""),
            "ground_truth": record.get("ground_truth", ""),
            "score": record["score"]
        })

        prompts = record.get("prompts")
        if prompts is not None:
            self._prompt_records.append({
                "id": id_,
                **prompts
            })

        intermediates = record.get("intermediates")
        if intermediates is not None:
            self._intermediate_records.append({
                "id": id_,
                **intermediates
            })

    def to_pandas(self, kind: Literal["standard", "prompts", "intermediates"]) -> "pd.DataFrame":
        import pandas as pd  # type: ignore[import]

        if kind == "standard":
            df = pd.DataFrame(self._standard_records)
        elif kind == "prompts":
            df = pd.DataFrame(self._prompt_records)
        elif kind == "intermediates":
            df = pd.DataFrame(self._intermediate_records)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        return df
