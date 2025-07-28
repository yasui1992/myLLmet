from uuid import uuid4
from typing import TYPE_CHECKING
from typing import Literal

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
        if record.get("prompts"):
            self._prompt_records.append({
                "id": id_,
                **record["prompts"]
            })
        if record.get("intermediates"):
            self._intermediate_records.append({
                "id": id_,
                **record["intermediates"]
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
