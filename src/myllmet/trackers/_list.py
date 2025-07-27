from uuid import uuid4
from typing import TYPE_CHECKING

from .base import BaseTracker, LLMMetricsRecord


if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import]


class ListTracker(BaseTracker):
    def __init__(self):
        self.records = []
        self._extras_records = []

    def log(self, record: LLMMetricsRecord) -> None:
        id_ = str(uuid4())
        record_without_extras = {
            "id": id_,
            "question": record["question"],
            "answer": record["answer"],
            "context": record["context"],
            "ground_truth": record["ground_truth"],
            "score": record["score"]
        }
        record_only_extras = {
            "id": id_,
            **record.get("extras", {})
        }

        self.records.append(record_without_extras)
        self._extras_records.append(record_only_extras)

    def to_pandas(self, extra: bool = False) -> "pd.DataFrame":
        import pandas as pd  # type: ignore[import]

        if extra:
            df = pd.DataFrame(self._extras_records)
        else:
            df = pd.DataFrame(self.records)

        return df
