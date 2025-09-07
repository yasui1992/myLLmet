from typing import TYPE_CHECKING, Any, Dict, Literal
from uuid import uuid4

from myllmet.metrics.interface import TrackerInterface

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import]


class ListTracker(TrackerInterface):
    def __init__(self):
        self._standard_records = []
        self._prompt_records = []
        self._intermediate_records = []

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
        id_ = str(uuid4())
        self._standard_records.append({
            "id": id_,
            "question": question,
            "answer": answer,
            "context": context,
            "ground_truth": ground_truth,
            "score": score
        })

        self._intermediate_records.append({
            "id": id_,
            **intermediates
        })

        self._prompt_records.append({
            "id": id_,
            **prompts
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
