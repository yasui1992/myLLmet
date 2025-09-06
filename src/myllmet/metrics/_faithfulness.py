import logging
from typing import TYPE_CHECKING, Dict, Optional

from myllmet.trackers import BaseTracker, NoOPTracker

from .components import ClaimExtractor, FaithfulnessJudge
from .interface import LLMClientInterface

if TYPE_CHECKING:
    from myllmet.metrics.components.claim_extractor import InputSchema as ClaimExtractorIS
    from myllmet.metrics.components.claim_extractor import OutputSchema as ClaimExtractorOS
    from myllmet.metrics.components.faithfulness_judge import InputSchema as FaithfulnessJudgeIS
    from myllmet.metrics.components.faithfulness_judge import OutputSchema as FaithfulnessJudgeOS


logger = logging.getLogger(__name__)


class Faithfulness:
    def __init__(
        self,
        claim_extractor: ClaimExtractor,
        faithfulness_judge: FaithfulnessJudge
    ):

        self._claim_extractor = claim_extractor
        self._faithfulness_judge = faithfulness_judge

        self._tracker: BaseTracker = NoOPTracker()

    @classmethod
    def from_clients(
        cls,
        claim_extractor_client: LLMClientInterface["ClaimExtractorIS", "ClaimExtractorOS"],
        faithfulness_judge_client: LLMClientInterface["FaithfulnessJudgeIS", "FaithfulnessJudgeOS"],
        kwargs_claim_extractor: Optional[Dict] = None,
        kwargs_faithfulness_judge: Optional[Dict] = None,
    ) -> "Faithfulness":

        claim_extractor = ClaimExtractor(
            client=claim_extractor_client,
            **(kwargs_claim_extractor or {})
        )
        faithfulness_judge = FaithfulnessJudge(
            client=faithfulness_judge_client,
            **(kwargs_faithfulness_judge or {})
        )

        return cls(
            claim_extractor=claim_extractor,
            faithfulness_judge=faithfulness_judge
        )

    def set_tracker(self, tracker: BaseTracker) -> None:
        self._tracker = tracker

    def score(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> float:

        if context is None:
            raise ValueError(f"`context` must be provided in {self.__class__.__name__} score calculation.")
        if ground_truth is not None:
            logger.warning(
                f"`ground_truth` is not used in {self.__class__.__name__} score calculation. "
                "It will be ignored."
            )

        claim_extractor_output = self._claim_extractor.invoke(question,answer)
        claims = claim_extractor_output["claims"]

        faithfulness_judge_output = self._faithfulness_judge.invoke(context, claims)
        verdicts = [v["verdict"] for v in faithfulness_judge_output["verdicts"]]

        if len(claims) != len(verdicts):
            raise ValueError(
                f"Number of claims ({len(claims)}) "
                f"does not match number of verdicts ({len(verdicts)})."
            )

        score = sum(verdicts) / len(claims)

        self._log_to_tracker(
            question=question,
            answer=answer,
            context=context,
            score=score,
            claim_extractor_output=claim_extractor_output,
            faithfulness_judge_output=faithfulness_judge_output
        )

        return score

    def _log_to_tracker(
        self,
        question: str,
        answer: str,
        context: str,
        score: float,
        claim_extractor_output: "ClaimExtractorOS",
        faithfulness_judge_output: "FaithfulnessJudgeOS",
    ) -> None:
        self._tracker.log(
            {
                "question": question,
                "answer": answer,
                "context": context,
                "ground_truth": "",  # Not used in Faithfulness score calculation
                "score": score,
                "prompts": {
                    "claim_extractor_instruction": self._claim_extractor.instruction,
                    "claim_extractor_examples": self._claim_extractor.fewshot_examples,
                    "faithfulness_judge_instruction": self._faithfulness_judge.instruction,
                    "faithfulness_judge_examples": self._faithfulness_judge.fewshot_examples,
                },
                "intermediates": {
                    "claims": claim_extractor_output["claims"],
                    "verdicts": faithfulness_judge_output["verdicts"],
                }
            }
        )
