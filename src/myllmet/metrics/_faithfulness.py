import logging
from typing import TYPE_CHECKING, Optional

from myllmet.io_aws import BedrockClient
from myllmet.trackers import BaseTracker, NoOPTracker

from .components import ClaimExtractor, FaithfulnessJudge

if TYPE_CHECKING:
    from myllmet.metrics.components.claim_extractor import OutputSchema as ClaimExtractorOutputSchema
    from myllmet.metrics.components.faithfulness_judge import OutputSchema as FaithfulnessJudgeOutputSchema


logger = logging.getLogger(__name__)


class Faithfulness:
    def __init__(
        self,
        claim_extractor_client: BedrockClient,
        faithfulness_judge_client: BedrockClient,
        *,
        claim_extractor: Optional[ClaimExtractor] = None,
        faithfulness_judge: Optional[FaithfulnessJudge] = None
    ):
        self.claim_extractor_client = claim_extractor_client
        self.faithfulness_judge_client = faithfulness_judge_client

        self._claim_extractor = claim_extractor or ClaimExtractor(
            client=claim_extractor_client
        )
        self._faithfulness_judge = faithfulness_judge or FaithfulnessJudge(
            client=faithfulness_judge_client
        )

        self._tracker: BaseTracker = NoOPTracker()

    def _log_to_tracker(
        self,
        question: str,
        answer: str,
        context: str,
        score: float,
        claim_extractor_output: "ClaimExtractorOutputSchema",
        faithfulness_judge_output: "FaithfulnessJudgeOutputSchema",
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
            raise ValueError(
                "`context` must be provided "
                "in Faithfulness score calculation."
            )
        if ground_truth is not None:
            logger.warning(
                "`ground_truth` is not used in Faithfulness score calculation. "
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
