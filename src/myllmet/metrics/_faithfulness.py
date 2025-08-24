# This module includes prompt designs adapted and translated
# from the RAGAS project (https://github.com/explodinggradients/ragas).
# RAGAS is licensed under the Apache License 2.0.
# No source code from RAGAS has been copied or included.


import json
import logging
from typing import Any, Dict, List, Optional, cast

from pydantic import BaseModel, Field

from myllmet.io_aws import BedrockClient
from myllmet.metrics.components import ClaimExtractor
from myllmet.schemas import ChatMessage
from myllmet.trackers import BaseTracker, NoOPTracker

logger = logging.getLogger(__name__)


DEFAULT_FAITHFULNESS_JUDGE_INSTRUCTION = (
    "あなたは日本語の言語分析のAIアシスタントです。\n"
    "あなたのタスクは、与えられたコンテキストに基づいて、個々の主張の忠実性を判断することです。\n"
    "各主張について、文脈から直接推論できる場合は「1」、直接推論できない場合は「0」を返してください。\n"
)


DEFAULT_FAITHFULNESS_JUDGE_EXAMPLES = [
    {
        "user": {
            "context": "彼はドイツ生まれの理論物理学者であり、相対性理論の提唱で知られています。",
            "claims": [
                {"text": "アルベルト・アインシュタインは、ドイツ生まれの理論物理学者です。"},
                {"text": "アルベルト・アインシュタインは、アメリカ生まれの理論物理学者です。"},
                {"text": "アルベルト・アインシュタインは、量子力学の理論発展に重要な貢献をしました。"},
            ]
        },
        "assistant": {
            "verdicts": [
                {
                    "claim": "アルベルト・アインシュタインは、ドイツ生まれの理論物理学者です。",
                    "verdict": "1",
                    "reason": "コンテキストの記載と合致しています。"
                },
                {
                    "claim": "アルベルト・アインシュタインは、アメリカ生まれの理論物理学者です。",
                        "verdict": "0",
                        "reason": "コンテキストによると、アインシュタインはドイツ生まれです"
                },
                {
                    "claim": "アルベルト・アインシュタインは、量子力学の理論発展に重要な貢献をしました。",
                        "verdict": "0",
                        "reason": "コンテキストには、アインシュタインの量子力学への貢献は記載されていません。"
                }
            ]
        }
    }
]


class SingleFaithfulnessJudgResult(BaseModel):
    claim: str = Field(..., description="与えられた主張")
    verdict: int = Field(..., description="忠実性の判定(0/1)")
    reason: str = Field(..., description="判定の理由")


class FaithfulnessJudgeInput(BaseModel):
    context: str = Field(..., description="コンテキスト")
    claims: List[str] = Field(..., description="主張")


class FaithfulnessJudgeOutput(BaseModel):
    verdicts: List[SingleFaithfulnessJudgResult]


class FaithfulnessJudgeExample(BaseModel):
    user: FaithfulnessJudgeInput
    assistant: FaithfulnessJudgeOutput


class Faithfulness:
    def __init__(
        self,
        claim_extractor_client: BedrockClient,
        faithfulness_judge_client: BedrockClient,
        enable_fewshot: bool = False,
        *,
        claim_extractor: Optional[ClaimExtractor] = None,
        faithfulness_judge_instruction: Optional[str] = None,
        faithfulness_judge_examples: Optional[List[Dict[str, Any]]] = None
    ):
        self.claim_extractor_client = claim_extractor_client
        self.faithfulness_judge_client = faithfulness_judge_client
        self.enable_fewshot_examples = enable_fewshot

        self._claim_extractor = claim_extractor or ClaimExtractor(
            client=claim_extractor_client,
            enable_fewshot_examples=enable_fewshot
        )
        self._faithfulness_judge_instruction = faithfulness_judge_instruction
        self._faithfulness_judge_examples = faithfulness_judge_examples

        self._tracker: BaseTracker = NoOPTracker()


    @property
    def faithfulness_judge_instruction(self) -> str:
        if self._faithfulness_judge_instruction is None:
            logger.debug(
                f"Using default faithfulness judge instruction in `{self.__class__.__name__}` metrics"
                " as no custom instruction is provided."
            )
            instruct = DEFAULT_FAITHFULNESS_JUDGE_INSTRUCTION

        else:
            instruct = self._faithfulness_judge_instruction

        output = (
            f"{instruct}"
            "\n"
            "次のスキーマに準拠した形式で、出力をJSON文字列として返してください。\n"
            f"{json.dumps(FaithfulnessJudgeOutput.model_json_schema(), ensure_ascii=False)}\n"
        )

        return output

    @property
    def faithfulness_judge_examples(self) -> List[FaithfulnessJudgeExample]:
        if not self.enable_fewshot_examples:
            logger.debug(
                f"Few-shot examples are disabled in `{self.__class__.__name__}` metrics. "
            )
            examples = []
        elif self._faithfulness_judge_examples is None:
            logger.debug(
                f"Using default faithfulness judge examples in `{self.__class__.__name__}` metrics"
                " as no custom examples are provided."
            )
            examples = DEFAULT_FAITHFULNESS_JUDGE_EXAMPLES

        else:
            examples = self._faithfulness_judge_examples

        output = []
        for ex in examples:
            user_dict = cast(dict[str, Any], ex["user"])
            assistant_dict = cast(dict[str, Any], ex["assistant"])

            output.append(
                FaithfulnessJudgeExample(
                    user=FaithfulnessJudgeInput(**user_dict),
                    assistant=FaithfulnessJudgeOutput(**assistant_dict)
                )
            )

        return output

    def _judge_faithfulness(
        self,
        context: str,
        claims: List[str]
    ) -> FaithfulnessJudgeOutput:

        system = [{"text": self.faithfulness_judge_instruction}]
        examples: List[ChatMessage] = []
        for ex in self.faithfulness_judge_examples:
            examples += [
                ChatMessage(
                    role="user",
                    content=json.dumps(ex.user, ensure_ascii=False)
                ),
                ChatMessage(
                    role="assistant",
                    content=json.dumps(ex.assistant, ensure_ascii=False)
                )
            ]

        input_text = json.dumps(
            {
                "context": context,
                "claims": claims
            },
            ensure_ascii=False
        )

        result = self.faithfulness_judge_client.chat(
            input_text,
            chat_history=examples,
            converse_kwargs={"system": system}
        )

        return FaithfulnessJudgeOutput.model_validate_json(result)

    def _log_to_tracker(
        self,
        question: str,
        answer: str,
        context: str,
        score: float,
        claims: List[str],
        verdicts: List[SingleFaithfulnessJudgResult],
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
                    "faithfulness_judge_instruction": self.faithfulness_judge_instruction,
                    "faithfulness_judge_examples": self.faithfulness_judge_examples
                },
                "intermediates": {
                    "claims": claims,
                    "verdicts": [v.model_dump() for v in verdicts],
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

        claim_extract_result = self._claim_extractor.invoke(
            question=question,
            answer=answer
        )
        claims = claim_extract_result["claims"]

        verdicts = self._judge_faithfulness(context, claims).verdicts
        verdicts_as_int = [v.verdict for v in verdicts]

        if len(claims) != len(verdicts):
            raise ValueError(
                f"Number of claims ({len(claims)}) "
                f"does not match number of verdicts ({len(verdicts)})."
            )

        score = sum(verdicts_as_int) / len(claims)

        self._log_to_tracker(
            question=question,
            answer=answer,
            context=context,
            score=score,
            claims=claims,
            verdicts=verdicts
        )

        return score
