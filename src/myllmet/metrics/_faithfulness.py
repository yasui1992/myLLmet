# This module includes prompt designs adapted and translated
# from the RAGAS project (https://github.com/explodinggradients/ragas).
# RAGAS is licensed under the Apache License 2.0.
# No source code from RAGAS has been copied or included.


from typing import Dict, List, Optional
import logging
import json
from pydantic import BaseModel, Field

from myllmet.io_aws import BedrockClient
from myllmet.trackers import BaseTracker, NoOPTracker


logger = logging.getLogger(__name__)


class SingleFaithfulnessJudgResult(BaseModel):
    claim: str = Field(..., description="与えられた主張")
    verdict: int = Field(..., description="忠実性の判定(0/1)")
    reason: str = Field(..., description="判定の理由")


class ClaimExtractorResult(BaseModel):
    claims: List[str] = Field(..., description="抽出された主張のリスト")


class FaithfulnessJudgeResult(BaseModel):
    verdicts: List[SingleFaithfulnessJudgResult]


class Faithfulness:
    DEFAULT_CLAIM_EXTRACTOR_INSTRUCTION = (
        "あなたは日本語の言語分析のAIアシスタントです。\n"
        "あなたのタスクは、与えられた質問と回答の組に対して、回答を1つ以上の主張に分解することです。\n"
        "主張には代名詞を一切使用しないでください。\n"
        "各主張は完全に自己完結しており、それ単体で理解可能でなければなりません。前の文脈に依存してはいけません。\n"
        "\n"
        "次のスキーマに準拠した形式で、出力をJSON文字列として返してください。\n"
        f"{json.dumps(ClaimExtractorResult.model_json_schema(), ensure_ascii=False)}\n"
    )

    DEFAULT_FAITHFULNESS_JUDGE_INSTRUCTION = (
        "あなたは日本語の言語分析のAIアシスタントです。\n"
        "あなたのタスクは、与えられたコンテキストに基づいて、個々の主張の忠実性を判断することです。\n"
        "各主張について、文脈から直接推論できる場合は「1」、直接推論できない場合は「0」を返してください。\n"
        "\n"
        "次のスキーマに準拠した形式で、出力をJSON文字列として返してください。\n"
        f"{json.dumps(FaithfulnessJudgeResult.model_json_schema(), ensure_ascii=False)}\n"
    )

    DEFAULT_CLAIM_EXTRACTOR_EXAMPLES = [
        {
            "user": {
                "question": "アルベルト・アインシュタインについて教えてください。",
                "answer": (
                    "彼はドイツ生まれの理論物理学者であり、史上最も偉大で影響力のある物理学者の一人として広く認められています。"
                    "彼は相対性理論の提唱で最もよく知られており、また量子力学の理論発展にも重要な貢献をしました。"
                )
            },
            "assistant": {
                "claims": [
                    "アルベルト・アインシュタインは、ドイツ生まれの理論物理学者です。",
                    "アルベルト・アインシュタインは、史上最も偉大で影響力のある物理学者の一人として認められています。",
                    "アルベルト・アインシュタインは、相対性理論の提唱で最もよく知られています。",
                    "アルベルト・アインシュタインは、量子力学の理論発展に重要な貢献をしました。"
                ]
            }
        }
    ]

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

    def __init__(
        self,
        claim_extractor_client: BedrockClient,
        faithfulness_judge_client: BedrockClient,
        *,
        activate_fewshot_examples: bool = False,
        claim_extractor_instruction: Optional[str] = None,
        claim_extractor_examples: Optional[List] = None,
        faithfulness_judge_instruction: Optional[str] = None,
        faithfulness_judge_examples: Optional[List] = None
    ):
        self.claim_extractor_client = claim_extractor_client
        self.faithfulness_judge_client = faithfulness_judge_client

        self._activate_fewshot_examples = activate_fewshot_examples
        self._claim_extractor_instruction = claim_extractor_instruction
        self._claim_extractor_examples = claim_extractor_examples
        self._faithfulness_judge_instruction = faithfulness_judge_instruction
        self._faithfulness_judge_examples = faithfulness_judge_examples

        self._tracker = NoOPTracker()

    @property
    def claim_extractor_instruction(self) -> str:
        if self._claim_extractor_instruction is None:
            logger.debug(
                f"Using default claim extractor instruction in `{self.__class__.__name__}` metrics"
                " as no custom instruction is provided."
            )
            output = self.DEFAULT_CLAIM_EXTRACTOR_INSTRUCTION

        else:
            output = self._claim_extractor_instruction

        return output

    @property
    def faithfulness_judge_instruction(self) -> str:
        if self._faithfulness_judge_instruction is None:
            logger.debug(
                f"Using default faithfulness judge instruction in `{self.__class__.__name__}` metrics"
                " as no custom instruction is provided."
            )
            output = self.DEFAULT_FAITHFULNESS_JUDGE_INSTRUCTION

        else:
            output = self._faithfulness_judge_instruction

        return output

    @property
    def claim_extractor_examples(self) -> List:
        if self._claim_extractor_examples is None:
            logger.debug(
                f"Using default claim extractor examples in `{self.__class__.__name__}` metrics"
                " as no custom examples are provided."
            )
            output = self.DEFAULT_CLAIM_EXTRACTOR_EXAMPLES

        else:
            output = self._claim_extractor_examples

        return output

    @property
    def faithfulness_judge_examples(self) -> List:
        if self._faithfulness_judge_examples is None:
            logger.debug(
                f"Using default faithfulness judge examples in `{self.__class__.__name__}` metrics"
                " as no custom examples are provided."
            )
            output = self.DEFAULT_FAITHFULNESS_JUDGE_EXAMPLES

        else:
            output = self._faithfulness_judge_examples

        return output

    def _convert_to_chat_history(self, examples: List) -> List[Dict]:
        chat_history = []
        for ex in examples:
            chat_history += [
                {
                    "role": "user",
                    "content": json.dumps(ex["user"], ensure_ascii=False)
                },
                {
                    "role": "assistant",
                    "content": json.dumps(ex["assistant"], ensure_ascii=False)
                }
            ]

        return chat_history

    def _extract_claims(
        self,
        question: str,
        answer: str
    ) -> ClaimExtractorResult:

        system = [{"text": self.claim_extractor_instruction}]
        if self._activate_fewshot_examples:
            chat_history = self._convert_to_chat_history(self.claim_extractor_examples)
        else:
            logger.debug(
                "Few-shot examples of claim extractor are deactivated"
                f"in `{self.__class__.__name__}` metrics."
            )
            chat_history = []

        input_text = json.dumps(
            {
                "question": question,
                "answer": answer
            },
            ensure_ascii=False
        )

        result = self.claim_extractor_client.chat(
            input_text,
            chat_history=chat_history,
            converse_kwargs={"system": system}
        )

        return ClaimExtractorResult.model_validate_json(result)

    def _judge_faithfulness(
        self,
        context: str,
        claims: List[str]
    ) -> FaithfulnessJudgeResult:

        system = [{"text": self.faithfulness_judge_instruction}]
        if self._activate_fewshot_examples:
            chat_history = self._convert_to_chat_history(self.faithfulness_judge_examples)
        else:
            logger.debug(
                "Few-shot examples of faithfulness judge are deactivated "
                f"in `{self.__class__.__name__}` metrics."
            )
            chat_history = []

        input_text = json.dumps(
            {
                "context": context,
                "claims": claims
            },
            ensure_ascii=False
        )

        result = self.faithfulness_judge_client.chat(
            input_text,
            chat_history=chat_history,
            converse_kwargs={"system": system}
        )

        return FaithfulnessJudgeResult.model_validate_json(result)

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
                    "claim_extractor_instruction": self.claim_extractor_instruction,
                    "claim_extractor_examples": self.claim_extractor_examples,
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

        claims = self._extract_claims(question, answer).claims
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
