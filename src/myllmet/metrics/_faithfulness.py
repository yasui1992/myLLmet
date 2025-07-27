# This module includes prompt designs adapted and translated
# from the RAGAS project (https://github.com/explodinggradients/ragas).
# RAGAS is licensed under the Apache License 2.0.
# No source code from RAGAS has been copied or included.


from typing import List, Optional
import logging
import json
from pydantic import BaseModel, Field

from myllmet.io_aws import BedrockClient


logger = logging.getLogger(__name__)


class Claim(BaseModel):
    text: str = Field(..., description="回答から抽出された主張")

class SingleFaithfulnessJudgResult(BaseModel):
    claim: Claim = Field(..., description="与えられた主張")
    verdict: int = Field(..., description="忠実性の判定(0/1)")
    reason: str = Field(..., description="判定の理由")


class ClaimExtractorResult(BaseModel):
    claims: List[Claim]


class FaithfulnessJudgeResult(BaseModel):
    verdicts: List[SingleFaithfulnessJudgResult]


class Faithfulness:
    DEFAULT_CLAIM_EXTRACTOR_INSTRUCTION = (
        "あなたは日本語の言語分析のAIアシスタントです。\n"
        "あなたのタスクは、与えられた質問と回答の組に対して、回答を1つ以上の主張に分解することです。\n"
        "主張には代名詞を一切使用しないでください。\n"
        "各主張は完全に自己完結しており、それ単体で理解可能でなければなりません。前の文脈に依存してはいけません。\n"
        "\n"
        "**必ず**次のJSON Schemaに準拠した形式で、出力をJSONとして返してください。\n"
        "出力ではシングルクォートではなく、エスケープ付きのバックスラッシュを使用してください。\n"
        f"{json.dumps(ClaimExtractorResult.model_json_schema(), ensure_ascii=False)}\n"
        "--------\n"
        "[例]\n"
        "ユーザー: "
        "  質問: アルベルト・アインシュタインは、20世紀を代表する理論物理学者であり、相対性理論を提唱したことで最もよく知られています。\n"
        "  回答: 彼はドイツ生まれの理論物理学者であり、史上最も偉大で影響力のある物理学者の一人として広く認められています。彼は相対性理論の開発で最もよく知られており、また量子力学の理論発展にも重要な貢献をしました。"
        "\n"
        "AIアシスタント: \n"
        "  主張: アルベルト・アインシュタインはドイツ生まれの理論物理学者です。\n"
        "  主張: アルベルト・アインシュタインは、史上最も偉大で影響力のある物理学者の一人として認められています。\n"
        "  主張: アルベルト・アインシュタインは、相対性理論の提唱で最もよく知られています。\n"
        "  主張: アルベルト・アインシュタインはまた、量子力学の理論発展にも重要な貢献をしました。\n"
    )

    DEFAULT_FAITHFULNESS_JUDGE_INSTRUCTION = (
        "あなたは日本語の言語分析のAIアシスタントです。\n"
        "あなたのタスクは、与えられたコンテキストに基づいて、個々の主張の忠実性を判断することです。\n"
        "各主張について、文脈から直接推論できる場合は「1」、直接推論できない場合は「0」を返してください。\n"
        "\n"
        "**必ず**次のJSON Schemaに準拠した形式で、出力をJSONとして返してください。\n"
        "出力ではシングルクォートではなく、エスケープ付きのバックスラッシュを使用してください。\n"
        f"{json.dumps(FaithfulnessJudgeResult.model_json_schema(), ensure_ascii=False)}\n"
    )

    def __init__(
        self,
        claim_extractor_client: BedrockClient,
        faithfulness_judge_client: BedrockClient,
        *,
        claim_extractor_instruction: Optional[str] = None,
        faithfulness_judge_instruction: Optional[str] = None
    ):
        self.claim_extractor_client = claim_extractor_client
        self.faithfulness_judge_client = faithfulness_judge_client

        self.claim_extractor_instruction = claim_extractor_instruction
        self.faithfulness_judge_instruction = faithfulness_judge_instruction

    def _extract_claims(
        self,
        question: str,
        answer: str
    ) -> ClaimExtractorResult:
        system = [{
            "text": self.claim_extractor_instruction \
                or self.DEFAULT_CLAIM_EXTRACTOR_INSTRUCTION
        }]

        user_input = (
            f"質問: {question}\n"
            f"回答: {answer}\n"
        )

        verdict_json = self.claim_extractor_client.chat(
            user_input,
            converse_kwargs={"system": system}
        )

        return ClaimExtractorResult.model_validate_json(verdict_json)

    def _judge_faithfulness(
        self,
        context: str,
        claims: List[str]
    ) -> FaithfulnessJudgeResult:

        system = [{
            "text": self.faithfulness_judge_instruction \
                or self.DEFAULT_FAITHFULNESS_JUDGE_INSTRUCTION
        }]

        context
        claims_as_text = "\n".join(f"- {cl}" for cl in claims)

        user_input = (
            f"コンテキスト: {context}\n"
            "--------\n"
            "主張:\n"
            f"{claims_as_text}\n"
        )

        claims_json = self.faithfulness_judge_client.chat(
            user_input,
            converse_kwargs={"system": system}
        )

        return FaithfulnessJudgeResult.model_validate_json(claims_json)

    def set_claim_extractor_instruction(self, instruction: str) -> None:
        self.claim_extractor_instruction = instruction

    def set_faithfulness_judge_instruction(self, instruction: str) -> None:
        self.faithfulness_judge_instruction = instruction

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
        return score
