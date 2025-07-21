# This module includes prompt designs adapted and translated
# from the RAGAS project (https://github.com/explodinggradients/ragas).
# RAGAS is licensed under the Apache License 2.0.
# No source code from RAGAS has been copied or included.


import json
from typing import TypedDict

from myllmet.io_aws import BedrockClient


class ClaimExtractorResult(TypedDict):
    claims: list[str]


class FaithfulnessJudgeResult(TypedDict):
    verdict: list[int]


class Faithfulness:
    DEFAULT_CLAIM_EXTRACTOR_INSTRUCTION = (
        'あなたは日本語の言語分析のAIツールです。\n'
        'あなたのタスクは、与えられた質問と回答に対して、回答を1つ以上の主張に分解することです。\n'
        '主張には代名詞を一切使用しないでください。\n'
        '各主張は完全に自己完結しており、それ単体で理解可能でなければなりません。前の文脈に依存してはいけません。\n'
        '**必ず**次のJSON形式で返答してください:\n'
        '{"claims":["主張1","主張2","主張3"]}\n\n'
    )

    DEFAULT_FAITHFULNESS_JUDGE_INSTRUCTION = (
        'あなたは日本語の言語分析のAIツールです。\n'
        'あなたのタスクは、与えられたコンテキストに基づいて一連の主張の忠実性を判断することです。\n'
        '各主張について、文脈から直接推論できる場合は「1」、直接推論できない場合は「0」を返してください。\n'
        '**必ず**次のJSON形式で返答してください:\n'
        '{"verdict": [1, 0, 1]}\n\n'
    )

    def __init__(
        self,
        claim_extractor_client: BedrockClient,
        faithfulness_judge_client: BedrockClient,
    ):
        self.claim_extractor_client = claim_extractor_client
        self.faithfulness_judge_client = faithfulness_judge_client

        self._claim_extractor_instruction = None
        self._faithfulness_judge_instruction = None

    def _extract_claims(
        self,
        question: str,
        answer: str
    ) -> ClaimExtractorResult:
        system = [{
            "text": self._claim_extractor_instruction \
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

        return json.loads(verdict_json)

    def _judge_faithfulness(
        self,
        retrived_contexts: list[str],
        claims: list[str]
    ) -> FaithfulnessJudgeResult:

        system = [{
            "text": self._faithfulness_judge_instruction \
                or self.DEFAULT_FAITHFULNESS_JUDGE_INSTRUCTION
        }]

        contexts_as_text = "\n".join(retrived_contexts)
        claims_as_text = "\n".join(claims)

        user_input = (
            f"コンテキスト: {contexts_as_text}\n"
            f"主張: {claims_as_text}\n"
        )

        claims_json = self.faithfulness_judge_client.chat(
            user_input,
            converse_kwargs={"system": system}
        )

        return json.loads(claims_json)

    def set_claim_extractor_instruction(self, instruction: str) -> None:
        self._claim_extractor_instruction = instruction

    def set_faithfulness_judge_instruction(self, instruction: str) -> None:
        self._faithfulness_judge_instruction = instruction

    def score(
        self,
        question: str,
        answer: str,
        retrived_contexts: list[str] | None = None,
        ground_truth: str | None = None,  # noqa: F401
    ) -> float:

        if retrived_contexts is None:
            raise ValueError(
                "`retrived_contexts` must be provided "
                "in Faithfulness score calculation."
            )

        claims = self._extract_claims(question, answer)["claims"]
        verdicts = self._judge_faithfulness(retrived_contexts, claims)["verdict"]

        if len(claims) != len(verdicts):
            raise ValueError(
                f"Number of claims ({len(claims)}) "
                f"does not match number of verdicts ({len(verdicts)})."
            )

        score = sum(verdicts) / len(claims)
        return score
