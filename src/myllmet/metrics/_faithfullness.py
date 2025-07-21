import json
from typing import TypedDict

from myllmet.io_aws import BedrockClient


class ClaimExtractorResult(TypedDict):
    claims: list[str]


class Faithfulness:
    DEFAULT_CLAIM_EXTRACTOR_INSTRUCTION = (
        'あなたは日本語の言語分析のAIツールです。\n'
        '質問とそれに対応する回答が与えられたら、回答を1つ以上の明確な主張に分解してください。\n'
        '主張には代名詞を一切使用しないでください。\n'
        '各主張は完全に自己完結しており、それ単体で理解可能でなければなりません。前の文脈に依存してはいけません。\n'
        '**必ず**次のJSON形式で返答してください:\n'
        '{"claims":["主張1","主張2","主張3"]}\n\n'
    )

    def __init__(
        self,
        claim_extractor_client: BedrockClient,
        claim_extractor_instruction: str | None = None
    ):
        self.claim_extractor_client = claim_extractor_client
        self.claim_extractor_instruction = claim_extractor_instruction

    def _extract_claims(self, question: str, answer: str) -> ClaimExtractorResult:
        system = [{
            "text": self.claim_extractor_instruction \
                or self.DEFAULT_CLAIM_EXTRACTOR_INSTRUCTION
        }]

        user_input = (
            f"質問: {question}\n"
            f"回答: {answer}\n"
        )

        claims_json = self.claim_extractor_client.chat(
            user_input,
            converse_kwargs={"system": system}
        )

        return json.loads(claims_json)
