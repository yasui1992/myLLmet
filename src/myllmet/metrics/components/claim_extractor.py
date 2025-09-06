# This module includes prompt designs adapted and translated
# from the RAGAS project (https://github.com/explodinggradients/ragas).
# RAGAS is licensed under the Apache License 2.0.
# No source code from RAGAS has been copied or included.


import json
import logging
from typing import List, Optional, TypedDict

import jsonschema

from ..interface import LLMClientInterface

logger = logging.getLogger(__name__)


class InputSchema(TypedDict):
    question: str
    answer: str


class OutputSchema(TypedDict):
    claims: List[str]


class FewShotExample(TypedDict):
    user: InputSchema
    assistant: OutputSchema


OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "抽出された主張のリスト"
        }
    },
    "required": ["claims"]
}


DEFAULT_INSTRUCTION = (
    "あなたは日本語の言語分析のAIアシスタントです。\n"
    "あなたのタスクは、与えられた質問と回答の組に対して、回答を1つ以上の主張に分解することです。\n"
    "主張には代名詞を一切使用しないでください。\n"
    "各主張は完全に自己完結しており、それ単体で理解可能でなければなりません。前の文脈に依存してはいけません。\n"
)


DEFAULT_FEWSHOT_EXAMPLES: List[FewShotExample] = [
    {
        "user": {
            "question": "アインシュタインについて教えてください。",
            "answer": (
                "彼はドイツ生まれの理論物理学者であり、史上最も偉大で影響力のある物理学者の一人として広く認められています。"
                "彼は相対性理論の提唱で最もよく知られており、また量子力学の理論発展にも重要な貢献をしました。"
            )
        },
        "assistant": {
            "claims": [
                "アインシュタインは、ドイツ生まれの理論物理学者です。",
                "アインシュタインは、史上最も偉大で影響力のある物理学者の一人として認められています。",
                "アインシュタインは、相対性理論の提唱で最もよく知られています。",
                "アインシュタインは、量子力学の理論発展に重要な貢献をしました。"
            ]
        }
    }
]


class ClaimExtractor:
    def __init__(
        self,
        client: LLMClientInterface,
        *,
        instruction: Optional[str] = None,
        fewshot_examples: Optional[List[FewShotExample]] = None
    ):
        self.client = client

        self._instruction = instruction
        self._fewshot_examples = fewshot_examples

    @property
    def instruction(self) -> str:
        if self._instruction is None:
            logger.debug(
                f"Using default claim extractor instruction in `{self.__class__.__name__}` metrics"
                " as no custom instruction is provided."
            )
            instruct = DEFAULT_INSTRUCTION

        else:
            instruct = self._instruction

        output = (
            f"{instruct}"
            "\n"
            "次のスキーマに準拠した形式で、出力をJSON文字列として返してください。\n"
            f"{json.dumps(OUTPUT_JSON_SCHEMA, ensure_ascii=False)}\n"
        )

        return output

    @property
    def fewshot_examples(self) -> List[FewShotExample]:
        if self._fewshot_examples is None:
            logger.debug(
                f"Using default few-shot examples in `{self.__class__.__name__}` metrics"
                " as no custom examples are provided."
            )
            examples = DEFAULT_FEWSHOT_EXAMPLES
        else:
            examples = self._fewshot_examples

        return examples

    def invoke(
        self,
        question: str,
        answer: str
    ) -> OutputSchema:
        input_text = json.dumps(
            {
                "question": question,
                "answer": answer
            },
            ensure_ascii=False
        )

        result = self.client.invoke(
            instruction=self.instruction,
            fewshot_examples=self.fewshot_examples,
            input_text=input_text,
            output_json_schema=OUTPUT_JSON_SCHEMA
        )

        jsonschema.validate(instance=result, schema=OUTPUT_JSON_SCHEMA)
        return result
