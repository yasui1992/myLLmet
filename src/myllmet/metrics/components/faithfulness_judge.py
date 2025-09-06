# This module includes prompt designs adapted and translated
# from the RAGAS project (https://github.com/explodinggradients/ragas).
# RAGAS is licensed under the Apache License 2.0.
# No source code from RAGAS has been copied or included.


import json
import logging
from typing import List, Optional, TypedDict

import jsonschema

from myllmet.metrics.interface import LLMClientInterface

logger = logging.getLogger(__name__)


class SingleFaithfulnessJudgResult(TypedDict):
    claim: str
    verdict: int  # 1 or 0
    reason: str


class InputSchema(TypedDict):
    context: str
    claims: List[str]


class OutputSchema(TypedDict):
    verdicts: List[SingleFaithfulnessJudgResult]


class FewShotExample(TypedDict):
    user: InputSchema
    assistant: OutputSchema


OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "与えられた主張"
                    },
                    "verdict": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "忠実性の判定結果 (0=False, 1=True)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "判定の理由"
                    }
                },
                "required": ["claim", "verdict", "reason"],
            }
        }
    },
    "required": ["verdicts"],
}


DEFAULT_INSTRUCTION = (
    "あなたは日本語の言語分析のAIアシスタントです。\n"
    "あなたのタスクは、与えられたコンテキストに基づいて、個々の主張の忠実性を判断することです。\n"
    "各主張について、文脈から直接推論できる場合は「1」、直接推論できない場合は「0」を返してください。\n"
)


DEFAULT_FEWSHOT_EXAMPLES: List[FewShotExample] = [
    {
        "user": {
            "context": "彼はドイツ生まれの理論物理学者であり、相対性理論の提唱で知られています。",
            "claims": [
                "アルベルト・アインシュタインは、ドイツ生まれの理論物理学者です。",
                "アルベルト・アインシュタインは、アメリカ生まれの理論物理学者です。",
                "アルベルト・アインシュタインは、量子力学の理論発展に重要な貢献をしました。",
            ]
        },
        "assistant": {
            "verdicts": [
                {
                    "claim": "アルベルト・アインシュタインは、ドイツ生まれの理論物理学者です。",
                    "verdict": 1,
                    "reason": "コンテキストの記載と合致しています。"
                },
                {
                    "claim": "アルベルト・アインシュタインは、アメリカ生まれの理論物理学者です。",
                    "verdict": 0,
                    "reason": "コンテキストによると、アインシュタインはドイツ生まれです"
                },
                {
                    "claim": "アルベルト・アインシュタインは、量子力学の理論発展に重要な貢献をしました。",
                    "verdict": 0,
                    "reason": "コンテキストには、アインシュタインの量子力学への貢献は記載されていません。"
                }
            ]
        }
    }
]


class FaithfulnessJudge:
    def __init__(
        self,
        client: LLMClientInterface[InputSchema, OutputSchema],
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
        context: str,
        claims: List[str]
    ) -> OutputSchema:

        input_json: InputSchema = {
            "context": context,
            "claims": claims
        }

        result = self.client.invoke(
            instruction=self.instruction,
            fewshot_examples=self.fewshot_examples,
            input_json=input_json,
            output_json_schema=OUTPUT_JSON_SCHEMA
        )

        jsonschema.validate(instance=result, schema=OUTPUT_JSON_SCHEMA)
        return result
