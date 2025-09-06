from typing import Any, Dict, List, Protocol, TypedDict, runtime_checkable

type JSONType = Dict[str, Any]
type JSONSchema = Dict[str, Any]


class FewShotExample(TypedDict):
    user: JSONSchema
    assistant: JSONSchema


@runtime_checkable
class LLMClientInterface(Protocol):
    def invoke(
        self,
        instruction: str,
        fewshot_examples: List[FewShotExample],
        input_text: str,
        output_json_schema: JSONSchema,
    ) -> JSONType:
        ...
