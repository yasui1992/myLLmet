from typing import Any, Dict, Generic, Protocol, TypeVar, runtime_checkable

type JSONSchema = Dict[str, Any]

IS = TypeVar("IS", contravariant=True)  # Assume a TypedDict-derived type
OS = TypeVar("OS", covariant=True)  # Assume a TypedDict-derived type


@runtime_checkable
class LLMClientInterface(Protocol, Generic[IS, OS]):
    def invoke(
        self,
        instruction: str,
        fewshot_examples,
        input_json: IS,
        output_json_schema: JSONSchema,
    ) -> OS: ...
