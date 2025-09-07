from typing import Any, Dict, Generic, Protocol, TypedDict, TypeVar, runtime_checkable

type JSONSchema = Dict[str, Any]

IS = TypeVar("IS", contravariant=True)  # Assume a TypedDict-derived type
OS = TypeVar("OS", covariant=True)  # Assume a TypedDict-derived type


class LLMMetricsRecord(TypedDict):
    question: str
    answer: str
    context: str
    ground_truth: str
    score: float
    prompts: Dict[str, Any]
    intermediates: Dict[str, Any]


@runtime_checkable
class LLMClientInterface(Protocol, Generic[IS, OS]):
    def invoke(
        self,
        instruction: str,
        fewshot_examples,  # List of TypedDict with "user" (resp. "assistant") which is mapped to IS (resp. OS)
        input_json: IS,
        output_json_schema: JSONSchema,
    ) -> OS: ...


@runtime_checkable
class TrackerInterface(Protocol):
    def log(self, record: LLMMetricsRecord) -> None:
        pass
