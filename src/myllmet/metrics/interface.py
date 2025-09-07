from typing import Any, Dict, Generic, Protocol, TypeVar, runtime_checkable, TypedDict, List

type JSONSchema = Dict[str, Any]

IS = TypeVar("IS")  # Assume a TypedDict-derived type
OS = TypeVar("OS")  # Assume a TypedDict-derived type


class FewshotExample(TypedDict, Generic[IS, OS]):
    user: IS
    assistant: OS


@runtime_checkable
class LLMClientInterface(Protocol, Generic[IS, OS]):
    def invoke(
        self,
        instruction: str,
        fewshot_examples: List[FewshotExample[IS, OS]],
        input_json: IS,
        output_json_schema: JSONSchema,
    ) -> OS: ...


@runtime_checkable
class TrackerInterface(Protocol):
    def log(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str,
        score: float,
        intermediates: Dict[str, Any],
        prompts: Dict[str, Any],
    ) -> None: ...
