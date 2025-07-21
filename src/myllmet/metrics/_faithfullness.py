from myllmet.io_aws import BedrockClient

_DEFAULT_INSTRUCTION = (
    "You are an expert in Japanese language analysis.\n"
    "Given a question and its corresponding answer, analyze the answer by breaking it down into one or more distinct claims.\n"
    "Do not use any pronouns in the claims.\n"
    "Each claim must be fully self-contained and understandable without relying on any prior context.\n"
)


class FaithfulnessClaimBreakDownPromptBuilder:
    def __init__(
        self,
        instruction: str | None = None
    ):
        self.instruction = instruction or self._DEFAULT_INSTRUCTION

    def build(self, question: str, answer: str) -> str:
        return (
            f"{self.instruction}\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            "Break down the answer into distinct claims:\n"
        )


class Faithfulness:
    def __init__(self, judger: BedrockClient):
        self.judger = judger

    def _extract_claims(self, answer: str) -> list[str]:
        # Placeholder for claim extraction logic
        # This should be replaced with actual logic to extract claims from the answer
        return [answer]

    def score(
        self,
        question: str,  # noqa: F401
        answer: str,
        retrieved_contexts: list[str] | None = None,   # noqa: F401
        ground_truth: str | None = None,  # noqa: F401
    ) -> float:
        claims = self._extract_claims(answer)

        for c in claims:
            self.judger.chat()
