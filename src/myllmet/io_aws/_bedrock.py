import boto3

from .exceptions import BedrockClientError


class BedrockClient:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client = boto3.client("bedrock-runtime")

    def _parse_response(self, response) -> str:
        stop_reason = response["stopReason"]

        if stop_reason != "end_turn":
            raise BedrockClientError(
                f"Supported only `end_turn` stop reason. Got: {stop_reason}"
            )

        message = response["output"]["message"]
        role = message["role"]
        contents = message["content"]

        if role != "assistant":
            raise BedrockClientError(
                f"Supported only `assistant` role. Got: {role}"
            )

        if len(contents) > 1:
            raise BedrockClientError(
                f"Supported only single content. Got: {len(contents)}"
            )

        return contents[0]["text"]


    def single_turn_chat(self, user_text: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"text": user_text}]
            },
        ]

        response = self._client.converse(
            modelId=self.model_id,
            messages=messages
        )

        llm_text = self._parse_response(response)

        return llm_text
