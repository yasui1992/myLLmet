import json
import logging
import time
from typing import Generic, List, Optional

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from myllmet.metrics.interface import IS, OS, JSONSchema, LLMClientInterface

from .exceptions import BedrockClientError

logger = logging.getLogger(__name__)


class BedrockChatClient(LLMClientInterface, Generic[IS, OS]):
    def __init__(
        self,
        model_id: str,
        max_attempts: int = 5,
        max_wait: int = 60,
        bedrock_runtime_client: Optional[BaseClient] = None
    ):
        self.model_id = model_id
        self.max_attempts = max_attempts
        self.max_wait = max_wait
        self._client = bedrock_runtime_client or boto3.client("bedrock-runtime")

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

    def _call_converse_api(self, system, messages, converse_kwargs=None):
        logger.debug(f"Calling converse API with model ID: {self.model_id}")

        # TODO: Resolve temperature hardcoded value
        request = {"system": system, "messages": messages} \
            | {"inferenceConfig": {"temperature": 0.0}} \
            | (converse_kwargs or {})
        logger.debug(f"Sending request: {json.dumps(request, ensure_ascii=False)}")

        response = self._client.converse(
            modelId=self.model_id,
            **request
        )

        logger.debug(f"Received response: {json.dumps(response, ensure_ascii=False)}")
        return response

    def _build_messages(
        self,
        fewshot_examples: List,
        input_json: IS,
    ) -> List:

        messages = []
        for ex in fewshot_examples:
            messages += [
                {
                    "role": "user",
                    "content": [
                        {"text": json.dumps(ex["user"], ensure_ascii=False)}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"text": json.dumps(ex["assistant"], ensure_ascii=False)}
                    ]
                }
            ]
        messages.append({
            "role": "user",
            "content": [
                {"text": json.dumps(input_json, ensure_ascii=False)}
            ]
        })

        return messages

    def _build_system_prompt(
        self,
        instruction: str,
        output_json_schema: JSONSchema
    ) -> str:

        system = (
            "<Instruction>\n"
            f"{instruction}\n"
            "</Instruction>\n"
            "<Output Format>\n"
            f"{json.dumps(output_json_schema, ensure_ascii=False)}\n"
            "</Output Format>"
        )

        return system

    def invoke(
        self,
        instruction: str,
        fewshot_examples: List,
        input_json: IS,
        output_json_schema: JSONSchema,
    ) -> OS:

        system = [{"text": self._build_system_prompt(instruction, output_json_schema)}]
        messages = self._build_messages(fewshot_examples, input_json)

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._call_converse_api(system, messages)

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "ThrottlingException":
                    logger.debug(f"ThrottlingException occurred: {e}.")
                else:
                    logger.debug(f"Unsupported ClientError occurred: {e}.")

                if attempt < self.max_attempts:
                    wait_time = min(2 ** attempt, self.max_wait)
                    logger.debug(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max attempts reached ({self.max_attempts}).")
                    raise
            else:
                break

        llm_text = self._parse_response(response)
        result = json.loads(llm_text)

        return result
