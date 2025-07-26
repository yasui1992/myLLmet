import logging
import json
import time

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from .exceptions import BedrockClientError


logger = logging.getLogger(__name__)


class BedrockClient:
    def __init__(
        self,
        model_id: str,
        max_attempts: int = 5,
        max_wait: int = 60
    ):
        self.model_id = model_id
        self.max_attempts = max_attempts
        self.max_wait = max_wait
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

    def _call_converse_api(self, messages, converse_kwargs=None):
        logger.debug(f"Calling converse API with model ID: {self.model_id}")

        request = {"messages": messages} | (converse_kwargs or {})
        logger.debug(f"Sending request: {json.dumps(request, ensure_ascii=False)}")

        response = self._client.converse(
            modelId=self.model_id,
            **request
        )

        logger.debug(f"Received response: {json.dumps(response, ensure_ascii=False)}")
        return response

    def chat(self, user_text: str, *, converse_kwargs: dict | None = None) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"text": user_text}]
            },
        ]

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._call_converse_api(
                    messages,
                    converse_kwargs=converse_kwargs
                )

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

        return llm_text
