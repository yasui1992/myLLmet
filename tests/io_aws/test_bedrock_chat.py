import json

import pytest
from botocore.exceptions import ClientError

from myllmet.io_aws import BedrockChatClient


@pytest.fixture
def chat_client(mocker):
    fake_client = mocker.Mock()
    mocker.patch("boto3.client", return_value=fake_client)
    return BedrockChatClient(model_id="dummy-model", bedrock_runtime_client=fake_client)


@pytest.fixture
def json_schema():
    return {
        "type": "object",
        "properties": {
            "output": {"type": "string"}
        },
        "required": ["output"]
    }


def test_invoke_valid(chat_client, json_schema, mocker):
    client_return = {
        "stopReason": "end_turn",
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": json.dumps({"output": "output_text"})}]
            }
        }
    }
    mocker.patch.object(chat_client._client, "converse", return_value=client_return)

    expected = {"output": "output_text"}
    actual = chat_client.invoke(
        instruction="instruction",
        fewshot_examples=[],
        input_json={"input": "input_text"},
        output_json_schema=json_schema
    )

    assert actual == expected

def test_invoke_invalid_stop_reason(chat_client, json_schema, mocker):
    client_return = {
        "stopReason": "other_reason",
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": json.dumps({"output": "output_text"})}]
            }
        }
    }
    mocker.patch.object(chat_client._client, "converse", return_value=client_return)

    with pytest.raises(ValueError):
        chat_client.invoke(
            instruction="instruction",
            fewshot_examples=[],
            input_json={"input": "input_text"},
            output_json_schema=json_schema
        )

def test_invoke_invalid_role(chat_client, json_schema, mocker):
    client_return = {
        "stopReason": "end_turn",
        "output": {
            "message": {
                "role": "user",
                "content": [{"text": json.dumps({"output": "output_text"})}]
            }
        }
    }
    mocker.patch.object(chat_client._client, "converse", return_value=client_return)

    with pytest.raises(ValueError):
        chat_client.invoke(
            instruction="instruction",
            fewshot_examples=[],
            input_json={"input": "input_text"},
            output_json_schema=json_schema
        )

def test_parse_response_multiple_contents(chat_client, json_schema, mocker):
    client_return = {
        "stopReason": "end_turn",
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": json.dumps({"output": "output_text_1"})},
                    {"text": json.dumps({"output": "output_text_2"})}
                ]
            }
        }
    }
    mocker.patch.object(chat_client._client, "converse", return_value=client_return)

    with pytest.raises(ValueError):
        chat_client.invoke(
            instruction="instruction",
            fewshot_examples=[],
            input_json={"input": "input_text"},
            output_json_schema=json_schema
        )


def test_invoke_throttling_retry(chat_client, json_schema, mocker):
    side_effects = [
        ClientError({"Error": {"Code": "ThrottlingException"}}, "converse"),
        {
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": json.dumps({"output": "output_text"})}]
                }
            }
        }
    ]
    mocker.patch.object(chat_client._client, "converse", side_effect=side_effects)
    mocker.patch("time.sleep", lambda x: None)  # No actual sleep during tests

    chat_client.invoke(
        instruction="instruction",
        fewshot_examples=[],
        input_json={"input": "input_text"},
        output_json_schema=json_schema
    )

    assert chat_client._client.converse.call_count == 2
