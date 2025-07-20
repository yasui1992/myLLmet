import pytest
from unittest.mock import MagicMock, patch
from myllmet.io_aws import BedrockClient
from myllmet.io_aws.exceptions import BedrockClientError


_SUPPORTED_RESPONSE = {
    "stopReason": "end_turn",
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hello from mock"}]
        }
    }
}
_UNSUPPORTED_RESPONSE_STOP_REASON = {
    "stopReason": "tool_use",
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hello"}]
        }
    }
}
_UNSUPPORTED_RESPONSE_ROLE = {
    "stopReason": "end_turn",
    "output": {
        "message": {
            "role": "user",
            "content": [{"text": "Hello"}]
        }
    }
}
_UNSUPPORTED_RESPONSE_MULTIPLE_CONTENTS = {
    "stopReason": "end_turn",
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hello"}, {"text": "World"}]
        }
    }
}


@pytest.mark.parametrize(
    "supported_response",
    [pytest.param(_SUPPORTED_RESPONSE, id="supported_response")]
)
@patch("boto3.client")
def test_single_turn_chat_success(mock_boto_client, supported_response):
    mock_client = MagicMock()
    mock_client.converse.return_value = supported_response
    mock_boto_client.return_value = mock_client
    bedrock_client = BedrockClient(model_id="model-id")

    text = bedrock_client.single_turn_chat("Hello")

    assert text == "Hello from mock"


@pytest.mark.parametrize(
    "supported_response",
    [pytest.param(_SUPPORTED_RESPONSE, id="supported_response")]
)
@patch("boto3.client")
def test_single_turn_chat_kwargs_system_success(mock_boto_client, supported_response):
    mock_client = MagicMock()
    mock_client.converse.return_value = supported_response
    mock_boto_client.return_value = mock_client

    system = [{"text": "You are a helpful assistant."}]
    bedrock_client = BedrockClient(
        model_id="model-id",
        converse_kwargs={"system": system}
    )

    text = bedrock_client.single_turn_chat("Hello")

    assert text == "Hello from mock"



@pytest.mark.parametrize(
    "unsupported_response",
    [
        pytest.param(_UNSUPPORTED_RESPONSE_STOP_REASON, id="stop_reason"),
        pytest.param(_UNSUPPORTED_RESPONSE_ROLE, id="role"),
        pytest.param(_UNSUPPORTED_RESPONSE_MULTIPLE_CONTENTS, id="multiple_contents")
    ]
)
@patch("boto3.client")
def test_single_turn_chat_failed(mock_boto_client, unsupported_response):
    mock_client = MagicMock()
    mock_client.converse.return_value = unsupported_response
    mock_boto_client.return_value = mock_client
    bedrock_client = BedrockClient(model_id="model-id")

    with pytest.raises(BedrockClientError):
        bedrock_client.single_turn_chat("Hello")
