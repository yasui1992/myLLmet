import pytest
from unittest.mock import MagicMock, patch
from myllmet.io_aws import BedrockClient
from myllmet.io_aws.exceptions import BedrockClientError


@pytest.mark.parametrize("supported_response", [
    {
        "stopReason": "end_turn",
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello from mock"}]
            }
        }
    }
])
@patch("boto3.client")
def test_single_turn_chat_success(mock_boto_client, supported_response):
    mock_client = MagicMock()
    mock_client.converse.return_value = supported_response
    mock_boto_client.return_value = mock_client
    bedrock_client = BedrockClient(model_id="model-id")

    text = bedrock_client.single_turn_chat("Hello")

    assert text == "Hello from mock"


@pytest.mark.parametrize("unsupported_response", [
    pytest.param(
        {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello"}]
                }
            }
        },
        id="stop_reason"
    ),
    pytest.param(
        {
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "role": "user",
                    "content": [{"text": "Hello"}]
                }
            }
        },
        id="role"
    ),
    pytest.param(
        {
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello"}, {"text": "World"}]
                }
            }
        },
        id="multiple_contents"
    ),
])
@patch("boto3.client")
def test_single_turn_chat_failed(mock_boto_client, unsupported_response):
    mock_client = MagicMock()
    mock_client.converse.return_value = unsupported_response
    mock_boto_client.return_value = mock_client
    bedrock_client = BedrockClient(model_id="model-id")

    with pytest.raises(BedrockClientError):
        bedrock_client.single_turn_chat("Hello")
