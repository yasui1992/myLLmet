import jsonschema
import pytest

from myllmet.metrics.components import ClaimExtractor
from myllmet.metrics.components.claim_extractor import (
    DEFAULT_FEWSHOT_EXAMPLES,
    DEFAULT_INSTRUCTION,
    OUTPUT_JSON_SCHEMA,
    OutputSchema,
)


def test_invoke_valid(client_stub_factory):
    return_value: OutputSchema = {"claims": ["c1", "c2", "c3"]}
    client = client_stub_factory(
        return_value=return_value
    )
    extractor = ClaimExtractor(client=client)

    expected = return_value
    actual = extractor.invoke("question", "answer")

    assert actual == expected


def test_invoke_args_passed_correctly(client_stub_factory):
    return_value: OutputSchema = {"claims": ["c1", "c2", "c3"]}
    client = client_stub_factory(
        return_value=return_value
    )
    extractor = ClaimExtractor(client=client)

    expected_input_json = {
        "question": "question",
        "answer": "answer"
    }
    expected_output_json_schema = OUTPUT_JSON_SCHEMA
    expected_instruction = DEFAULT_INSTRUCTION
    expected_fewshot_examples = DEFAULT_FEWSHOT_EXAMPLES

    extractor.invoke("question", "answer")

    params = client.received_invoke_params
    assert params["input_json"] == expected_input_json
    assert params["output_json_schema"] == expected_output_json_schema
    assert params["instruction"] == expected_instruction
    assert params["fewshot_examples"] == expected_fewshot_examples


def test_invoke_with_custom_instruction_and_fewshot_examples(client_stub_factory):
    return_value: OutputSchema = {"claims": ["custom1", "custom2"]}
    client = client_stub_factory(return_value=return_value)

    custom_instruction = "Custom instruction"
    custom_fewshot_examples = [
        {
            "user": {"question": "q1", "answer": "a1"},
            "assistant": {"claims": ["c1"]}
        }
    ]
    extractor = ClaimExtractor(
        client=client,
        instruction=custom_instruction,
        fewshot_examples=custom_fewshot_examples
    )

    expected_instruction = custom_instruction
    expected_fewshot_examples = custom_fewshot_examples
    extractor.invoke("q1", "a1")

    params = client.received_invoke_params
    assert params["instruction"] == expected_instruction
    assert params["fewshot_examples"] == expected_fewshot_examples


def test_invoke_invalid_schema(client_stub_factory):
    bad_output = {"claims": "not-a-list"}
    client = client_stub_factory(return_value=bad_output)
    extractor = ClaimExtractor(client=client)

    with pytest.raises(jsonschema.ValidationError):
        extractor.invoke("question", "answer")
