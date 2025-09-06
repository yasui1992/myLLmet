import jsonschema
import pytest

from myllmet.metrics.components import FaithfulnessJudge
from myllmet.metrics.components.faithfulness_judge import (
    DEFAULT_FEWSHOT_EXAMPLES,
    DEFAULT_INSTRUCTION,
    OUTPUT_JSON_SCHEMA,
    OutputSchema,
)


def test_invoke_valid(client_stub_factory):
    return_value: OutputSchema = {
        "verdicts": [
            {"claim": "c1", "verdict": 1, "reason": "r1"},
            {"claim": "c2", "verdict": 0, "reason": "r2"},
        ]
    }
    client = client_stub_factory(return_value=return_value)
    judge = FaithfulnessJudge(client=client)

    actual = judge.invoke("context text", ["c1", "c2"])
    assert actual == return_value


def test_invoke_args_passed_correctly(client_stub_factory):
    return_value: OutputSchema = {
        "verdicts": [
            {"claim": "c1", "verdict": 1, "reason": "r1"}
        ]
    }
    client = client_stub_factory(return_value=return_value)
    judge = FaithfulnessJudge(client=client)

    context = "sample context"
    claims = ["c1", "c2"]

    judge.invoke(context, claims)

    params = client.received_invoke_params
    assert params["input_json"] == {"context": context, "claims": claims}
    assert params["output_json_schema"] == OUTPUT_JSON_SCHEMA
    assert params["instruction"] == DEFAULT_INSTRUCTION
    assert params["fewshot_examples"] == DEFAULT_FEWSHOT_EXAMPLES


def test_invoke_with_custom_instruction_and_fewshot(client_stub_factory):
    return_value: OutputSchema = {
        "verdicts": [
            {"claim": "c1", "verdict": 1, "reason": "r1"}
        ]
    }
    client = client_stub_factory(return_value=return_value)

    custom_instruction = "Custom instruction"
    custom_fewshot = [
        {
            "user": {"context": "ctx", "claims": ["c1"]},
            "assistant": {"verdicts": [{"claim": "c1", "verdict": 1, "reason": "r1"}]}
        }
    ]

    judge = FaithfulnessJudge(
        client=client,
        instruction=custom_instruction,
        fewshot_examples=custom_fewshot
    )

    judge.invoke("ctx", ["c1"])

    params = client.received_invoke_params
    assert params["instruction"] == custom_instruction
    assert params["fewshot_examples"] == custom_fewshot


def test_invoke_invalid_schema(client_stub_factory):
    bad_output = {"verdicts": "not-a-list"}
    client = client_stub_factory(return_value=bad_output)
    judge = FaithfulnessJudge(client=client)

    with pytest.raises(jsonschema.ValidationError):
        judge.invoke("context", ["c1"])
