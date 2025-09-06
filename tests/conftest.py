import pytest

from myllmet.metrics.interface import LLMClientInterface


@pytest.fixture
def client_stub_factory():
    class DummyLLMClient(LLMClientInterface):
        def __init__(self, return_value):
            self._return_value = return_value
            self.received_invoke_params = None

        def invoke(self, instruction, fewshot_examples, input_json, output_json_schema):
            self.received_invoke_params = {
                "instruction": instruction,
                "fewshot_examples": fewshot_examples,
                "input_json": input_json,
                "output_json_schema": output_json_schema,
            }
            return self._return_value

    return DummyLLMClient
