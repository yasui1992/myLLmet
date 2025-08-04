# myLLmet

A personal LLM evaluation project — the name stands for **my LLM metrics**.  
Inspired by tools like [Ragas](https://github.com/explodinggradients/ragas), but reimplemented from scratch to better understand how LLM evaluation works.  
The LLM used in this project is provided by AWS Bedrock.

## Installation

1. To install a package from GitHub

```sh
pip install git+https://github.com/yasui1992/myllmet.git@main
```

2. To install from a local copy after cloning the repository:

```sh
git clone https://github.com/yasui1992/myllmet.git
cd myllmet
pip install .
```

3. To specify the dependency in pyproject.toml:

When using uv:

```toml
[project]
name = "example"
version = "0.1.0"
dependencies = [
    "myllmet",
]

[tool.uv.sources]
myllmet = { git = "https://github.com/yasui1992/myllmet.git", branch = "main" }
```

When using Poetry:

```toml
[tool.poetry.dependencies]
myllmet = { git = "https://github.com/yasui1992/myllmet.git", branch = "main" }
```

For installation, please follow the instructions for your package manager (e.g., run `uv sync` or `poetry install`).

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
See the [LICENSE](./LICENSE) file for full details.

## Acknowledgements

This project was inspired by [Ragas](https://github.com/explodinggradients/ragas),  
which is also licensed under the Apache License 2.0.  
While no source code has been directly copied, some prompt designs were adapted from Ragas,  
with Japanese translations derived from the original English prompts.
