# Advanced Transformer Techniques

This project contains working examples for Chapter 11 of the Hugging Face Transformers book.

## Overview

Learn how to implement and understand:

## Prerequisites

- Python 3.12 (managed via pyenv)
- Poetry for dependency management
- Go Task for build automation
- API keys for any required services (see .env.example)

## Setup

1. Clone this repository
2. Run the setup task:
   ```bash
   task setup
   ```
3. Copy `.env.example` to `.env` and configure as needed

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and utilities
│   ├── main.py                # Entry point with all examples
│   ├── prompt_engineering.py        # Prompt Engineering implementation
│   ├── few_shot_learning.py        # Few Shot Learning implementation
│   ├── chain_of_thought.py        # Chain Of Thought implementation
│   ├── constitutional_ai.py        # Constitutional Ai implementation
│   └── utils.py               # Utility functions
├── tests/
│   └── test_examples.py       # Unit tests
├── .env.example               # Environment template
├── Taskfile.yml               # Task automation
└── pyproject.toml             # Poetry configuration
```

## Running Examples

Run all examples:
```bash
task run
```

Or run individual modules:
```bash
task run-prompt-engineering    # Run prompt engineering
task run-few-shot-learning    # Run few shot learning
task run-chain-of-thought    # Run chain of thought
```

## Available Tasks

- `task setup` - Set up Python environment and install dependencies
- `task run` - Run all examples
- `task test` - Run unit tests
- `task format` - Format code with Black and Ruff
- `task clean` - Clean up generated files

## Learn More

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Library](https://github.com/huggingface/transformers)
- [Book Resources](https://example.com/book-resources)
