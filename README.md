# Advanced Transformer Techniques - Chapter 11 Implementation

This project contains comprehensive working examples for Chapter 11: **Dataset Curation and Training Language Models from Scratch** from the Hugging Face Transformers book.

**Repository**: https://github.com/RichardHightower/art_hug_11

## Overview

This project demonstrates the complete lifecycle of building custom language models, from raw data curation to production-ready AI systems. Learn how to implement and understand:

- **Data Curation Fundamentals**: Selecting, cleaning, and preparing domain-specific text data
- **Scalable Processing Techniques**: Handling massive datasets efficiently with streaming and batching
- **Privacy Protection**: Comprehensive PII redaction and data security practices
- **Bias Detection & Mitigation**: Ensuring fair and ethical AI development
- **Custom Tokenizer Training**: Building domain-specific vocabularies for improved efficiency
- **Modern Model Architectures**: Configuration and selection strategies
- **Parameter-Efficient Fine-Tuning**: PEFT methods including LoRA and QLoRA
- **Training Workflows**: Distributed computing, experiment tracking, and monitoring
- **Advanced Techniques**: Few-shot learning, chain of thought reasoning, and synthetic data generation

## Prerequisites

- Python 3.12 (managed via pyenv)
- Poetry for dependency management
- Go Task for build automation
- API keys for any required services (see .env.example)

## Setup

1. Clone this repository
   ```bash 
   git clone git@github.com:RichardHightower/art_hug_11.git
   ```
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
│   ├── config.py                    # Configuration utilities with device selection and API key validation
│   ├── main.py                      # Entry point with orchestrated examples
│   ├── prompt_engineering.py       # Advanced prompt engineering techniques
│   ├── few_shot_learning.py        # Few-shot learning implementations
│   ├── chain_of_thought.py         # Chain of thought reasoning examples
│   ├── constitutional_ai.py        # Constitutional AI implementations
│   └── utils.py                     # Shared utility functions
├── notebooks/
│   ├── tutorial.ipynb               # Comprehensive tutorial covering all aspects
│   ├── medical_tokenizer.json      # Trained domain-specific tokenizer
│   └── custom_tokenizer.json       # Additional tokenizer artifacts
├── docs/
│   └── article11.md                 # Comprehensive guide on dataset curation (1000+ lines)
├── tests/
│   └── test_examples.py             # Unit tests for all implementations
├── data/                            # Data directory (created during execution)
├── models/                          # Model artifacts directory (created during execution)
├── .env.example                     # Environment template with API key examples
├── Taskfile.yml                     # Cross-platform task automation
├── pyproject.toml                   # Poetry configuration with locked dependencies
└── README.md                        # This file
```

## Getting Started

### 🚀 Quick Start with Jupyter Notebooks

For the best learning experience, start with the comprehensive tutorial:

```bash
# 1. Set up the environment
task setup


task run # 2. Run all examples

# 3. Start Jupyter
jupyter lab notebooks/tutorial.ipynb
```

The tutorial notebook covers:
- Complete data curation pipeline
- Custom tokenizer training
- Model configuration and PEFT methods
- Training workflows and monitoring
- Advanced techniques and best practices

### 📝 Running Code Examples

Run all examples from the command line:
```bash
task run
```

Or run individual modules:
```bash
task run-prompt-engineering    # Advanced prompt engineering techniques
task run-few-shot-learning     # Few-shot learning implementations  
task run-chain-of-thought      # Chain of thought reasoning examples
```

## Available Tasks

- `task setup` - Set up Python 3.12.9 environment and install dependencies via Poetry
- `task run` - Run all examples from src/main.py
- `task run-prompt-engineering` - Run prompt engineering examples only
- `task run-few-shot-learning` - Run few shot learning examples only  
- `task run-chain-of-thought` - Run chain of thought examples only
- `task test` - Run all tests with pytest
- `task format` - Format code with Black (line-length: 88) and Ruff
- `task clean` - Clean up generated files and caches

## Key Features

### 🔧 Production-Ready Components

- **Comprehensive PII Redaction**: Multiple approaches from basic regex to transformer-based detection
- **Custom Tokenizer Training**: Domain-specific vocabularies with BPE and statistical analysis
- **Parameter-Efficient Fine-Tuning**: LoRA, QLoRA, and other PEFT implementations
- **Training Diagnostics**: Automated issue detection and resolution recommendations
- **Data Versioning**: Track dataset changes for reproducible ML workflows

### 📊 Educational Resources

- **Interactive Visualizations**: Training progress, tokenizer comparisons, pipeline diagrams
- **Best Practices**: Production deployment patterns and common pitfall avoidance
- **Comprehensive Examples**: Real-world scenarios with medical, financial, and technical domains
- **Performance Analysis**: Memory usage, cost comparisons, and efficiency metrics

## Technology Stack

- **Python 3.12.9** (managed via pyenv)
- **Hugging Face Transformers 4.53.3** for model implementations
- **Datasets 2.14.4** for efficient data processing
- **PEFT Library** for parameter-efficient fine-tuning
- **PyTorch** for deep learning operations
- **Poetry** for modern Python dependency management
- **Task** for cross-platform build automation
- **Jupyter Lab** for interactive development

## API Key Configuration

To use examples requiring external APIs:

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Add your API keys to `.env`:
   ```env
   OPENAI_API_KEY=your-openai-key-here
   ANTHROPIC_API_KEY=your-anthropic-key-here  
   HUGGINGFACE_TOKEN=your-huggingface-token-here
   ```

3. Restart your Python environment to load the new variables

## Learn More

- **Repository**: https://github.com/RichardHightower/art_hug_11
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Library](https://github.com/huggingface/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Datasets Documentation](https://huggingface.co/docs/datasets)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes and improvements
- Additional examples and techniques
- Documentation enhancements
- Performance optimizations

## License

This project is open source and available under the MIT License.
