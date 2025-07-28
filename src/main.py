"""Main entry point for all examples."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from prompt_engineering import run_prompt_engineering_examples
from few_shot_learning import run_few_shot_learning_examples
from chain_of_thought import run_chain_of_thought_examples
from constitutional_ai import run_constitutional_ai_examples
from data_curation import run_data_curation_examples
from tokenization import run_tokenization_examples
from model_configuration import run_model_configuration_examples
from training_workflow import run_training_workflow_examples

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def main():
    """Run all examples."""
    print_section("CHAPTER 11: DATASET CURATION & TRAINING LLMs")
    print("Welcome! This script demonstrates building custom language models.")
    print("From data curation to training workflows.\n")
    
    print_section("1. DATA CURATION & CLEANING")
    run_data_curation_examples()
    
    print_section("2. TOKENIZATION & VOCABULARY")
    run_tokenization_examples()
    
    print_section("3. MODEL CONFIGURATION")
    run_model_configuration_examples()
    
    print_section("4. TRAINING WORKFLOWS")
    run_training_workflow_examples()
    
    print_section("5. PROMPT ENGINEERING")
    run_prompt_engineering_examples()
    
    print_section("6. FEW SHOT LEARNING")
    run_few_shot_learning_examples()
    
    print_section("7. CHAIN OF THOUGHT")
    run_chain_of_thought_examples()
    
    print_section("CONCLUSION")
    print("You've learned the complete pipeline from data to deployment!")
    print("Next steps: Try fine-tuning on your own domain-specific data.")

if __name__ == "__main__":
    main()
