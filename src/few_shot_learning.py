"""Few-shot Learning Examples."""

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from config import get_device


def basic_few_shot_example():
    """Basic few-shot learning example with GPT-2."""
    print("Basic Few-Shot Learning Example...")
    
    # Load model
    model_name = "gpt2"
    generator = pipeline(
        "text-generation",
        model=model_name,
        device=0 if get_device() == "cuda" else -1
    )
    
    # Few-shot prompt for sentiment classification
    prompt = """Classify the sentiment of these reviews as positive or negative.

Review: The product arrived quickly and works perfectly.
Sentiment: positive

Review: Terrible quality, broke after one day.
Sentiment: negative

Review: Amazing value for money, highly recommend!
Sentiment: positive

Review: Complete waste of money, very disappointed.
Sentiment:"""

    # Generate
    output = generator(
        prompt,
        max_new_tokens=5,
        temperature=0.1,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    print("Few-shot prompt:")
    print(prompt)
    print("\nModel output:", output[0]['generated_text'][len(prompt):].strip())


def domain_specific_few_shot():
    """Few-shot learning for domain-specific tasks."""
    print("Domain-Specific Few-Shot Learning...")
    
    generator = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if get_device() == "cuda" else -1
    )
    
    # Medical diagnosis few-shot example
    prompt = """Based on symptoms, suggest possible conditions:

Symptoms: Chest pain, shortness of breath, sweating
Possible condition: Myocardial infarction (heart attack)

Symptoms: Frequent urination, excessive thirst, fatigue
Possible condition: Diabetes mellitus

Symptoms: Severe headache, stiff neck, sensitivity to light
Possible condition: Meningitis

Symptoms: Persistent cough, fever, difficulty breathing
Possible condition:"""

    output = generator(
        prompt,
        max_new_tokens=10,
        temperature=0.3,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    print("Medical few-shot prompt:")
    print(prompt)
    print("\nModel suggestion:", output[0]['generated_text'][len(prompt):].strip())


def structured_output_few_shot():
    """Few-shot learning for structured output generation."""
    print("Structured Output Few-Shot Example...")
    
    generator = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if get_device() == "cuda" else -1
    )
    
    # JSON generation few-shot
    prompt = """Convert natural language to JSON format:

Input: John Smith is 30 years old and lives in New York
Output: {"name": "John Smith", "age": 30, "city": "New York"}

Input: The product costs $49.99 and weighs 2.5 kg
Output: {"price": 49.99, "weight": 2.5, "unit": "kg"}

Input: Meeting scheduled for March 15 at 2:30 PM in Room 101
Output:"""

    output = generator(
        prompt,
        max_new_tokens=30,
        temperature=0.2,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    print("Structured output prompt:")
    print(prompt)
    print("\nGenerated JSON:", output[0]['generated_text'][len(prompt):].strip())


def run_few_shot_learning_examples():
    """Run all few-shot learning examples."""
    
    print("Running few-shot learning examples...")
    print("These demonstrate in-context learning without fine-tuning.\n")
    
    basic_few_shot_example()
    print("\n" + "-" * 60 + "\n")
    
    domain_specific_few_shot()
    print("\n" + "-" * 60 + "\n")
    
    structured_output_few_shot()


if __name__ == "__main__":
    run_few_shot_learning_examples()