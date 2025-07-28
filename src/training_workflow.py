"""Training Workflow Examples."""

from transformers import (
    Trainer, TrainingArguments, EarlyStoppingCallback,
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import Dataset
import evaluate
import torch
from config import get_device, MODELS_DIR
import numpy as np


def create_sample_dataset():
    """Create a sample dataset for training."""
    # Sample medical texts
    texts = [
        "The patient presented with chest pain and shortness of breath.",
        "Diagnosis confirmed myocardial infarction based on ECG results.",
        "Treatment included aspirin and thrombolytic therapy.",
        "Post-operative care following cardiac surgery is essential.",
        "Regular monitoring of cardiac function recommended.",
        "Patient history includes hypertension and diabetes.",
        "Electrocardiogram showed ST-segment elevation.",
        "Cardiac catheterization revealed significant stenosis."
    ]
    
    return Dataset.from_dict({"text": texts})


def basic_training_setup():
    """Basic training setup with logging."""
    print("Basic Training Setup...")
    
    # Load small model for demonstration
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = create_sample_dataset()
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=128
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Split into train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )
    
    # Training arguments - simplified for compatibility
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=5,
        logging_steps=5,
        save_steps=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        warmup_steps=10,
        logging_dir='./logs',
        report_to=[],  # Disable reporting for compatibility
        load_best_model_at_end=False,  # Simplify for demo
    )
    
    try:
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            data_collator=data_collator,
        )
    except TypeError as e:
        print(f"Note: Trainer initialization issue (likely version mismatch): {e}")
        print("Creating a simplified training demo instead...")
        return None
    
    print("Training configuration:")
    print(f"  - Model: {model_name}")
    print(f"  - Train samples: {len(split_dataset['train'])}")
    print(f"  - Eval samples: {len(split_dataset['test'])}")
    print(f"  - Device: {get_device()}")
    
    return trainer


def training_with_metrics():
    """Training with custom metrics and evaluation."""
    print("Training with Metrics...")
    
    # Setup trainer
    trainer = basic_training_setup()
    
    if trainer is None:
        print("Skipping metrics example due to trainer initialization issue")
        return
    
    # Load evaluation metrics
    accuracy = evaluate.load("accuracy")
    perplexity = evaluate.load("perplexity")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Calculate perplexity
        loss = predictions[0] if isinstance(predictions, tuple) else predictions
        perplexity_score = np.exp(np.mean(loss))
        
        return {
            "perplexity": perplexity_score
        }
    
    # Update trainer with metrics
    trainer.compute_metrics = compute_metrics
    
    print("\nStarting training with metrics...")
    # Note: We'll do a minimal training for demonstration
    trainer.args.num_train_epochs = 1
    trainer.args.max_steps = 10
    
    # Train
    train_result = trainer.train()
    
    print("\nTraining completed!")
    print(f"Final loss: {train_result.training_loss:.4f}")
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Eval loss: {eval_results['eval_loss']:.4f}")
    if 'eval_perplexity' in eval_results:
        print(f"Eval perplexity: {eval_results['eval_perplexity']:.4f}")


def early_stopping_example():
    """Training with early stopping."""
    print("Early Stopping Example...")
    
    # Setup trainer
    trainer = basic_training_setup()
    
    if trainer is None:
        print("Skipping early stopping example due to trainer initialization issue")
        return
    
    # Add early stopping callback
    trainer.add_callback(
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
    )
    
    print("\nTraining with early stopping (patience=3)...")
    trainer.args.num_train_epochs = 10  # Set high to test early stopping
    trainer.args.max_steps = 20
    
    # Train
    trainer.train()
    
    print("\nTraining completed (may have stopped early)")


def mixed_precision_training():
    """Training with mixed precision for efficiency."""
    print("Mixed Precision Training...")
    
    # Check if GPU supports mixed precision
    device = get_device()
    fp16_available = device == "cuda"  # Only CUDA supports fp16 in transformers
    
    if not fp16_available:
        print(f"Mixed precision (fp16) not available on {device}. Using fp32.")
        print("Note: fp16 is only supported on CUDA devices in transformers")
    
    # Training arguments with mixed precision
    training_args = TrainingArguments(
        output_dir="./results_fp16",
        evaluation_strategy="steps",
        eval_steps=5,
        save_steps=10,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        fp16=fp16_available,  # Enable mixed precision only if CUDA available
        logging_steps=5,
        report_to=[],  # Disable reporting for this example
        max_steps=10,  # Limit steps for demonstration
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create simple dataset
    dataset = create_sample_dataset()
    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], padding=True, truncation=True, max_length=128),
        batched=True
    )
    
    # Trainer
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            tokenizer=tokenizer,
        )
        
        print(f"\nTraining with fp16={fp16_available}")
        trainer.train()
        
        print("Mixed precision training completed!")
    except Exception as e:
        print(f"Note: Training failed due to: {e}")
        print("This is likely due to version compatibility issues")


def error_analysis():
    """Sampling model outputs for error analysis."""
    print("Error Analysis Example...")
    
    # Train a small model first
    trainer = basic_training_setup()
    
    if trainer is None:
        print("Skipping error analysis due to trainer initialization issue")
        print("Using pre-trained model for generation example instead...")
        
        # Use pre-trained model directly
        model_name = "distilgpt2"
        text_generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if get_device() == "cuda" else -1
        )
    else:
        trainer.args.max_steps = 10
        trainer.train()
        
        # Save the model
        output_dir = "./results/checkpoint-final"
        trainer.save_model(output_dir)
        
        # Load for generation
        text_generator = pipeline(
            "text-generation",
            model=output_dir,
            tokenizer=trainer.tokenizer,
            device=0 if get_device() == "cuda" else -1
        )
    
    # Test prompts
    prompts = [
        "The patient presented with",
        "Diagnosis confirmed",
        "Treatment included"
    ]
    
    print("\nGenerating samples for error analysis:")
    for prompt in prompts:
        output = text_generator(
            prompt,
            max_length=50,
            num_return_sequences=1,
            temperature=0.8,
            pad_token_id=text_generator.tokenizer.eos_token_id
        )
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {output[0]['generated_text']}")
    
    print("\nError analysis tips:")
    print("- Check for repetition or nonsensical output")
    print("- Verify domain-specific terms are used correctly")
    print("- Note any biases or inappropriate content")
    print("- Consider if more training data is needed")


def run_training_workflow_examples():
    """Run all training workflow examples."""
    
    print("1. Basic Training Setup")
    print("-" * 40)
    trainer = basic_training_setup()
    
    print("\n\n2. Training with Metrics")
    print("-" * 40)
    training_with_metrics()
    
    print("\n\n3. Early Stopping")
    print("-" * 40)
    early_stopping_example()
    
    print("\n\n4. Mixed Precision Training")
    print("-" * 40)
    mixed_precision_training()
    
    print("\n\n5. Error Analysis")
    print("-" * 40)
    error_analysis()


if __name__ == "__main__":
    run_training_workflow_examples()