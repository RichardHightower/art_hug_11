"""Model Configuration and Initialization Examples."""

from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
    AutoConfig, AutoModelForCausalLM, AutoTokenizer
)
import torch
from config import get_device, MODELS_DIR

# Handle optional dependencies
try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


def configure_gpt2_from_scratch():
    """Configure a GPT-2 model from scratch."""
    print("Configuring GPT-2 from Scratch...")
    
    # Use modern config parameter names
    config = GPT2Config(
        vocab_size=30000,                # Match your tokenizer's vocab size
        max_position_embeddings=512,     # Max sequence length
        n_embd=768,                      # Embedding size
        n_layer=12,                      # Number of transformer layers
        n_head=12,                       # Number of attention heads
        use_cache=True                   # Enable caching for faster generation
    )
    
    model = GPT2LMHeadModel(config)
    
    # Sanity check: vocab size should match embedding matrix
    assert config.vocab_size == model.transformer.wte.weight.shape[0], "Vocab size mismatch!"
    
    print(f"Model initialized with:")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Max position embeddings: {config.max_position_embeddings}")
    print(f"  - Hidden size: {config.n_embd}")
    print(f"  - Layers: {config.n_layer}")
    print(f"  - Attention heads: {config.n_head}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config


def load_and_adapt_pretrained():
    """Load and adapt a pre-trained GPT-2 model."""
    print("Loading and Adapting Pre-trained GPT-2...")
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    print(f"Original vocab size: {len(tokenizer)}")
    
    # Add new domain-specific tokens
    new_tokens = ["<medical>", "<diagnosis>", "<treatment>", "<patient>"]
    num_added = tokenizer.add_tokens(new_tokens)
    
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {num_added} new tokens")
        print(f"New vocab size: {len(tokenizer)}")
    
    # Test the new tokens
    test_text = "<patient> presented with <diagnosis> requiring <treatment>"
    tokens = tokenizer.tokenize(test_text)
    print(f"\nTest text: {test_text}")
    print(f"Tokens: {tokens}")
    
    return model, tokenizer


def configure_modern_llm():
    """Configure a modern LLM (like Llama-2) for fine-tuning."""
    print("Configuring Modern LLM...")
    
    # For demonstration, we'll use a smaller model
    model_name = "microsoft/phi-2"
    
    print(f"Loading configuration for {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    
    print("\nModel Architecture:")
    print(f"  - Model type: {config.model_type}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Num attention heads: {config.num_attention_heads}")
    print(f"  - Vocab size: {config.vocab_size}")
    
    # In practice, load with pre-trained weights
    print("\nFor fine-tuning, load with pre-trained weights:")
    print(f'model = AutoModelForCausalLM.from_pretrained("{model_name}")')
    
    return config


def parameter_efficient_finetuning():
    """Demonstrate parameter-efficient fine-tuning with LoRA."""
    print("Parameter-Efficient Fine-Tuning with LoRA...")
    
    if not HAS_PEFT:
        print("Skipping: PEFT library not available")
        print("Note: bitsandbytes requires scipy which may have installation issues")
        return
    
    # Use a small model for demonstration
    model_name = "gpt2"
    
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nOriginal model:")
    print(f"  - Total parameters: {original_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]  # GPT-2 attention layers
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Count LoRA parameters
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nWith LoRA:")
    print(f"  - Trainable parameters: {lora_params:,}")
    print(f"  - Reduction: {(1 - lora_params/original_params)*100:.2f}%")
    
    # Show trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def inspect_model_architecture():
    """Inspect model parameters and architecture."""
    print("Inspecting Model Architecture...")
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    print("Model Architecture:")
    print(model)
    
    print("\n\nEmbedding Shapes:")
    print(f"Token embeddings: {model.transformer.wte.weight.shape}")
    print(f"Position embeddings: {model.transformer.wpe.weight.shape}")
    
    # Verify alignment
    config = model.config
    assert model.transformer.wte.weight.shape == (config.vocab_size, config.n_embd), \
        "Token embedding shape mismatch!"
    assert model.transformer.wpe.weight.shape == (config.n_positions, config.n_embd), \
        "Position embedding shape mismatch!"
    
    print("\nShape verification passed!")


def run_model_configuration_examples():
    """Run all model configuration examples."""
    
    print("1. Configure GPT-2 from Scratch")
    print("-" * 40)
    configure_gpt2_from_scratch()
    
    print("\n\n2. Load and Adapt Pre-trained Model")
    print("-" * 40)
    load_and_adapt_pretrained()
    
    print("\n\n3. Configure Modern LLM")
    print("-" * 40)
    configure_modern_llm()
    
    print("\n\n4. Parameter-Efficient Fine-Tuning")
    print("-" * 40)
    parameter_efficient_finetuning()
    
    print("\n\n5. Inspect Model Architecture")
    print("-" * 40)
    inspect_model_architecture()


if __name__ == "__main__":
    run_model_configuration_examples()