"""Tokenization and Vocabulary Creation Examples."""

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from config import DATA_DIR
import os
from pathlib import Path
from typing import List, Optional


def train_custom_tokenizer():
    """Train a custom SentencePiece Unigram tokenizer."""
    print("Training Custom Tokenizer...")
    
    # Validate DATA_DIR exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory does not exist: {DATA_DIR}")
    
    # Create sample corpus
    corpus_path = DATA_DIR / "sample_corpus.txt"
    sample_texts = [
        "myocardial infarction is a serious condition",
        "The patient presented with acute myocardial symptoms",
        "Diagnosis confirmed myocardial damage",
        "Treatment for myocardial conditions includes medication",
        "Post-myocardial care is essential"
    ]
    
    # Validate sample texts
    if not sample_texts:
        raise ValueError("Sample texts cannot be empty")
    
    # Write sample corpus with validation
    try:
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for text in sample_texts:
                if not isinstance(text, str):
                    raise TypeError(f"Expected string, got {type(text)}")
                f.write(text + '\n')
    except IOError as e:
        raise IOError(f"Failed to write corpus file: {e}")
    
    # Initialize a Unigram model
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Configure trainer
    trainer = trainers.UnigramTrainer(
        vocab_size=1000,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
    )
    
    # Train tokenizer with validation
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    tokenizer.train([str(corpus_path)], trainer)
    
    # Save tokenizer with validation
    tokenizer_path = DATA_DIR / "custom-tokenizer.json"
    
    try:
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved to: {tokenizer_path}")
    except Exception as e:
        raise IOError(f"Failed to save tokenizer: {e}")
    
    # Load into Hugging Face
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<s>",
        sep_token="</s>",
        mask_token="<mask>"
    )
    
    # Test tokenization
    test_text = "myocardial infarction treatment"
    tokens = hf_tokenizer.tokenize(test_text)
    print(f"\nTest text: {test_text}")
    print(f"Tokens: {tokens}")
    
    return hf_tokenizer


def compare_tokenizers():
    """Compare pre-trained vs custom tokenizer."""
    print("Comparing Tokenizers...")
    
    # Validate DATA_DIR exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory does not exist: {DATA_DIR}")
    
    # Load pre-trained tokenizer
    pretrained = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load existing custom tokenizer if available, otherwise train new one
    tokenizer_path = DATA_DIR / "custom-tokenizer.json"
    if tokenizer_path.exists() and tokenizer_path.is_file():
        print("Loading existing custom tokenizer...")
        try:
            custom = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_path),
                unk_token="<unk>",
                pad_token="<pad>",
                cls_token="<s>",
                sep_token="</s>",
                mask_token="<mask>"
            )
        except Exception as e:
            print(f"Failed to load custom tokenizer: {e}")
            print("Training new tokenizer...")
            custom = train_custom_tokenizer()
    else:
        custom = train_custom_tokenizer()
    
    # Medical terms to test
    test_sentences = [
        "myocardial infarction",
        "cardiac treatment",
        "patient diagnosis"
    ]
    
    print("\n\nTokenizer Comparison:")
    print("-" * 60)
    
    for sentence in test_sentences:
        print(f"\nText: {sentence}")
        print(f"BERT tokens: {pretrained.tokenize(sentence)}")
        print(f"Custom tokens: {custom.tokenize(sentence)}")


def vocabulary_analysis():
    """Analyze vocabulary coverage."""
    print("Vocabulary Analysis...")
    
    # Create domain-specific corpus
    medical_terms = [
        "myocardial", "infarction", "electrocardiogram",
        "thrombolytic", "angioplasty", "stenosis",
        "arrhythmia", "bradycardia", "tachycardia"
    ]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    print("\nDomain term tokenization:")
    unknown_count = 0
    
    for term in medical_terms:
        tokens = tokenizer.tokenize(term)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Check for unknown tokens
        unk_id = tokenizer.unk_token_id
        has_unk = unk_id in token_ids
        
        if has_unk:
            unknown_count += 1
            
        print(f"{term:20} -> {tokens}")
        if has_unk:
            print(f"{'':20}    Contains [UNK] token!")
    
    print(f"\nVocabulary coverage: {len(medical_terms) - unknown_count}/{len(medical_terms)} terms")
    print("Recommendation: Train custom tokenizer for better domain coverage")


def run_tokenization_examples():
    """Run all tokenization examples."""
    
    print("1. Training Custom Tokenizer")
    print("-" * 40)
    train_custom_tokenizer()
    
    print("\n\n2. Comparing Tokenizers")
    print("-" * 40)
    compare_tokenizers()
    
    print("\n\n3. Vocabulary Analysis")
    print("-" * 40)
    vocabulary_analysis()


if __name__ == "__main__":
    run_tokenization_examples()