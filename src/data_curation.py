"""Data Curation and Cleaning Examples."""

import re
import unicodedata
from config import DATA_DIR

# Handle import issues gracefully
try:
    from datasets import load_dataset, get_dataset_config_names, Dataset
    HAS_DATASETS = True
except (ImportError, AttributeError) as e:
    print(f"Warning: datasets library import issue: {e}")
    HAS_DATASETS = False
    
try:
    from langdetect import detect
    HAS_LANGDETECT = True
except ImportError:
    print("Warning: langdetect not available")
    HAS_LANGDETECT = False


def basic_data_cleaning():
    """Basic data cleaning with Hugging Face Datasets."""
    print("Loading customer logs dataset...")
    
    if not HAS_DATASETS:
        print("Skipping: datasets library not available")
        return
    
    # Create sample data for demonstration
    sample_data = {
        "text": [
            "<p>Customer complaint: Product <b>broken</b></p>   Multiple   spaces!",
            "<div>Great service!</div>\n\n\nExtra newlines",
            "Normal text without HTML"
        ]
    }
    
    dataset = Dataset.from_dict(sample_data)
    
    def clean_text(example):
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', example["text"])
        # Replace multiple spaces/newlines with a single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return {"text": text}
    
    cleaned_dataset = dataset.map(clean_text)
    
    print("\nOriginal vs Cleaned:")
    for i in range(len(dataset)):
        print(f"Original: {dataset[i]['text']}")
        print(f"Cleaned:  {cleaned_dataset[i]['text']}\n")


def scalable_text_cleaning():
    """Scalable text cleaning and deduplication."""
    print("Scalable text cleaning example...")
    
    if not HAS_DATASETS:
        print("Skipping: datasets library not available")
        return
    
    # Create sample data that simulates Wikipedia-like content
    sample_wiki_data = {
        "text": [
            "Python is a <b>high-level</b> programming language. Visit https://python.org for more info.",
            "Machine   learning   involves    multiple    spaces and <i>algorithms</i>.",
            "Natural language processing (NLP) is a field of AI. See https://example.com/nlp",
        ]
    }
    
    try:
        dataset = Dataset.from_dict(sample_wiki_data)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    def clean_text(example):
        text = unicodedata.normalize('NFKC', example['text'])  # Unicode normalization
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        return {"text": text}
    
    # Apply cleaning to dataset
    cleaned_dataset = dataset.map(clean_text)
    
    print("\nCleaning examples:")
    for i in range(len(dataset)):
        print(f"\nExample {i+1}:")
        print(f"Original: {dataset[i]['text']}")
        print(f"Cleaned:  {cleaned_dataset[i]['text']}")


def language_detection_filtering():
    """Automated language detection and filtering."""
    print("Demonstrating language detection...")
    
    if not HAS_DATASETS:
        print("Skipping: datasets library not available")
        return
        
    if not HAS_LANGDETECT:
        print("Skipping: langdetect not available")
        return
    
    # Sample multilingual data
    sample_texts = [
        "This is an English sentence.",
        "Ceci est une phrase en français.",
        "Dies ist ein deutscher Satz.",
        "This is another English example.",
        "これは日本語の文です。"
    ]
    
    dataset = Dataset.from_dict({"text": sample_texts})
    
    def filter_english(example):
        try:
            return detect(example['text']) == 'en'
        except (ImportError, LookupError, ValueError) as e:
            # LookupError: langdetect couldn't detect language
            # ValueError: invalid input text
            print(f"Language detection error: {e}")
            return False
    
    # Filter for English only
    english_dataset = dataset.filter(filter_english)
    
    print("\nOriginal texts:")
    for text in dataset['text']:
        try:
            lang = detect(text)
        except (ImportError, LookupError, ValueError):
            lang = "unknown"
        print(f"  [{lang}] {text}")
    
    print("\nFiltered (English only):")
    for text in english_dataset['text']:
        print(f"  {text}")


def pii_redaction():
    """Simple PII redaction example."""
    print("PII Redaction Example...")
    
    def redact_pii(text):
        # Apply patterns in specific order to avoid conflicts
        
        # SSN pattern (xxx-xx-xxxx) - do this before phone numbers
        text = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[SSN]',
            text
        )
        
        # Credit card patterns (basic - 16 digits with optional spaces/dashes)
        text = re.sub(
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            '[CREDIT_CARD]',
            text
        )
        
        # Improved email pattern - handles more complex email formats
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            text
        )
        
        # Improved phone patterns - handles various formats
        # US phone numbers: (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx, etc.
        text = re.sub(
            r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            '[PHONE]',
            text
        )
        
        # International phone formats (more specific to avoid catching SSN/CC)
        text = re.sub(
            r'\+[0-9]{1,3}[-.\s]?\(?[0-9]{1,4}\)?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}',
            '[PHONE]',
            text
        )
        
        # Improved name patterns - handles more titles and name formats
        # Titles with first and last names
        text = re.sub(
            r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Rev\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            '[NAME]',
            text
        )
        
        # Single title with name
        text = re.sub(
            r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Rev\.)\s+[A-Z][a-z]+\b',
            '[NAME]',
            text
        )
        
        return text
    
    samples = [
        "Contact Dr. Smith at dr.smith@example.com or 555-123-4567.",
        "Please email john.doe@company.com for support.",
        "Ms. Johnson can be reached at 123-456-7890.",
        "Prof. John Williams called from +1 (800) 555-1234.",
        "SSN: 123-45-6789 and card: 4532-1234-5678-9012"
    ]
    
    print("\nBefore and after PII redaction:")
    for sample in samples:
        print(f"Original: {sample}")
        print(f"Redacted: {redact_pii(sample)}\n")


def streaming_batch_processing():
    """Streaming and batch processing example."""
    print("Streaming and Batch Processing Example...")
    
    if not HAS_DATASETS:
        print("Skipping: datasets library not available")
        return
    
    # Create a larger sample dataset
    large_sample = {
        "text": [
            "This is a long text about machine learning. " * 50,  # Repeated text
            "Natural language processing involves many techniques. " * 50,
            "Deep learning has revolutionized AI. " * 50,
            "Transformers are powerful models. " * 50,
            "Data science requires many skills. " * 50,
        ]
    }
    
    dataset = Dataset.from_dict(large_sample)
    
    def process_batch(batch):
        # Example batch processing (e.g., truncating text)
        return {"processed_text": [t[:200] for t in batch["text"]]}
    
    # Process data in batches
    processed = dataset.map(process_batch, batched=True, batch_size=2)
    
    print("\nProcessing examples (truncated to 200 chars):")
    for i in range(min(3, len(processed))):
        print(f"\nExample {i+1}:")
        print(f"Original length: {len(dataset[i]['text'])} chars")
        print(f"Processed: {processed[i]['processed_text']}...")


def run_data_curation_examples():
    """Run all data curation examples."""
    
    print("1. Basic Data Cleaning")
    print("-" * 40)
    basic_data_cleaning()
    
    print("\n\n2. Scalable Text Cleaning")
    print("-" * 40)
    scalable_text_cleaning()
    
    print("\n\n3. Language Detection and Filtering")
    print("-" * 40)
    language_detection_filtering()
    
    print("\n\n4. PII Redaction")
    print("-" * 40)
    pii_redaction()
    
    print("\n\n5. Streaming and Batch Processing")
    print("-" * 40)
    streaming_batch_processing()


if __name__ == "__main__":
    run_data_curation_examples()