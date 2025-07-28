"""Chain of Thought Examples."""

from transformers import pipeline
from config import get_device


def basic_chain_of_thought():
    """Basic chain of thought reasoning example."""
    print("Basic Chain of Thought Example...")
    
    generator = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if get_device() == "cuda" else -1
    )
    
    # Math problem with step-by-step reasoning
    prompt = """Solve step by step:

Q: If a shirt costs $20 and is on sale for 25% off, what is the final price?
A: Let me solve this step by step:
1. The shirt costs $20
2. The discount is 25% of $20 = 0.25 × $20 = $5
3. The final price = $20 - $5 = $15
Therefore, the final price is $15.

Q: A recipe needs 3 cups of flour for 12 cookies. How much flour is needed for 20 cookies?
A: Let me solve this step by step:
1. For 12 cookies, we need 3 cups of flour
2. For 1 cookie, we need 3/12 = 0.25 cups of flour
3. For 20 cookies, we need 0.25 × 20 = 5 cups of flour
Therefore, we need 5 cups of flour for 20 cookies.

Q: If a train travels 60 miles in 45 minutes, what is its speed in miles per hour?
A: Let me solve this step by step:"""

    output = generator(
        prompt,
        max_new_tokens=100,
        temperature=0.3,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    print("Chain of Thought Prompt:")
    print(prompt)
    print("\nModel's reasoning:", output[0]['generated_text'][len(prompt):])


def medical_reasoning_cot():
    """Chain of thought for medical reasoning."""
    print("Medical Reasoning with Chain of Thought...")
    
    generator = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if get_device() == "cuda" else -1
    )
    
    prompt = """Diagnose based on symptoms using step-by-step reasoning:

Patient: 45-year-old male presenting with chest pain and shortness of breath
Analysis: Let me evaluate step by step:
1. Key symptoms: chest pain + shortness of breath
2. These are cardinal symptoms of cardiac issues
3. Age (45) puts patient in risk category for heart disease
4. Most likely: Acute coronary syndrome
5. Immediate actions needed: ECG, cardiac enzymes, oxygen
Conclusion: Possible myocardial infarction, requires immediate cardiac evaluation.

Patient: 28-year-old female with severe headache, fever, and neck stiffness
Analysis: Let me evaluate step by step:"""

    output = generator(
        prompt,
        max_new_tokens=120,
        temperature=0.3,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    print("Medical Chain of Thought:")
    print(prompt)
    print("\nModel's analysis:", output[0]['generated_text'][len(prompt):])


def logical_reasoning_cot():
    """Chain of thought for logical reasoning problems."""
    print("Logical Reasoning with Chain of Thought...")
    
    generator = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if get_device() == "cuda" else -1
    )
    
    prompt = """Solve the logic puzzle step by step:

Puzzle: Three friends (Alice, Bob, Carol) have different pets (cat, dog, bird). 
- Alice doesn't have the dog
- The person with the cat lives next to Bob
- Carol doesn't have the bird

Solution: Let me work through this step by step:
1. Alice doesn't have the dog, so Alice has either cat or bird
2. Carol doesn't have the bird, so Carol has either cat or dog
3. The person with the cat lives next to Bob, so Bob doesn't have the cat
4. From step 3: Bob has either dog or bird
5. If Carol had the cat, that would work with clue 2
6. This means: Carol has cat, Bob has bird, Alice has dog
Wait, but Alice can't have the dog (clue 1)
7. So Carol must have the dog, Bob has the bird, Alice has the cat
Final answer: Alice-cat, Bob-bird, Carol-dog

Puzzle: Four people are in a line. Sarah is not first or last. Tom is somewhere after Sarah. Mike is not next to Tom. Where is everyone?

Solution: Let me work through this step by step:"""

    output = generator(
        prompt,
        max_new_tokens=150,
        temperature=0.3,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    print("Logical Reasoning Chain of Thought:")
    print(prompt)
    print("\nModel's solution:", output[0]['generated_text'][len(prompt):])


def run_chain_of_thought_examples():
    """Run all chain of thought examples."""
    
    print("Running chain of thought examples...")
    print("These show step-by-step reasoning patterns.\n")
    
    basic_chain_of_thought()
    print("\n" + "-" * 60 + "\n")
    
    medical_reasoning_cot()
    print("\n" + "-" * 60 + "\n")
    
    logical_reasoning_cot()


if __name__ == "__main__":
    run_chain_of_thought_examples()