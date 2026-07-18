import random
from day04_preprocess import load_and_preprocess
from day04_model import BigramLanguageModel

def main():
    sample_genesis = """
    In the beginning God created the heaven and the earth.
    And the earth was without form and void and darkness was upon the face of the deep.
    And the Spirit of God moved upon the face of the waters.
    And God said Let there be light and there was light.
    And God saw the light that it was good and God divided the light from the darkness.
    And God called the light Day and the darkness he called Night.
    And the evening and the morning were the first day.
    And God said Let there be a firmament in the midst of the waters.
    And God made the firmament and divided the waters which were under the firmament.
    And God called the firmament Heaven and the evening and the morning were the second day.
    """
    
    # Preprocess
    print("Preprocessing text...")
    tokens = load_and_preprocess(sample_genesis)
    
    # Split into train/test
    split = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    test_tokens = tokens[split:]
    
    # Train
    print("Training models...")
    model_no_smooth = BigramLanguageModel(smoothing=0.0)
    model_no_smooth.train(train_tokens)
    
    model_smooth = BigramLanguageModel(smoothing=1.0)
    model_smooth.train(train_tokens)
    
    # Generate
    random.seed(42)
    print("\nGENERATED (no smoothing, seed='god'):")
    print(f"  {model_no_smooth.generate('god', max_tokens=20)}")
    
    print("\nGENERATED (smoothing, seed='god'):")
    print(f"  {model_smooth.generate('god', max_tokens=20)}")
    
    # Evaluate
    ppl_no_smooth = model_no_smooth.perplexity(test_tokens)
    ppl_smooth = model_smooth.perplexity(test_tokens)
    
    print(f"\nPERPLEXITY (No Smooth): {ppl_no_smooth:.1f}")
    print(f"PERPLEXITY (Smooth):    {ppl_smooth:.1f}")

if __name__ == "__main__":
    main()
