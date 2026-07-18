import re
import math
from collections import Counter, defaultdict

class NaiveBayesClassifier:
    """A Naive Bayes Classifier implemented from scratch for text classification."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_priors = {}
        self.word_counts = defaultdict(Counter)
        self.vocab = set()
        self.class_totals = defaultdict(int)

    def tokenize(self, text):
        """Simple tokenizer that lowercases and extracts alphanumeric words."""
        return re.findall(r'\w+', text.lower())

    def train(self, data, labels):
        """Train the classifier on a list of texts and their corresponding labels."""
        n_samples = len(data)
        label_counts = Counter(labels)
        
        # Calculate P(Class)
        for label, count in label_counts.items():
            self.class_priors[label] = count / n_samples
        
        # Calculate word frequencies per class
        for text, label in zip(data, labels):
            words = self.tokenize(text)
            self.word_counts[label].update(words)
            self.class_totals[label] += len(words)
            self.vocab.update(words)

    def predict_log_prob(self, text):
        """Calculate the log-probability of each class for a given text."""
        words = self.tokenize(text)
        log_probs = {}
        
        vocab_size = len(self.vocab)
        
        for label in self.class_priors:
            # Start with log(P(Class))
            prob = math.log(self.class_priors[label])
            
            # Add log(P(Word | Class)) for each word in the text
            for word in words:
                # Use Laplace smoothing: (count + alpha) / (total_words_in_class + alpha * vocab_size)
                count = self.word_counts[label].get(word, 0)
                word_prob = (count + self.alpha) / (self.class_totals[label] + self.alpha * vocab_size)
                prob += math.log(word_prob)
            
            log_probs[label] = prob
            
        return log_probs

    def predict(self, text):
        """Predict the class with the highest log-probability."""
        log_probs = self.predict_log_prob(text)
        return max(log_probs, key=log_probs.get)

def main():
    print("=== Day 11: Naive Bayes Text Classification from Scratch ===")
    
    # 1. Prepare Data
    # Biblical text (from KJV Sample)
    kjv_texts = [
        "In the beginning God created the heaven and the earth.",
        "And the earth was without form and void; and darkness was upon the face of the deep.",
        "And God said Let there be light: and there was light.",
        "And God saw the light that it was good: and God divided the light from the darkness.",
        "And God called the light Day and the darkness he called Night.",
        "Now these are the names of the children of Israel which came into Egypt.",
        "The LORD is my shepherd; I shall not want.",
        "He maketh me to lie down in green pastures: he leadeth me beside the still waters.",
        "In the beginning was the Word and the Word was with God and the Word was God.",
        "For God so loved the world that he gave his only begotten Son."
    ]

    # Non-biblical text (generic modern/technical sentences)
    non_biblical_texts = [
        "The new artificial intelligence model achieved state of the art results on the benchmark.",
        "Python is a versatile programming language used for data science and web development.",
        "The stock market experienced a significant dip after the latest economic report.",
        "Scientists discovered a new species of deep sea fish in the Pacific Ocean.",
        "The recipe calls for three cups of flour and two teaspoons of baking powder.",
        "The latest smartphone features a high resolution camera and a powerful processor.",
        "Climate change is a global challenge that requires international cooperation.",
        "The football team won the championship after a thrilling overtime victory.",
        "Cybersecurity experts warn about the increasing threat of ransomware attacks.",
        "The symphony orchestra performed a stunning rendition of Beethoven's Ninth."
    ]

    train_data = kjv_texts + non_biblical_texts
    train_labels = ["KJV"] * len(kjv_texts) + ["Modern"] * len(non_biblical_texts)

    # 2. Train Model
    print(f"Training on {len(train_data)} samples...")
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.train(train_data, train_labels)
    print(f"Vocabulary size: {len(clf.vocab)} words.")

    # 3. Evaluate Model
    test_cases = [
        ("The spirit of the Lord moved upon the waters", "KJV"),
        ("Machine learning algorithms require large amounts of data", "Modern"),
        ("And they came into the land of Egypt with their families", "KJV"),
        ("The software engineer updated the database schema", "Modern"),
        ("He leadeth me by the quiet streams of righteousness", "KJV"),
        ("The central bank raised interest rates to combat inflation", "Modern")
    ]

    print("\n--- Test Results ---")
    correct = 0
    for text, actual in test_cases:
        prediction = clf.predict(text)
        is_correct = (prediction == actual)
        if is_correct:
            correct += 1
        print(f"Text: '{text}'")
        print(f"Actual: {actual}, Predicted: {prediction} | {'[CORRECT]' if is_correct else '[INCORRECT]'}\n")

    accuracy = (correct / len(test_cases)) * 100
    print(f"Final Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
