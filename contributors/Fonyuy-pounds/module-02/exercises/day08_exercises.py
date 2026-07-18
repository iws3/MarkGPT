import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

def load_data(filepath):
    """Load the KJV Bible CSV file."""
    return pd.read_csv(filepath)

def clean_text(text):
    """Clean text by removing punctuation and lowercasing."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_word_frequencies(df):
    """Compute word frequency distribution from the 'text' column."""
    all_text = " ".join(df['text'].apply(clean_text))
    words = all_text.split()
    return Counter(words)

def plot_top_words(word_counts, top_n=20):
    """Plot the top N most frequent words."""
    top_words = word_counts.most_common(top_n)
    words, counts = zip(*top_words)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(words), palette='viridis')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.savefig('top_words.png')
    print("Saved top_words.png")
    plt.close()

def plot_zipf_law(word_counts):
    """Plot word frequency vs rank to visualize Zipf's Law."""
    sorted_counts = sorted(word_counts.values(), reverse=True)
    ranks = range(1, len(sorted_counts) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, sorted_counts, marker='.')
    plt.title("Zipf's Law Visualization (Log-Log Plot)")
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('zipf_law.png')
    print("Saved zipf_law.png")
    plt.close()

def main():
    print("Welcome to Day 8: Data Manipulation with Pandas")
    filepath = "../../../data/raw/kjv_sample.csv"
    
    # 1. Load Data
    df = load_data(filepath)
    print("First few rows:")
    print(df.head())
    
    # 2. Basic Analysis
    print("\nVerses per book:")
    print(df['book'].value_counts())
    
    # 3. Text Cleaning & Frequencies
    word_counts = get_word_frequencies(df)
    print(f"\nTotal unique words: {len(word_counts)}")
    print("\nTop 10 words:")
    for word, count in word_counts.most_common(10):
        print(f"{word}: {count}")
    
    # 4. Visualization
    plot_top_words(word_counts)
    plot_zipf_law(word_counts)

if __name__ == "__main__":
    main()
