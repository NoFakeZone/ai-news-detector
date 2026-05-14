import spacy
import json
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm # For displaying the progress bar

def build_wiki_popularity_dict(num_articles=10000, output_file="wiki_popularity_dict.json"):
    print(f"--- Downloading {num_articles} articles from Polish Wikipedia ---")
    # Download the Polish Wikipedia dataset from HuggingFace
    dataset = load_dataset("wikimedia/wikipedia", "20231101.pl", split=f"train[:{num_articles}]")
    texts = dataset['text']
    
    print("\n--- Loading Polish NLP model ---")
    nlp = spacy.load("pl_core_news_md")
    
    word_counts = Counter()
    
    print("\n--- Processing texts (Lemmatization & Filtering) ---")
    # nlp.pipe is optimized for large datasets. 
    # We disable the parser and ner to significantly speed up processing.
    for doc in tqdm(nlp.pipe(texts, disable=["parser", "ner", "textcat"]), total=len(texts)):
        for token in doc:
            # token.is_alpha ensures we only take real words (no numbers or strange wiki symbols)
            if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha:
                word_counts[token.lemma_.lower()] += 1
                
    if not word_counts:
        print("Error: No words processed.")
        return
        
    print("\n--- Preparing raw counts dictionary ---")
    # Dropped the normalization block.
    # Convert the Counter directly to a standard dictionary to save raw integer counts.
    popularity_dict = dict(word_counts)
    
    print(f"--- Saving dictionary with {len(popularity_dict)} unique words ---")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(popularity_dict, f, ensure_ascii=False, indent=4)
        
    print(f"Done! Dictionary saved to: {output_file}")

if __name__ == "__main__":
    build_wiki_popularity_dict(num_articles=50000, output_file="wiki_popularity_dict_unnormalized.json")