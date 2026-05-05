import spacy
import json
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm # Do wyświetlania paska postępu

def build_wiki_popularity_dict(num_articles=10000, output_file="wiki_popularity_dict.json"):
    print(f"--- Downloading {num_articles} articles from Polish Wikipedia ---")
    # Pobieramy paczkę polskiej wikipedii z HuggingFace
    dataset = load_dataset("wikimedia/wikipedia", "20231101.pl", split=f"train[:{num_articles}]")
    texts = dataset['text']
    
    print("\n--- Loading Polish NLP model ---")
    nlp = spacy.load("pl_core_news_md")
    
    word_counts = Counter()
    
    print("\n--- Processing texts (Lemmatization & Filtering) ---")
    # nlp.pipe jest zoptymalizowane pod duże zbiory danych. 
    # Wyłączamy parser i ner żeby znacznie przyspieszyć działanie.
    for doc in tqdm(nlp.pipe(texts, disable=["parser", "ner", "textcat"]), total=len(texts)):
        for token in doc:
            # token.is_alpha upewnia się, że bierzemy tylko prawdziwe słowa (bez liczb i dziwnych znaków z wiki)
            if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha:
                word_counts[token.lemma_.lower()] += 1
                
    if not word_counts:
        print("Error: No words processed.")
        return
        
    print("\n--- Normalizing values ---")
    max_count = max(word_counts.values())
    popularity_dict = {word: count / max_count for word, count in word_counts.items()}
    
    print(f"--- Saving dictionary with {len(popularity_dict)} unique words ---")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(popularity_dict, f, ensure_ascii=False, indent=4)
        
    print(f"Done! Dictionary saved to: {output_file}")

if __name__ == "__main__":
    build_wiki_popularity_dict(num_articles=10000) 
    # Możesz zwiększyć num_articles np. do 50000, jeśli chcesz dokładniejszy słownik, 
    # ale zajmie to więcej czasu i pamięci RAM.