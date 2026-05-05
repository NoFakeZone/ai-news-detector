import spacy
import re
import json
import os
from collections import Counter

from ai_news_detector.features.pos import pos_per_word, UD_TAGS, all_pos_per_word
from ai_news_detector.features.punctuation import punctuation_per_letter, punctuation_per_word
from ai_news_detector.features.text_stats import avg_sentence_len, capital_ratio, ttr, ttr_lemmatized
from ai_news_detector.features.syllables import avg_syllables_per_sentence, avg_word_length

FOLDERS = [
    'gemini-2.5-flash',
    'gemini-3.1-flash-lite-preview',
    'gemini-3-flash-preview',
    'gpt-oss-120b',
    'llama-3.3-70b-instruct-fp8-fast',
    'nemotron-3-120b-a12b'
]

def build_popularity_dictionary(train_texts: list, train_labels: list, nlp) -> dict:
    print("\n--- Building Basic Popularity Dictionary ---")
    human_texts = [text for text, label in zip(train_texts, train_labels) if label == 0]
    word_counts = Counter()
    
    # Using nlp.pipe is much faster for processing large lists of text
    for doc in nlp.pipe(human_texts, disable=["parser", "ner"]):
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha:
                word_counts[token.lemma_.lower()] += 1
                
    if not word_counts:
        return {}
        
    max_count = max(word_counts.values())
    popularity_dict = {word: count / max_count for word, count in word_counts.items()}
    
    print(f"Dictionary built with {len(popularity_dict)} unique lemmas.")
    return popularity_dict

def append_popularity_feature(texts: list, features: list, popularity_dict: dict, nlp):
    """Calculates the index for a list of texts and appends it to the features list."""
    # Process texts in bulk for speed
    docs = nlp.pipe(texts, disable=["parser", "ner"])
    
    for i, doc in enumerate(docs):
        valid_words_count = 0
        total_score = 0.0
        
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha:
                lemma = token.lemma_.lower()
                total_score += popularity_dict.get(lemma, 0.0)
                valid_words_count += 1
                
        pop_index = total_score / valid_words_count if valid_words_count > 0 else 0.0
        # Append the new feature to the existing feature list for this specific text
        features[i].append(pop_index)

def preprocess_for_bert(text: str) -> str:
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'^\d+\.\s*|-\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Dodano opcję use_stylistic_features, by łatwo wyłączać ciężkie obliczenia
def load_dataset(test_dataset, dataset_path, use_stylistic_features=True, basic_popularity_index=True, wiki_popularity_index=False, wiki_dict_path="wiki_popularity_dict.json"):
    if test_dataset not in FOLDERS:
        raise ValueError('Invalid dataset name')
    
    test_ids = []
    test_texts = []
    test_labels = []
    test_features = []
    
    n = len(os.listdir(os.path.join(dataset_path, test_dataset)))
    i = 0
    for file in os.listdir(os.path.join(dataset_path, test_dataset)):
        with open(os.path.join(dataset_path, test_dataset, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            test_ids.append(int(file.split('.')[0]))
            texts = [sentence.strip() for sentence in data['Wygenerowany tekst'].split('\n\n')]
            for t in texts:
                if len(t.split(' ')) < 15:
                    continue 
                
                temp_features = []
                # Obliczaj cechy stylistyczne tylko jeśli flaga jest True
                if use_stylistic_features:
                    pos_ratios = all_pos_per_word(t)
                    for pos in UD_TAGS:
                        temp_features.append(pos_ratios[pos])
                    temp_features.append(punctuation_per_letter(t))
                    temp_features.append(punctuation_per_word(t))
                    temp_features.append(avg_sentence_len(t))
                    temp_features.append(capital_ratio(t))
                    temp_features.append(ttr(t))
                    temp_features.append(ttr_lemmatized(t))
                    temp_features.append(avg_syllables_per_sentence(t))
                    temp_features.append(avg_word_length(t))
                else:
                    temp_features.append(None)
                test_features.append(temp_features)
                test_texts.append(preprocess_for_bert(t))
                test_labels.append(1)
        i += 1
        if i % 100 == 0:
            print(f'{i}/{n}')

    test_ids = set(test_ids)
    train_ids = []
    train_texts = []
    train_labels = []
    train_features = []

    for folder in FOLDERS:
        if folder == test_dataset:
            continue
        n = len(os.listdir(os.path.join(dataset_path, folder)))
        i = 0
        for file in os.listdir(os.path.join(dataset_path, folder)):
            with open(os.path.join(dataset_path, folder, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if int(file.split('.')[0]) in test_ids:
                    continue
                train_ids.append(int(file.split('.')[0]))
                texts = [sentence.strip() for sentence in data['Wygenerowany tekst'].split('\n\n')]
                for t in texts:
                    if len(t.split(' ')) < 15:
                        continue 
                    
                    temp_features = []
                    if use_stylistic_features:
                        pos_ratios = all_pos_per_word(t)
                        for pos in UD_TAGS:
                            temp_features.append(pos_ratios[pos])
                        temp_features.append(punctuation_per_letter(t))
                        temp_features.append(punctuation_per_word(t))
                        temp_features.append(avg_sentence_len(t))
                        temp_features.append(capital_ratio(t))
                        temp_features.append(ttr(t))
                        temp_features.append(ttr_lemmatized(t))
                        temp_features.append(avg_syllables_per_sentence(t))
                        temp_features.append(avg_word_length(t))
                    else:
                        temp_features.append(None)
                    train_features.append(temp_features)
                    train_texts.append(preprocess_for_bert(t))
                    train_labels.append(1)
            i += 1
            if i % 100 == 0:
                print(f'{i}/{n}')

    train_ids = set(train_ids)

    with open(os.path.join(dataset_path, 'scraped_news.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
        n = len(data)
        i = 0
        for row in data:
            texts = [sentence.strip() for sentence in row['body'].split('\n\n')]
            for t in texts:
                if len(t.split(' ')) < 15:
                    continue 
                
                temp_features = []
                if use_stylistic_features:
                    pos_ratios = all_pos_per_word(t)
                    for pos in UD_TAGS:
                        temp_features.append(pos_ratios[pos])
                    temp_features.append(punctuation_per_letter(t))
                    temp_features.append(punctuation_per_word(t))
                    temp_features.append(avg_sentence_len(t))
                    temp_features.append(capital_ratio(t))
                    temp_features.append(ttr(t))
                    temp_features.append(ttr_lemmatized(t))
                    temp_features.append(avg_syllables_per_sentence(t))
                    temp_features.append(avg_word_length(t))
                else:
                    temp_features.append(None)
                if row['id'] in test_ids:
                    test_features.append(temp_features)
                    test_texts.append(preprocess_for_bert(t))
                    test_labels.append(0)
                elif row['id'] in train_ids:
                    train_features.append(temp_features)
                    train_texts.append(preprocess_for_bert(t))
                    train_labels.append(0)
            i += 1
            if i % 100 == 0:
                print(f'{i}/{n}')

    # ==========================================
    # --- POPULARITY INDEX INTEGRATION ---
    # ==========================================
    if basic_popularity_index or wiki_popularity_index:
        print("\nLoading Polish NLP model for popularity indices...")
        nlp = spacy.load("pl_core_news_md") # Loads only once to save time

    if basic_popularity_index:
        print("\n--- Applying Basic Popularity Index ---")
        popularity_dict = build_popularity_dictionary(train_texts, train_labels, nlp)
        
        print("Calculating basic popularity index for training data...")
        append_popularity_feature(train_texts, train_features, popularity_dict, nlp)
        
        print("Calculating basic popularity index for testing data...")
        append_popularity_feature(test_texts, test_features, popularity_dict, nlp)

    if wiki_popularity_index:
        print("\n--- Applying Wikipedia Popularity Index ---")
        if not os.path.exists(wiki_dict_path):
            raise FileNotFoundError(f"Wikipedia dictionary not found at '{wiki_dict_path}'. Please run the build script first.")
            
        with open(wiki_dict_path, "r", encoding="utf-8") as f:
            wiki_popularity_dict = json.load(f)
            
        print("Calculating Wiki popularity index for training data...")
        append_popularity_feature(train_texts, train_features, wiki_popularity_dict, nlp)
        
        print("Calculating Wiki popularity index for testing data...")
        append_popularity_feature(test_texts, test_features, wiki_popularity_dict, nlp)

    return test_texts, test_labels, test_features, train_texts, train_labels, train_features