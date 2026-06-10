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
        features[i].append(pop_index)

def preprocess_for_bert(text: str) -> str:
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'^\d+\.\s*|-\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# NOTE: Added normalize_nkjp parameter
def load_dataset(test_dataset, dataset_path, use_stylistic_features=True, basic_popularity_index=True, wiki_popularity_index=False, nkjp_popularity_index=False, normalize_nkjp=True, wiki_dict_path="wiki_popularity_dict.json", nkjp_dict_path="nkjp_popularity_dict.json", max_train_samples=7200, max_test_samples=2000):
    if test_dataset not in FOLDERS:
        raise ValueError('Invalid dataset name')
    
    # Calculate target splits (50% AI, 50% Human)
    target_test_ai = max_test_samples // 2
    target_test_human = max_test_samples - target_test_ai
    target_train_ai = max_train_samples // 2
    target_train_human = max_train_samples - target_train_ai

    test_ids = []
    
    # Tymczasowe listy dla testowych (żeby je na koniec zbalansować)
    temp_test_texts_ai = []
    temp_test_labels_ai = []
    temp_test_features_ai = []
    
    temp_test_texts_human = []
    temp_test_labels_human = []
    temp_test_features_human = []
    
    # ---------------------------------------------------------
    # 1. LOAD TEST AI DATA (Label 1) - BEZ FALLBACKU
    # ---------------------------------------------------------
    print(f"Loading AI Test Data (Target: {target_test_ai})...")
    test_files = sorted(os.listdir(os.path.join(dataset_path, test_dataset))) 
    
    for file in test_files:
        if len(temp_test_texts_ai) >= target_test_ai:
            break
            
        with open(os.path.join(dataset_path, test_dataset, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            doc_id = int(file.split('.')[0])
            texts = [sentence.strip() for sentence in data['Wygenerowany tekst'].split('\n\n')]
            
            for t in texts:
                if len(t.split(' ')) < 15:
                    continue 
                if len(temp_test_texts_ai) >= target_test_ai:
                    break 
                
                test_ids.append(doc_id)
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
                    temp_features.append(0)
                temp_test_features_ai.append(temp_features)
                temp_test_texts_ai.append(preprocess_for_bert(t))
                temp_test_labels_ai.append(1)

    test_ids = set(test_ids) # Optymalizacja do szybkiego wyszukiwania
    train_ids = []
    train_texts = []
    train_labels = []
    train_features = []

    # ---------------------------------------------------------
    # 2. LOAD TRAIN AI DATA (Label 1) - Z PRZEPLOTEM (INTERLEAVING)
    # ---------------------------------------------------------
    train_folders = [folder for folder in FOLDERS if folder != test_dataset]
    print(f"Loading AI Train Data (Target: {target_train_ai})...")
    
    # Tworzymy listę plików dla każdego folderu
    all_train_files = {folder: sorted(os.listdir(os.path.join(dataset_path, folder))) for folder in train_folders}
    
    # Przeplatamy pliki
    interleaved_files = []
    max_files = max([len(files) for files in all_train_files.values()]) if all_train_files else 0
    for i in range(max_files):
        for folder in train_folders:
            if i < len(all_train_files[folder]):
                interleaved_files.append((folder, all_train_files[folder][i]))

    # Przetwarzamy przeplataną listę aż uzyskamy dokładny target
    for folder, file in interleaved_files:
        if len(train_texts) >= target_train_ai:
            break
            
        doc_id = int(file.split('.')[0])
        if doc_id in test_ids:
            continue
            
        with open(os.path.join(dataset_path, folder, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            texts = [sentence.strip() for sentence in data['Wygenerowany tekst'].split('\n\n')]
            
            for t in texts:
                if len(t.split(' ')) < 15:
                    continue 
                if len(train_texts) >= target_train_ai:
                    break
                
                train_ids.append(doc_id)
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
                    temp_features.append(0)
                    
                train_features.append(temp_features)
                train_texts.append(preprocess_for_bert(t))
                train_labels.append(1)

    train_ids = set(train_ids)

    # ---------------------------------------------------------
    # 3. LOAD HUMAN DATA (Label 0)
    # ---------------------------------------------------------
    print(f"Loading Human Data (Target Train: {target_train_human}, Target Test: {target_test_human})...")
    human_train_added = 0
    used_human_ids = set()

    with open(os.path.join(dataset_path, 'scraped_news.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = sorted(data, key=lambda x: x.get('id', 0)) 
        
        # --- ETAP 1: Próba ścisłego dopasowania po ID ---
        for row in data:
            if len(temp_test_texts_human) >= target_test_human and human_train_added >= target_train_human:
                break 
                
            texts = [sentence.strip() for sentence in row['body'].split('\n\n')]
            for t in texts:
                if len(t.split(' ')) < 15:
                    continue 
                
                is_test_target = row['id'] in test_ids and len(temp_test_texts_human) < target_test_human
                is_train_target = row['id'] in train_ids and human_train_added < target_train_human
                
                if not is_test_target and not is_train_target:
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
                    temp_features.append(0)
                
                used_human_ids.add(row['id'])

                if is_test_target:
                    temp_test_features_human.append(temp_features)
                    temp_test_texts_human.append(preprocess_for_bert(t))
                    temp_test_labels_human.append(0)
                elif is_train_target:
                    train_features.append(temp_features)
                    train_texts.append(preprocess_for_bert(t))
                    train_labels.append(0)
                    human_train_added += 1

        # --- ETAP 2: "Dobieranie" TYLKO dla danych treningowych ---
        if human_train_added < target_train_human:
            print(f"Fallback: Missing exact ID matches. Filling remaining gaps for TRAIN ONLY (Need: {target_train_human - human_train_added})...")
            
            for row in data:
                if human_train_added >= target_train_human:
                    break 
                
                if row['id'] in used_human_ids:
                    continue # Pomijamy już użyte artykuły
                    
                texts = [sentence.strip() for sentence in row['body'].split('\n\n')]
                for t in texts:
                    if len(t.split(' ')) < 15:
                        continue 
                    
                    if human_train_added >= target_train_human:
                        break

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
                        temp_features.append(0)

                    used_human_ids.add(row['id'])

                    train_features.append(temp_features)
                    train_texts.append(preprocess_for_bert(t))
                    train_labels.append(0)
                    human_train_added += 1

    # ==========================================
    # --- BALANSOWANIE ZBIORU TESTOWEGO ---
    # ==========================================
    actual_test_ai = len(temp_test_texts_ai)
    actual_test_human = len(temp_test_texts_human)
    balanced_test_size = min(actual_test_ai, actual_test_human)

    print(f"Test balancing: Found {actual_test_ai} AI and {actual_test_human} Human. Cropping to {balanced_test_size} per class for perfect 50/50 split.")

    test_texts = temp_test_texts_ai[:balanced_test_size] + temp_test_texts_human[:balanced_test_size]
    test_labels = temp_test_labels_ai[:balanced_test_size] + temp_test_labels_human[:balanced_test_size]
    test_features = temp_test_features_ai[:balanced_test_size] + temp_test_features_human[:balanced_test_size]

    print(f"Final Counts -> TRAIN: {len(train_texts)} | TEST: {len(test_texts)}")

    # ==========================================
    # --- POPULARITY INDEX INTEGRATION ---
    # ==========================================
    
    # Load spaCy if ANY of the indices are True
    if basic_popularity_index or wiki_popularity_index or nkjp_popularity_index:
        print("\nLoading Polish NLP model for popularity indices...")
        nlp = spacy.load("pl_core_news_md")

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

    # ---------------------------------------------------------
    # NEW: NKJP POPULARITY INDEX
    # ---------------------------------------------------------
    if nkjp_popularity_index:
        print("\n--- Applying NKJP Popularity Index ---")
        if not os.path.exists(nkjp_dict_path):
            raise FileNotFoundError(f"NKJP dictionary not found at '{nkjp_dict_path}'. Please run the build script first.")
            
        with open(nkjp_dict_path, "r", encoding="utf-8") as f:
            raw_nkjp_dict = json.load(f)
            
        if normalize_nkjp:
            # Normalization Logic: The word "być" becomes exactly 0.1 (1/10)
            byc_count = raw_nkjp_dict.get("być")
            
            if not byc_count or byc_count == 0:
                print("WARNING: Word 'być' not found or has 0 count. Falling back to max value normalization.")
                byc_count = max(raw_nkjp_dict.values()) if raw_nkjp_dict else 1
                normalization_factor = 1.0 # Standard 0.0 to 1.0 scale
            else:
                normalization_factor = 0.1 # Force 'być' to be 0.1
                
            # Build the normalized dictionary
            nkjp_popularity_dict = {}
            for word, count in raw_nkjp_dict.items():
                # Calculate proportion relative to "być" and multiply by 0.1
                normalized_score = (count / byc_count) * normalization_factor
                nkjp_popularity_dict[word] = normalized_score
                
            print(f"NKJP Dictionary normalized. Baseline 'być' ({byc_count} occurrences) mapped to {normalization_factor}.")
        else:
            print("NKJP Dictionary normalization disabled. Using raw occurrences.")
            nkjp_popularity_dict = raw_nkjp_dict

        print("Calculating NKJP popularity index for training data...")
        append_popularity_feature(train_texts, train_features, nkjp_popularity_dict, nlp)
        print("Calculating NKJP popularity index for testing data...")
        append_popularity_feature(test_texts, test_features, nkjp_popularity_dict, nlp)

    return test_texts, test_labels, test_features, train_texts, train_labels, train_features