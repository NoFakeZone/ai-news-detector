import spacy
import re
import json
import time
import os
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

# --- CONFIGURATION ---
DATA_PATH = r'C:\Users\PC\OneDrive\Pulpit\projekty\ai-news-generator'
DICT_OUTPUT_PATH = "nkjp_popularity_dict.json"

FOLDERS = [
    'gemini-2.5-flash',
    'gemini-3.1-flash-lite-preview',
    'gemini-3-flash-preview',
    'gpt-oss-120b',
    'llama-3.3-70b-instruct-fp8-fast',
    'nemotron-3-120b-a12b'
]

# --- 1. LOAD ALL DATA ---
def load_all_texts(dataset_path: str) -> list:
    print("Loading 100% of data from disk (no limits)...")
    all_texts = []
    
    # Load AI-generated texts
    for folder in FOLDERS:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"Skipped missing folder: {folder}")
            continue
            
        files = os.listdir(folder_path)
        for file in files:
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                texts = [sentence.strip() for sentence in data.get('Wygenerowany tekst', '').split('\n\n')]
                for text in texts:
                    if len(text.split(' ')) >= 15:
                        all_texts.append(text)
                        
    # Load human texts (scraped_news.json)
    human_path = os.path.join(dataset_path, 'scraped_news.json')
    if os.path.exists(human_path):
        with open(human_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for row in data:
                texts = [sentence.strip() for sentence in row.get('body', '').split('\n\n')]
                for text in texts:
                    if len(text.split(' ')) >= 15:
                        all_texts.append(text)
    
    print(f"Successfully loaded {len(all_texts)} paragraphs from the entire database!")
    return all_texts

# --- 2. LEMMATIZATION ---
def extract_unique_lemmas(texts: list, nlp) -> list:
    print("Starting lemmatization and extraction of unique words...")
    unique_words = set()
    
    # Disable parser and ner to speed up spaCy processing
    for doc in nlp.pipe(texts, disable=["parser", "ner"]):
        for token in doc:
            if token.is_alpha and not token.is_stop and not token.is_punct and not token.is_space:
                unique_words.add(token.lemma_.lower())
                
    print(f"Found {len(unique_words)} unique base words to check in NKJP.")
    return list(unique_words)

# --- 3. MAIN LOGIC ---
def build_dictionary():
    # Fetch absolutely all texts
    all_texts = load_all_texts(DATA_PATH)
    
    if not all_texts:
        print("No texts found! Check the DATA_PATH.")
        return

    print("\nLoading Polish language model (spaCy)...")
    nlp = spacy.load("pl_core_news_md")
    words_to_check = extract_unique_lemmas(all_texts, nlp)
    
    # Resume from file (if the script was interrupted)
    popularity_dict = {}
    if os.path.exists(DICT_OUTPUT_PATH):
        with open(DICT_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            popularity_dict = json.load(f)
        print(f"Resumed from file. The dictionary already contains {len(popularity_dict)} words.")
        
        # Filter out words that have already been checked
        words_to_check = [word for word in words_to_check if word not in popularity_dict]
        print(f"Remaining words to check: {len(words_to_check)}")

    if not words_to_check:
        print("All words have already been checked! The file is ready.")
        return

    print("\nStarting Selenium browser in the background...")
    options = webdriver.ChromeOptions()
    options.add_argument('--headless') 
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--log-level=3')
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        print("Starting NKJP database queries...")
        
        for i, word in enumerate(tqdm(words_to_check)):
            # Parameter perpage=1 speeds up the server response
            url = f"https://pelcra-nkjp.clarin-pl.eu/index_adv.jsp?query={word}**&Submit=SZUKAJ&span=0&perpage=1&m_nkjpSubcorpus=balanced&dummystring=ąĄćĆęĘłŁńŃóÓśŚźŹżŻ"
            
            driver.get(url)
            time.sleep(2.5) 
            
            # Handle frames if the site uses them to display results
            frames = driver.find_elements(By.TAG_NAME, "frame")
            if not frames:
                frames = driver.find_elements(By.TAG_NAME, "iframe")
                
            if len(frames) > 0:
                driver.switch_to.frame(frames[-1])
                time.sleep(0.5)

            html = driver.page_source
            driver.switch_to.default_content() 
            
            # REGEX extracting the number from: "Znaleziono <b>12,345</b> akapitów"
            pattern = r"Znaleziono\s*<b>([\d,]+)</b>\s*akapitów"
            match = re.search(pattern, html)
            
            if match:
                count_str = match.group(1).replace(',', '') 
                popularity_dict[word] = int(count_str)
            else:
                popularity_dict[word] = 0

            # Backup save every 50 words
            if (i + 1) % 50 == 0:
                with open(DICT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(popularity_dict, f, ensure_ascii=False, indent=4)
                    
    except Exception as e:
        print(f"\nScript interrupted due to error: {e}")
    finally:
        print("\nSaving current progress and closing the browser...")
        with open(DICT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(popularity_dict, f, ensure_ascii=False, indent=4)
        driver.quit()
        print("Done!")

if __name__ == "__main__":
    build_dictionary()