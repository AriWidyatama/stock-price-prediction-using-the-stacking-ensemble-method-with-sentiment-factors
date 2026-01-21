import re
import gc
import pickle
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import numpy as np
import pandas as pd
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from indoNLP.preprocessing import replace_slang

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

with open("data/stopWords.txt", "r", encoding="utf-8") as f:
    stopWords = set(f.read().splitlines())

@lru_cache(maxsize=5000)
def cached_stem(word: str) -> str:
    try:
        return stemmer.stem(word)

    except KeyError as e:
        return word

def preprocess_df(series):
    series = series.str.lower()

    series = series.str.replace(r'[^\x00-\x7F]+', '', regex=True)

    series = series.str.replace(r"@\w+|#\w+|http\S+|www\.\S+", "", regex=True)
    series = series.str.replace(r"\S+@\S+\.\S+", "", regex=True)

    series = series.str.replace(r'["#$%&()\*,\./:;@\[\\\]^_`{|}~\']', " ", regex=True)

    series = series.str.replace(r"\d+", "<num>", regex=True)
    series = series.str.replace(r"(<num>\s*)+", "<num> ", regex=True)

    return series

def preprocess_final(text):
    try:
        text = replace_slang(text)

    except KeyError as e:
        pass

    tokens = re.findall(r'\b\w+\b', text)

    tokens = [t for t in tokens if t not in stopWords]

    tokens = [cached_stem(t) for t in tokens]

    text = " ".join(tokens)

    return text

def parallel_processing(df, text_column="clean_text"):
    texts = df[text_column].tolist()
    total_size = len(texts)

    #OPTIMAL_WORKERS = cpu_count() if cpu_count() else 4
    OPTIMAL_WORKERS = min(cpu_count(), 8)

    BATCH_SIZE = 20000
    CHUNKSIZE = 1000

    all_results = []
    print(f"Memproses {total_size} data dengan {OPTIMAL_WORKERS} pekerja.")

    with Pool(processes=OPTIMAL_WORKERS) as pool:

        for start_idx in tqdm(range(0, total_size, BATCH_SIZE), desc="🔄 Main Progress"):
            end_idx = min(start_idx + BATCH_SIZE, total_size)
            batch_texts = texts[start_idx:end_idx]

            batch_results = list(tqdm(
                pool.imap(preprocess_final, batch_texts, chunksize=CHUNKSIZE),
                total=len(batch_texts),
                desc=f"Batch {start_idx//BATCH_SIZE + 1}",
                leave=False
            ))

            with open(f"data/backup/batch_{start_idx}.pkl", "wb") as f:
              pickle.dump(batch_results, f)

            all_results.extend(batch_results)

            del batch_texts, batch_results
            if start_idx % 40000 == 0:
                gc.collect()

    df["text_final"] = all_results
    return df

def predict_sentimen(df, sentimen_model, vectorizer, ticker):
    BATCH_SIZE = 5000
    texts = df["text_final"].tolist()
    prediksi = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_text = texts[i:i+BATCH_SIZE]

        batch_vect = vectorizer(batch_text)

        batch_pred = sentimen_model.predict(batch_vect, verbose=0)

        batch_label = np.argmax(batch_pred, axis=1)

        prediksi.extend(batch_label)

        with open(f"data/{ticker}/sentimen_predict_{ticker}.pkl", "wb") as f:
            pickle.dump(prediksi, f)

    return prediksi