# step5_score_v2.py

import os
import math
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import csv
from multiprocessing import Pool
import string
import logging
from nltk.stem import PorterStemmer
import nltk

# Initialize NLTK resources if needed
nltk.download('punkt')

# Initialize logging at the very beginning
logging.basicConfig(
    filename='step5_score.log',
    filemode='w',  # Use 'w' to overwrite the log file each time the script runs
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Set to DEBUG for detailed logs
)

# Import custom modules after logging configuration
import file_util
import global_options as gl
from narrativesBERT import attention_dictionary

# Initialize Porter Stemmer
ps = PorterStemmer()

def clean_word(word):
    """
    Clean a single word: remove punctuation, replace underscores, lowercase, and stem.

    Arguments:
        word (str): The word to clean.

    Returns:
        str: The cleaned word.
    """
    if pd.isna(word):
        return None  # Skip missing words
    word = str(word)
    word = word.replace('_', ' ')  # Replace underscores with spaces
    word = word.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    word = word.lower()
    word = ps.stem(word)  # Apply stemming
    return word

def clean_text(text):
    """
    Clean a text string: remove punctuation, replace underscores, lowercase, stem, and tokenize.

    Arguments:
        text (str): The text to clean.

    Returns:
        list[str]: A list of cleaned tokens.
    """
    text = text.replace('_', ' ')  # Replace underscores with spaces
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()
    tokens = text.split()
    tokens = [ps.stem(token) for token in tokens]  # Apply stemming
    return tokens

def load_word_to_categories(expanded_dict_path):
    """
    Load the expanded dictionary and create a mapping from words to categories.

    Arguments:
        expanded_dict_path (str or Path): Path to expanded_dictionary.csv

    Returns:
        tuple:
            word_to_categories (dict): Mapping from word to set of associated categories
            all_categories (list): List of all categories
    """
    expanded_df = pd.read_csv(expanded_dict_path)

    # Assuming the first column is 'word' and the rest are categories
    word_column = expanded_df.columns[0]
    category_columns = expanded_df.columns[1:]

    word_to_categories_local = defaultdict(set)

    for index, row in expanded_df.iterrows():
        word = row[word_column]
        cleaned_word = clean_word(word)

        if not cleaned_word:
            logging.warning(f"Skipping invalid word at row {index}: {word}")
            continue

        for category in category_columns:
            try:
                if int(row[category]) == 1:
                    word_to_categories_local[cleaned_word].add(category)
            except ValueError:
                logging.warning(f"Invalid value at row {index}, column {category}: {row[category]}")

    all_categories = list(category_columns)

    # Debugging: Log loaded words and sample mappings
    logging.debug(f"Loaded {len(word_to_categories_local)} words with category mappings.")
    sample_mappings = dict(list(word_to_categories_local.items())[:5])  # Get first 5 mappings
    logging.debug("Sample word to categories mapping:")
    for word, categories in sample_mappings.items():
        logging.debug(f"  {word}: {categories}")

    return word_to_categories_local, all_categories

def init_compute_tf_idf(args):
    """
    Initializer function for multiprocessing Pool.
    Sets global variables accessible by compute_tf_idf functions.
    """
    global global_df_dict
    global global_N_doc
    global global_method
    global global_word_weights
    global word_to_categories
    global_df_dict, global_N_doc, global_method, global_word_weights, word_to_categories = args
    logging.debug(f"Initializer: Method={global_method}, N_doc={global_N_doc}, Loaded {len(global_df_dict)} DF entries.")

def compute_tf_idf_per_category_local(doc):
    """
    Compute TF-IDF per category for a single document.

    Returns:
        tuple:
            dict[str, float]: Mapping from category to TF-IDF score
            dict[str, float]: Mapping from word to its contribution
    """
    tf_idf_category = defaultdict(float)
    word_contrib = defaultdict(float)

    tokens = clean_text(doc)
    # Generate bigrams
    bigrams = [' '.join(gram) for gram in nltk.ngrams(tokens, 2)]
    tokens += bigrams

    token_counts = defaultdict(int)
    for token in tokens:
        token_counts[token] += 1

    for token, tf in token_counts.items():
        if token not in global_df_dict:
            continue
        df = global_df_dict[token]
        idf = math.log((global_N_doc + 1) / (df + 1)) + 1  # Smoothing
        weight = global_word_weights.get(token, 1.0) if global_word_weights else 1.0
        if global_method == "WFIDF":
            tf_weight = math.log(1 + tf)
        else:  # TFIDF
            tf_weight = tf
        tf_idf = tf_weight * idf * weight
        if token in word_to_categories:
            for category in word_to_categories[token]:
                tf_idf_category[category] += tf_idf
                word_contrib[token] += tf_idf
                logging.debug(f"Added TF-IDF {tf_idf:.4f} to category '{category}' for token '{token}'.")
            logging.debug(f"Current category scores: {tf_idf_category}")
        else:
            logging.debug(f"No category mapping for token: {token}")
        # Debugging statements
        logging.debug(f"Token: {token}, TF: {tf}, DF: {df}, IDF: {idf:.4f}, TF-IDF: {tf_idf:.4f}")
    logging.debug(f"TF-IDF scores per category: {tf_idf_category}")
    return tf_idf_category, word_contrib

def insert_documents(cursor, id_doc_dict):
    """Insert a batch of documents into the SQLite database."""
    data_to_insert = [(doc_id, text.strip()) for doc_id, text in id_doc_dict.items()]
    cursor.executemany('''
        INSERT OR REPLACE INTO documents (doc_id, text) VALUES (?, ?)
    ''', data_to_insert)

def construct_doc_level_corpus(sent_corpus_file, sent_id_file, limit=None):
    """
    Construct document-level corpus from sentence-level corpus and write to disk.

    Returns:
        int: The number of documents processed
    """
    logging.info("Constructing document-level corpus")

    # Define the path to the temp directory
    temp_dir = Path(gl.OUTPUT_FOLDER) / "scores" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Define SQLite database path
    db_path = temp_dir / 'documents.db'

    # Remove existing database if it exists to prevent conflicts
    if db_path.exists():
        db_path.unlink()

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table
    c.execute('''
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            text TEXT
        )
    ''')
    conn.commit()

    id_doc_dict = defaultdict(str)
    N_doc = 0  # Document counter

    with open(sent_corpus_file, 'r', encoding='utf-8') as corpus_f, \
         open(sent_id_file, 'r', encoding='utf-8') as id_f:

        for corpus_line, id_line in tqdm(zip(corpus_f, id_f), desc="Aggregating sentences"):
            corpus_line = corpus_line.strip()
            id_line = id_line.strip()
            if not id_line:
                continue  # Skip empty lines
            doc_id = id_line.split("_")[0]
            try:
                doc_id = str(int(float(doc_id)))  # Convert to integer, then to string
            except ValueError:
                logging.warning(f"Invalid doc_id: {doc_id}, skipping this line.")
                continue
            # Append the sentence to the corresponding document
            id_doc_dict[doc_id] += " " + corpus_line

            N_doc = len(id_doc_dict)

            # Check if the limit has been reached
            if limit is not None and N_doc >= limit:
                logging.info(f"Reached the limit of {limit} documents.")
                break

    # Insert documents into SQLite
    insert_documents(c, id_doc_dict)
    conn.commit()
    N_doc = len(id_doc_dict)
    logging.info(f"Processed {N_doc} documents.")

    # Retrieve all documents and their IDs
    c.execute('SELECT doc_id, text FROM documents')
    data = c.fetchall()
    conn.close()

    # Write to plain text files
    corpus_output_path = temp_dir / "corpus_doc_level.txt"
    doc_ids_output_path = temp_dir / "doc_ids.txt"

    with open(corpus_output_path, 'w', encoding='utf-8') as corpus_out, \
         open(doc_ids_output_path, 'w', encoding='utf-8') as id_out:

        for doc_id, text in tqdm(data, desc="Writing to text files"):
            corpus_out.write(text.strip() + "\n")
            id_out.write(doc_id.strip() + "\n")

    logging.info(f"Constructing document-level corpus is done! Total documents: {N_doc}")
    return N_doc

def calculate_overlap(corpus_file, word_to_categories_local):
    """
    Calculate the percentage of tokens in the corpus that are present in the word_to_categories mapping.

    Returns:
        float: Percentage of tokens that have category mappings
    """
    total_tokens = 0
    mapped_tokens = 0

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for doc in tqdm(f, desc="Calculating token overlap"):
            tokens = clean_text(doc)
            bigrams = [' '.join(gram) for gram in nltk.ngrams(tokens, 2)]
            tokens += bigrams
            total_tokens += len(tokens)
            mapped_tokens += sum(1 for token in tokens if token in word_to_categories_local)

    overlap_percentage = (mapped_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    print(f"Total Tokens: {total_tokens}, Mapped Tokens: {mapped_tokens}, Overlap: {overlap_percentage:.2f}%")
    logging.info(f"Token overlap with category mapping: {overlap_percentage:.2f}%")
    return overlap_percentage

def calculate_df(corpus_file):
    """
    Calculate and dump a document-frequency dictionary for all the tokens.

    Returns:
        dict[str, int]: Document frequency for each token
    """
    logging.info("Calculating document frequencies.")
    df_dict = defaultdict(int)

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for doc in tqdm(f, desc="Calculating DF"):
            tokens = clean_text(doc)
            bigrams = [' '.join(gram) for gram in nltk.ngrams(tokens, 2)]
            tokens += bigrams
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df_dict[token] += 1

    # Save df_dict to disk
    df_output_path = Path(gl.OUTPUT_FOLDER) / "scores" / "temp" / "doc_freq.pickle"
    with open(df_output_path, "wb") as f:
        pickle.dump(df_dict, f)

    logging.info("Document frequencies calculation is done.")
    return df_dict

def score_tf_idf_per_category(
    corpus_file,
    doc_ids_file,
    df_dict,
    N_doc,
    word_to_categories_local,
    all_categories,
    output_file,
    word_contributions_file,
    method,
    normalize=False,
    word_weights=None,
    num_workers=2,
    batch_size=1000,
    limit=None
):
    """
    Score documents using TF-IDF per category.
    """
    logging.info(f"Scoring using {method} per category.")

    with open(corpus_file, 'r', encoding='utf-8') as corpus_f, \
         open(doc_ids_file, 'r', encoding='utf-8') as id_f, \
         open(output_file, 'w', encoding='utf-8', newline='') as out_f, \
         open(word_contributions_file, 'w', encoding='utf-8', newline='') as wc_f:

        writer = csv.writer(out_f)
        # Write header: doc_id, category_1, category_2, ..., category_n
        header = ['doc_id'] + all_categories
        writer.writerow(header)

        wc_writer = csv.writer(wc_f)
        # Write header for word contributions: doc_id, word, category, contribution
        wc_writer.writerow(['doc_id', 'word', 'category', 'contribution'])

        args = (df_dict, N_doc, method, word_weights, word_to_categories_local)
        pool = Pool(
            processes=num_workers,
            initializer=init_compute_tf_idf,
            initargs=(args,)
        )

        batch_docs = []
        batch_ids = []
        processed_docs = 0  # Counter for processed documents

        # Determine the total number of documents to process
        total_docs = sum(1 for _ in open(corpus_file, 'r', encoding='utf-8'))
        if limit is not None:
            total_docs = min(limit, total_docs)

        corpus_f.seek(0)  # Reset file pointer
        id_f.seek(0)

        for doc_id, doc in tqdm(zip(id_f, corpus_f), desc=f"Scoring {method} per category", total=total_docs):
            doc_id = doc_id.strip()
            doc = doc.strip()
            batch_ids.append(doc_id)
            batch_docs.append(doc)
            processed_docs += 1

            if len(batch_docs) >= batch_size or processed_docs >= total_docs:
                # Compute TF-IDF scores in parallel
                results = pool.map(compute_tf_idf_per_category_local, batch_docs)

                # Write scores and word contributions to CSV
                for d_id, (tf_idf_cat, word_contrib) in zip(batch_ids, results):
                    row = [d_id] + [tf_idf_cat.get(cat, 0.0) for cat in all_categories]
                    logging.debug(f"Document {d_id} scores: {row}")
                    writer.writerow(row)
                    for word, contrib in word_contrib.items():
                        for category in word_to_categories_local.get(word, []):
                            wc_writer.writerow([d_id, word, category, contrib])

                # Clear batches
                batch_docs = []
                batch_ids = []
                logging.debug(f"Processed {processed_docs}/{total_docs} documents.")

            if processed_docs >= total_docs:
                break

        pool.close()
        pool.join()
    logging.info(f"{method} per category scoring is done.")

def main():
    """
    Main function to orchestrate the processing of documents.
    """
    try:
        # ============================
        # 1. Define File Paths
        # ============================
        sent_corpus_file = Path(gl.DATA_FOLDER) / "input" / "documents.txt"
        sent_id_file = Path(gl.DATA_FOLDER) / "input" / "document_ids.txt"
        current_dict_path = Path(gl.OUTPUT_FOLDER) / "dict" / "expanded_dict_cleaned.csv"  # Updated to match your file name

        # ============================
        # 2. Load Word-to-Categories Mapping
        # ============================
        word_to_categories, all_categories = load_word_to_categories(current_dict_path)

        # ============================
        # 3. Compute Word Similarity Weights (Optional)
        # ============================
        # If this function is essential, ensure it's correctly implemented
        try:
            word_sim_weights = attention_dictionary.compute_word_sim_weights(current_dict_path)
            logging.info("Computed word similarity weights.")
        except Exception as e:
            logging.warning(f"Could not compute word similarity weights: {e}")
            word_sim_weights = None

        # ============================
        # 4. Set Processing Limit
        # ============================
        limit = 10000  # Process only the first 10,000 documents

        # ============================
        # 5. Construct Document-Level Corpus
        # ============================
        logging.info("Starting to construct document-level corpus.")
        N_doc = construct_doc_level_corpus(
            sent_corpus_file=sent_corpus_file,
            sent_id_file=sent_id_file,
            limit=limit
        )
        logging.info(f"Document-level corpus constructed with {N_doc} documents.")

        # ============================
        # 6. Calculate Document Frequencies
        # ============================
        corpus_file = Path(gl.OUTPUT_FOLDER) / "scores" / "temp" / "corpus_doc_level.txt"
        doc_ids_file = Path(gl.OUTPUT_FOLDER) / "scores" / "temp" / "doc_ids.txt"
        logging.info("Starting to calculate document frequencies.")
        df_dict = calculate_df(corpus_file)
        logging.info(f"Document Frequency Dictionary contains {len(df_dict)} tokens.")

        # ============================
        # 7. Calculate Token Overlap
        # ============================
        logging.info("Starting to calculate token overlap with category mapping.")
        overlap = calculate_overlap(corpus_file, word_to_categories)
        logging.info(f"Token overlap with category mapping: {overlap:.2f}%")

        # ============================
        # 8. Score Documents Using TF-IDF Per Category
        # ============================
        methods = ["TFIDF"]  # Add other methods as needed e.g., "WFIDF"
        for method in methods:
            # Define output file paths
            output_scores_file = Path(gl.OUTPUT_FOLDER) / "scores" / method / f"scores_{method}.csv"
            word_contributions_file = Path(gl.OUTPUT_FOLDER) / "scores" / "word_contributions" / f"word_contribution_{method}.csv"

            # Ensure that the output directories exist
            output_scores_file.parent.mkdir(parents=True, exist_ok=True)
            word_contributions_file.parent.mkdir(parents=True, exist_ok=True)

            if method in ["TFIDF"]: # "WFIDF"
                # Score using TF-IDF per category with the specified limit
                score_tf_idf_per_category(
                    corpus_file=corpus_file,
                    doc_ids_file=doc_ids_file,
                    df_dict=df_dict,
                    N_doc=N_doc,
                    word_to_categories_local=word_to_categories,
                    all_categories=all_categories,
                    output_file=output_scores_file,
                    word_contributions_file=word_contributions_file,
                    method=method,
                    normalize=False,
                    word_weights=word_sim_weights,
                    num_workers=2,
                    batch_size=1000,  # Adjust batch size for better performance
                    limit=limit
                )
                logging.info(f"Scoring method '{method}' completed.")

        # ============================
        # 9. Final Confirmation
        # ============================
        logging.info("All scoring methods completed for the first 10,000 documents.")
        print("Processing completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
