# step5_score_v2.py

import itertools
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

# Assuming these modules are correctly defined
import file_util
import global_options as gl
from narrativesBERT import attention_dictionary

# Global variables to be set by the initializer
global_df_dict = {}
global_N_doc = 0
global_method = ""
global_word_weights = {}
word_to_categories = defaultdict(list)  # Initialize as empty

def load_word_to_categories(expanded_dict_path):
    """
    Load the expanded dictionary and create a mapping from words to categories.

    Arguments:
        expanded_dict_path (str or Path): Path to expanded_dict_cleaned.csv

    Returns:
        tuple:
            word_to_categories (dict): Mapping from word to list of associated categories
            all_categories (list): List of all categories
    """
    expanded_df = pd.read_csv(expanded_dict_path)

    # Assuming the first column is 'word' and the rest are categories
    word_column = expanded_df.columns[0]
    category_columns = expanded_df.columns[1:]

    word_to_categories_local = defaultdict(list)

    for _, row in expanded_df.iterrows():
        word = row[word_column]
        word = clean_word(word)
        for category in category_columns:
            if row[category] == 1:  # Adjust based on your indicator
                word_to_categories_local[word].append(category)

    all_categories = list(category_columns)

    return word_to_categories_local, all_categories

def init_compute_tf_idf(df_dict, N_doc, method, word_weights):
    """
    Initializer function for multiprocessing Pool.
    Sets global variables accessible by compute_tf_idf functions.
    """
    global global_df_dict
    global global_N_doc
    global global_method
    global global_word_weights
    global_df_dict = df_dict
    global_N_doc = N_doc
    global_method = method
    global_word_weights = word_weights

def compute_tf_idf(doc):
    """
    Compute TF-IDF score and word contributions for a single document.
    This function must be at the top level to be picklable.

    Returns:
        tuple: (tf_idf_score, contributions_dict)
    """
    words = doc.split()
    tf_idf_score = 0.0
    contributions = {}

    for word in words:
        df = global_df_dict.get(word, 0)
        if df == 0:
            continue
        idf = math.log((global_N_doc + 1) / (df + 1)) + 1  # Smoothing
        weight = global_word_weights.get(word, 1.0) if global_word_weights else 1.0

        if global_method == "WFIDF":
            tf_weight = math.log(1 + 1)  # Adjust numerator as needed
        else:  # TFIDF
            tf_weight = 1 + math.log(len(words)) if len(words) > 0 else 1

        tf_idf = tf_weight * idf * weight
        tf_idf_score += tf_idf
        contributions[word] = contributions.get(word, 0) + tf_idf

    return tf_idf_score, contributions

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
    words = doc.split()
    for word in words:
        if word not in global_df_dict:
            continue
        df = global_df_dict[word]
        idf = math.log((global_N_doc + 1) / (df + 1)) + 1  # Smoothing
        weight = global_word_weights.get(word, 1.0) if global_word_weights else 1.0
        if global_method == "WFIDF":
            tf_weight = math.log(1 + 1)  # Example adjustment
        else:  # TFIDF
            tf_weight = 1 + math.log(len(words)) if len(words) > 0 else 1
        tf_idf = tf_weight * idf * weight
        if word_to_categories.get(word):
            for category in word_to_categories[word]:
                tf_idf_category[category] += tf_idf
                word_contrib[word] += tf_idf
    return tf_idf_category, word_contrib

def insert_documents(cursor, id_doc_dict):
    """Insert a batch of documents into the SQLite database."""
    data_to_insert = [(doc_id, text.strip()) for doc_id, text in id_doc_dict.items()]
    cursor.executemany('''
        INSERT OR REPLACE INTO documents (doc_id, text) VALUES (?, ?)
    ''', data_to_insert)

def construct_doc_level_corpus(sent_corpus_file, sent_id_file):
    """
    Construct document-level corpus from sentence-level corpus and write to disk using SQLite.
    Dumps "corpus_doc_level.txt" and "doc_ids.txt" to Path(gl.OUTPUT_FOLDER, "scores", "temp").

    Arguments:
        sent_corpus_file (str or Path): The sentence corpus after parsing and cleaning, each line is a sentence
        sent_id_file (str or Path): The sentence ID file, each line corresponds to a line in the sent_corpus (docID_sentenceID)

    Returns:
        int: The number of documents processed
    """
    print("Constructing document-level corpus")

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

    batch_size = 10000  # Number of documents to process before committing
    id_doc_dict = defaultdict(str)
    N_doc = 0  # Document counter

    with open(sent_corpus_file, 'r', encoding='utf-8') as corpus_f, \
         open(sent_id_file, 'r', encoding='utf-8') as id_f:

        for corpus_line, id_line in tqdm(zip(corpus_f, id_f), desc="Aggregating sentences", total=None):
            corpus_line = corpus_line.strip()
            id_line = id_line.strip()
            if not id_line:
                continue  # Skip empty lines
            doc_id = id_line.split("_")[0]

            # Append the sentence to the corresponding document
            id_doc_dict[doc_id] += " " + corpus_line

            # If the batch size is reached, write to SQLite and clear the dictionary
            if len(id_doc_dict) >= batch_size:
                insert_documents(c, id_doc_dict)
                conn.commit()
                N_doc += len(id_doc_dict)
                id_doc_dict.clear()
                print(f"Processed {N_doc} documents...")

    # Insert any remaining documents
    if id_doc_dict:
        insert_documents(c, id_doc_dict)
        conn.commit()
        N_doc += len(id_doc_dict)
        id_doc_dict.clear()
        print(f"Processed {N_doc} documents...")

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
            corpus_out.write(text + "\n")
            id_out.write(doc_id + "\n")

    print(f"Constructing document-level corpus is done! Total documents: {N_doc}")
    return N_doc

def calculate_df(corpus_file):
    """
    Calculate and dump a document-frequency dictionary for all the words.

    Arguments:
        corpus_file (str or Path): Path to the corpus_doc_level.txt file

    Returns:
        dict[str, int]: Document frequency for each word
    """
    print("Calculating document frequencies.")
    df_dict = defaultdict(int)

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for doc in tqdm(f, desc="Calculating DF"):
            doc_splited = doc.split()
            words_in_doc = set(doc_splited)
            for word in words_in_doc:
                df_dict[word] += 1

    # Save df_dict to disk
    df_output_path = Path(gl.OUTPUT_FOLDER) / "scores" / "temp" / "doc_freq.pickle"
    with open(df_output_path, "wb") as f:
        pickle.dump(df_dict, f)

    print("Document frequencies calculation is done.")
    return df_dict

def score_tf_per_category(corpus_file, doc_ids_file, word_to_categories_local, all_categories, output_file, num_workers=2, batch_size=10000):
    """
    Score documents using term frequency (TF) per category.

    Arguments:
        corpus_file (str or Path): Path to the corpus_doc_level.txt file
        doc_ids_file (str or Path): Path to the doc_ids.txt file
        word_to_categories_local (dict): Mapping from word to list of associated categories
        all_categories (list): List of all categories
        output_file (str or Path): Path to save the TF scores CSV
        num_workers (int): Number of multiprocessing workers
        batch_size (int): Number of documents to process in each batch
    """
    print("Scoring using Term-freq (TF) per category.")

    def compute_tf_per_category(doc):
        """
        Compute TF per category for a single document.

        Returns:
            dict[str, int]: Mapping from category to TF score
        """
        tf_category = defaultdict(int)
        words = doc.split()
        for word in words:
            for category in word_to_categories_local.get(word, []):
                tf_category[category] += 1
        return tf_category

    with open(corpus_file, 'r', encoding='utf-8') as corpus_f, \
         open(doc_ids_file, 'r', encoding='utf-8') as id_f, \
         open(output_file, 'w', encoding='utf-8', newline='') as out_f:

        writer = csv.writer(out_f)
        # Write header: doc_id, category_1, category_2, ..., category_90
        header = ['doc_id'] + all_categories
        writer.writerow(header)

        pool = Pool(processes=num_workers)

        batch_docs = []
        batch_ids = []

        for doc_id, doc in tqdm(zip(id_f, corpus_f), desc="Scoring documents", total=None):
            doc_id = doc_id.strip()
            doc = doc.strip()
            batch_ids.append(doc_id)
            batch_docs.append(doc)

            if len(batch_docs) >= batch_size:
                # Compute TF per category in parallel
                tf_categories = pool.map(compute_tf_per_category, batch_docs)

                # Write to CSV
                for d_id, tf_cat in zip(batch_ids, tf_categories):
                    row = [d_id] + [tf_cat.get(cat, 0) for cat in all_categories]
                    writer.writerow(row)

                # Clear batches
                batch_docs = []
                batch_ids = []
                print(f"Processed {len(batch_ids)} documents...")

        # Process any remaining documents
        if batch_docs:
            tf_categories = pool.map(compute_tf_per_category, batch_docs)
            for d_id, tf_cat in zip(batch_ids, tf_categories):
                row = [d_id] + [tf_cat.get(cat, 0) for cat in all_categories]
                writer.writerow(row)
            print(f"Processed {len(batch_ids)} documents...")

        pool.close()
        pool.join()

    print("TF per category scoring is done.")

def score_tf_idf_per_category(
    corpus_file,
    doc_ids_file,
    df_dict,
    N_doc,
    word_to_categories_local,
    all_categories,
    output_file,
    word_contributions_file,
    normalize=False,
    word_weights=None,
    num_workers=2,
    batch_size=10000
):
    """
    Score documents using TF-IDF per category.

    Arguments:
        corpus_file (str or Path): Path to the corpus_doc_level.txt file
        doc_ids_file (str or Path): Path to the doc_ids.txt file
        df_dict (dict): Document frequency dictionary
        N_doc (int): Number of documents
        word_to_categories_local (dict): Mapping from word to list of associated categories
        all_categories (list): List of all categories
        output_file (str or Path): Path to save the TF-IDF scores CSV
        word_contributions_file (str or Path): Path to save word contributions CSV
        normalize (bool): Whether to normalize the scores
        word_weights (dict): Word weights if applicable
        num_workers (int): Number of multiprocessing workers
        batch_size (int): Number of documents to process in each batch
    """
    print("Scoring using TF-IDF per category.")

    with open(corpus_file, 'r', encoding='utf-8') as corpus_f, \
         open(doc_ids_file, 'r', encoding='utf-8') as id_f, \
         open(output_file, 'w', encoding='utf-8', newline='') as out_f, \
         open(word_contributions_file, 'w', encoding='utf-8', newline='') as wc_f:

        writer = csv.writer(out_f)
        # Write header: doc_id, category_1, category_2, ..., category_90
        header = ['doc_id'] + all_categories
        writer.writerow(header)

        wc_writer = csv.writer(wc_f)
        # Write header for word contributions: word, category, contribution
        wc_writer.writerow(['word', 'category', 'contribution'])

        pool = Pool(
            processes=num_workers,
            initializer=init_compute_tf_idf,
            initargs=(df_dict, N_doc, "TFIDF", word_weights)
        )

        batch_docs = []
        batch_ids = []

        for doc_id, doc in tqdm(zip(id_f, corpus_f), desc="Scoring TF-IDF per category", total=N_doc):
            doc_id = doc_id.strip()
            doc = doc.strip()
            batch_ids.append(doc_id)
            batch_docs.append(doc)

            if len(batch_docs) >= batch_size:
                # Compute TF-IDF scores in parallel
                results = pool.map(compute_tf_idf_per_category_local, batch_docs)

                # Write scores and word contributions to CSV
                for d_id, (tf_idf_cat, word_contrib) in zip(batch_ids, results):
                    row = [d_id] + [tf_idf_cat.get(cat, 0.0) for cat in all_categories]
                    writer.writerow(row)
                    for word, contrib in word_contrib.items():
                        for category in word_to_categories_local.get(word, []):
                            wc_writer.writerow([word, category, contrib])

                # Clear batches
                batch_docs = []
                batch_ids = []
                print(f"Processed {len(batch_ids)} documents...")

        # Process any remaining documents
        if batch_docs:
            results = pool.map(compute_tf_idf_per_category_local, batch_docs)
            for d_id, (tf_idf_cat, word_contrib) in zip(batch_ids, results):
                row = [d_id] + [tf_idf_cat.get(cat, 0.0) for cat in all_categories]
                writer.writerow(row)
                for word, contrib in word_contrib.items():
                    for category in word_to_categories_local.get(word, []):
                        wc_writer.writerow([word, category, contrib])
            print(f"Processed {len(batch_ids)} documents...")

        pool.close()
        pool.join()

    print("TF-IDF per category scoring is done.")

def calculate_overlap(corpus_file, word_to_categories_local):
    """
    Calculate the percentage of words in the corpus that are present in the word_to_categories mapping.

    Arguments:
        corpus_file (str or Path): Path to the corpus_doc_level.txt file
        word_to_categories_local (dict): Mapping from word to list of associated categories

    Returns:
        float: Percentage of words that have category mappings
    """
    total_words = 0
    mapped_words = 0

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for doc in tqdm(f, desc="Calculating word overlap"):
            words = doc.lower().split()
            total_words += len(words)
            mapped_words += sum(1 for word in words if word in word_to_categories_local)

    overlap_percentage = (mapped_words / total_words) * 100 if total_words > 0 else 0
    return overlap_percentage

def clean_word(word):
    """
    Remove punctuation from a word and convert it to lowercase.
    
    Arguments:
        word (Any): The word to clean.
    
    Returns:
        str or None: The cleaned word if valid, else None.
    """
    if pd.isna(word):
        return None  # Skip missing words
    try:
        word_str = str(word)
        # Remove punctuation and convert to lowercase
        return word_str.translate(str.maketrans('', '', string.punctuation)).lower()
    except Exception as e:
        # Log the error if needed
        print(f"Error cleaning word '{word}': {e}")
        return None


def main():
    # Paths
    sent_corpus_file = Path(gl.DATA_FOLDER) / "input" / "documents.txt"
    sent_id_file = Path(gl.DATA_FOLDER) / "input" / "document_ids.txt"

    # Read expanded dictionary and load word to categories mapping
    current_dict_path = Path(gl.OUTPUT_FOLDER) / "dict" / "expanded_dict_cleaned.csv"
    word_to_categories_loaded, all_categories = load_word_to_categories(current_dict_path)

    # Assign to global variable for access in worker functions
    global word_to_categories
    word_to_categories = word_to_categories_loaded

    # Debugging: Print the number of words and sample mappings
    print(f"Loaded {len(word_to_categories)} words with category mappings.")
    sample_mappings = dict(list(word_to_categories.items())[:5])  # Get first 5 mappings
    print("Sample word to categories mapping:")
    for word, categories in sample_mappings.items():
        print(f"  {word}: {categories}")

    # Debugging: Print all categories
    print(f"All categories ({len(all_categories)}): {all_categories[:5]} ...")  # Print first 5 categories

    # Optional: Compute word similarity weights
    word_sim_weights = attention_dictionary.compute_word_sim_weights(current_dict_path)

    # Step 1: Construct the document-level corpus
    N_doc = construct_doc_level_corpus(
        sent_corpus_file=sent_corpus_file,
        sent_id_file=sent_id_file,
    )

    # Step 2: Calculate document frequencies
    corpus_file = Path(gl.OUTPUT_FOLDER) / "scores" / "temp" / "corpus_doc_level.txt"
    df_dict = calculate_df(corpus_file)

    # Debugging: Print DF dictionary size
    print(f"Document Frequency Dictionary contains {len(df_dict)} words.")

    # Step 3: Calculate word overlap
    overlap = calculate_overlap(corpus_file, word_to_categories)
    print(f"Word overlap with category mapping: {overlap:.2f}%")

    # Step 4: Score documents
    methods = ["TFIDF", "WFIDF"]  # Add other methods as needed
    for method in methods:
        output_scores_file = Path(gl.OUTPUT_FOLDER) / "scores" / method / f"scores_{method}.csv"
        word_contributions_file = Path(gl.OUTPUT_FOLDER) / "scores" / "word_contributions" / f"word_contribution_{method}.csv"

        # Ensure output directories exist
        output_scores_file.parent.mkdir(parents=True, exist_ok=True)
        word_contributions_file.parent.mkdir(parents=True, exist_ok=True)

        if method in ["TFIDF", "WFIDF"]:
            score_tf_idf_per_category(
                corpus_file=corpus_file,
                doc_ids_file=Path(gl.OUTPUT_FOLDER) / "scores" / "temp" / "doc_ids.txt",
                df_dict=df_dict,
                N_doc=N_doc,
                word_to_categories_local=word_to_categories,
                all_categories=all_categories,
                output_file=output_scores_file,
                word_contributions_file=word_contributions_file,
                normalize=False,
                word_weights=word_sim_weights,
                num_workers=2,      # Adjust based on your system
                batch_size=10000,   # Adjust based on available memory
            )
        # Add other methods similarly if needed

    # Optional: Score TF per category
    # score_tf_per_category(
    #     corpus_file=corpus_file,
    #     doc_ids_file=Path(gl.OUTPUT_FOLDER) / "scores" / "temp" / "doc_ids.txt",
    #     word_to_categories_local=word_to_categories,
    #     all_categories=all_categories,
    #     output_file=Path(gl.OUTPUT_FOLDER) / "scores" / "TF" / "scores_TF.csv",
    #     num_workers=2,
    #     batch_size=10000,
    # )

    print("All scoring methods completed.")


if __name__ == "__main__":
    main()
