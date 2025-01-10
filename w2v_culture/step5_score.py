# author: Zicheng(Leo) Xiao
# Date: 2024-10-24
# Description: This script scores the documents using different methods from attention_dictionary.py
# The scores are saved to the "scores" folder in the output directory.
# virtualenv: cuda_env
"""
Output Format Example:

1. Term Frequency (TF) scores:
   scores/TF/{topic}/scores_TF_{topic}.csv
   Example for Cash_Flows topic:
   Doc_ID, Cash_Flows, document_length
   doc1,   5,         100
   doc2,   3,         80

2. TF-IDF and WF-IDF scores:
   scores/{method}/{topic}/scores_{method}_{topic}.csv
   Example for Cash_Flows topic:
   Doc_ID, Cash_Flows, document_length
   doc1,   0.45,      100
   doc2,   0.23,      80

3. Word Contributions:
   scores/word_contributions/{topic}/word_contribution_{method}_{topic}.csv
   Example:
   word,    contribution
   profit,  0.85
   revenue, 0.65
"""

import os
import pickle
from pathlib import Path
import pandas as pd
import sys
from tqdm import tqdm
import logging
from datetime import datetime
# Add parent directory (TAD) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Import modules
from narrativesBERT import attention_dictionary
import global_options as gl
import file_util

def setup_logging():
    """Setup logging configuration.
    
    Returns:
        Path -- Path to the log file
    """
    # Create logs directory
    log_dir = Path(gl.OUTPUT_FOLDER_W2V, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"step5_score_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),  # Convert Path to string
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Started logging to: {log_file}")
    return log_file

def construct_doc_level_corpus(sent_corpus_file, sent_id_file):
    """Construct document level corpus from sentence level corpus and write to disk.
    
    Arguments:
        sent_corpus_file {str or Path} -- Path to the sentence corpus file (documents.txt)
        sent_id_file {str or Path} -- Path to the sentence ID file (document_ids.txt)
    
    Returns:
        tuple -- (corpus, doc_ids, N_doc) containing:
            - corpus: list of document texts
            - doc_ids: list of document IDs
            - N_doc: number of documents
    """
    logging.info("Constructing doc level corpus")
    
    # Read sentence-level corpus and IDs
    sent_corpus = file_util.file_to_list(sent_corpus_file)
    sent_IDs = file_util.file_to_list(sent_id_file)
    assert len(sent_IDs) == len(sent_corpus), "Number of sentences and IDs must match"
    
    logging.info(f"Read {len(sent_corpus)} sentences")
    
    # Extract document IDs from sentence IDs (format: docID_sentenceID)
    doc_ids = [x.split("_")[0] for x in sent_IDs]
    
    # Concatenate sentences belonging to the same document
    logging.info("Aggregating sentences into documents...")
    id_doc_dict = {}
    for i, doc_id in enumerate(doc_ids):
        if doc_id not in id_doc_dict:
            id_doc_dict[doc_id] = sent_corpus[i]
        else:
            id_doc_dict[doc_id] += " " + sent_corpus[i]
    
    # Convert to lists
    corpus = list(id_doc_dict.values())
    doc_ids = list(id_doc_dict.keys())
    N_doc = len(corpus)
    
    logging.info(f"Created {N_doc} documents")
    
    # Save to disk
    output_dir = Path(gl.OUTPUT_FOLDER_W2V, "scores", "temp")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save corpus
    corpus_path = output_dir / "corpus_doc_level.pickle"
    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f)
    logging.info(f"Saved corpus to {corpus_path}")
    
    # Save document IDs
    doc_ids_path = output_dir / "doc_ids.pickle"
    with open(doc_ids_path, "wb") as f:
        pickle.dump(doc_ids, f)
    logging.info(f"Saved document IDs to {doc_ids_path}")
    
    return corpus, doc_ids, N_doc

def calculate_df(corpus):
    """Calculate document frequencies for all words."""
    logging.info("Calculating document frequencies")
    df_dict = {}
    for doc in tqdm(corpus, desc="Calculating document frequencies", colour="green"):
        # Get unique words in document
        words = set(doc.split())
        for word in words:
            df_dict[word] = df_dict.get(word, 0) + 1
    
    # Save to disk
    output_dir = Path(gl.OUTPUT_FOLDER_W2V, "scores", "temp")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "doc_freq.pickle", "wb") as f:
        pickle.dump(df_dict, f)
    
    return df_dict

def load_doc_level_corpus():
    """load the corpus constructed by construct_doc_level_corpus()
    
    Returns:
        [str], [str], int -- a tuple of a list of documents, a list of document IDs, and the number of documents
    """
    print("Loading document level corpus.")
    with open(
        Path(gl.OUTPUT_FOLDER_W2V, "scores", "temp", "corpus_doc_level.pickle"),
        "rb",
    ) as in_f:
        corpus = pickle.load(in_f)
    with open(
        Path(gl.OUTPUT_FOLDER_W2V, "scores", "temp", "doc_ids.pickle"), "rb"
    ) as in_f:
        doc_ids = pickle.load(in_f)
    assert len(corpus) == len(doc_ids)
    N_doc = len(corpus)
    return corpus, doc_ids, N_doc

def score_tf(documents, doc_ids, expanded_dict, **kwargs):
    """
    Score documents using term freq. 
    """
    logging.info("Scoring using Term-freq (tf).")
    score = attention_dictionary.score_tf(
        documents=documents,
        document_ids=doc_ids,
        expanded_words=expanded_dict,
        n_core=gl.N_CORES,
    )
    score.to_csv(
        Path(gl.OUTPUT_FOLDER_W2V, "scores", "TF", f"scores_TF_{kwargs['topic_name']}.csv"), index=False
    )


def score_tf_idf(documents, doc_ids, N_doc, method, expanded_dict, **kwargs):
    """Score documents using tf-idf and its variations
    Arguments:
        documents {[str]} -- list of documents
        doc_ids {[str]} -- list of document IDs
        N_doc {int} -- number of documents
        method {str} -- 
            TFIDF: conventional tf-idf 
            WFIDF: use wf-idf log(1+count) instead of tf in the numerator
            TFIDF/WFIDF+SIMWEIGHT: using additional word weights given by the word_weights dict
        expanded_dict {dict[str, set(str)]} -- expanded dictionary
    """
    if method == "TF":
        logging.info("Scoring TF.")
        score_tf(documents, doc_ids, expanded_dict, **kwargs)
    else:
        logging.info("Scoring TF-IDF.")
        # load document freq
        df_dict = pd.read_pickle(
            Path(gl.OUTPUT_FOLDER_W2V, "scores", "temp", "doc_freq.pickle")
        )
        # score tf-idf
        score, contribution = attention_dictionary.score_tf_idf(
            documents=documents,
            document_ids=doc_ids,
            expanded_words=expanded_dict,
            df_dict=df_dict,
            N_doc=N_doc,
            method=method,
            **kwargs
        )
        # if the folder "{method}" does not exist, create it
        if not os.path.exists(str(Path(gl.OUTPUT_FOLDER_W2V, "scores", f"{method}"))):
            os.makedirs(str(Path(gl.OUTPUT_FOLDER_W2V, "scores", f"{method}")))
        # save the document level scores (without dividing by doc length)
        score.to_csv(
            str(
                Path(
                    gl.OUTPUT_FOLDER_W2V,
                    "scores",
                    f"{method}",
                    f"scores_{method}.csv",
                )
            ),
            index=False,
        )
        # save word contributions
        pd.DataFrame.from_dict(contribution, orient="index").to_csv(
            Path(
                gl.OUTPUT_FOLDER_W2V,
                "scores",
                "word_contributions",
                f"word_contribution_{method}.csv",
            )
        )

def aggregate_topic_scores(method, topics):
    """Aggregate scores from all topics into a single file.
    
    Arguments:
        method {str} -- Scoring method (TF, TFIDF, or WFIDF)
        topics {list} -- List of topics to aggregate
    
    Returns:
        pandas.DataFrame -- Combined scores for all topics
    """
    logging.info(f"\nAggregating {method} scores for all topics...")
    
    # Initialize with first topic to get document IDs and length
    first_topic = topics[0]
    scores_path = Path(gl.OUTPUT_FOLDER_W2V, "scores", method, first_topic, f"scores_{method}_{first_topic}.csv")
    combined_df = pd.read_csv(scores_path)
    
    # Rename the topic column to match topic name
    topic_col = [col for col in combined_df.columns if col not in ['Doc_ID', 'document_length']][0]
    combined_df = combined_df.rename(columns={topic_col: first_topic})
    
    # Add other topics
    for topic in tqdm(topics[1:], desc=f"Aggregating {method} scores", colour="green"):
        scores_path = Path(gl.OUTPUT_FOLDER_W2V, "scores", method, topic, f"scores_{method}_{topic}.csv")
        topic_df = pd.read_csv(scores_path)
        
        # Get the topic score column (excluding Doc_ID and document_length)
        topic_col = [col for col in topic_df.columns if col not in ['Doc_ID', 'document_length']][0]
        
        # Add to combined dataframe
        combined_df[topic] = topic_df[topic_col]
    
    # Reorder columns: Doc_ID, topics..., document_length
    cols = ['Doc_ID'] + topics + ['document_length']
    combined_df = combined_df[cols]
    
    # Save combined scores
    output_dir = Path(gl.OUTPUT_FOLDER_W2V, "scores", method)
    output_file = output_dir / f"combined_scores_{method}.csv"
    combined_df.to_csv(output_file, index=False)
    logging.info(f"Saved combined {method} scores to {output_file}")
    
    # Print sample
    logging.info(f"\nSample of combined {method} scores (first 5 documents):")
    logging.info("\n" + combined_df.head().to_string())
    
    return combined_df

if __name__ == "__main__":
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting document scoring process...")
    
    # Step 1: Create corpus from input files
    logging.info("Creating document corpus from input files...")
    corpus, doc_ids, N_doc = construct_doc_level_corpus(
        sent_corpus_file=Path(gl.DATA_FOLDER_W2V, "input", "documents.txt"),
        sent_id_file=Path(gl.DATA_FOLDER_W2V, "input", "document_ids.txt")
    )
    logging.info(f"Created corpus with {N_doc} documents")
    
    # Step 2: Calculate and save document frequencies
    logging.info("Calculating document frequencies...")
    df_dict = calculate_df(corpus)
    logging.info(f"Calculated frequencies for {len(df_dict)} unique words")
    
    # Step 3: Load expanded dictionary
    expanded_dict_path = str(Path(gl.OUTPUT_FOLDER, "dict", "expanded_dict.csv"))
    logging.info(f"Loading expanded dictionary from {expanded_dict_path}")
    
    try:
        # Read the expanded dictionary CSV
        expanded_df = pd.read_csv(expanded_dict_path)
        logging.info(f"Found {len(expanded_df.columns)} topics in expanded dictionary:")
        for topic in expanded_df.columns:
            logging.info(f"- {topic}")
    except FileNotFoundError:
        logging.error(f"Could not find expanded dictionary at {expanded_dict_path}")
        logging.error("Please make sure the file exists at the correct location.")
        logging.error(f"Full path attempted: {os.path.abspath(expanded_dict_path)}")
        sys.exit(1)
    
    # Compute word similarity weights
    logging.info("Computing word similarity weights...")
    word_sim_weights = attention_dictionary.compute_word_sim_weights(expanded_dict_path)
    
    # Step 4: Score each topic separately
    methods = ["TF", "TFIDF", "WFIDF"]
    
    # Process each topic
    for topic in tqdm(expanded_df.columns, desc="Processing topics", colour="green", position=0):
        logging.info(f"\nProcessing topic: {topic}")
        
        # Create dictionary for this topic
        topic_words = expanded_df[topic].dropna().tolist()
        topic_dict = {topic: set(topic_words)}  # Convert to set for faster lookups
        logging.info(f"Number of words in {topic}: {len(topic_words)}")
        
        # Score using each method
        for method in tqdm(methods, desc=f"Computing scores for {topic}", colour="green", position=1, leave=False):
            logging.info(f"Computing {method} scores for topic {topic}...")
            
            # Create output directory
            output_dir = Path(gl.OUTPUT_FOLDER_W2V, "scores", method, topic)
            os.makedirs(output_dir, exist_ok=True)
            
            if method == "TF":
                # Term Frequency scoring
                score = attention_dictionary.score_tf(
                    documents=corpus,
                    document_ids=doc_ids,
                    expanded_words=topic_dict,
                    n_core=gl.N_CORES
                )
                
                # Save scores
                output_file = output_dir / f"scores_{method}_{topic}.csv"
                score.to_csv(output_file, index=False)
                logging.info(f"Saved {method} scores for {topic} to {output_file}")
                
            else:
                # TF-IDF variants
                score, contribution = attention_dictionary.score_tf_idf(
                    documents=corpus,
                    document_ids=doc_ids,
                    expanded_words=topic_dict,
                    df_dict=df_dict,
                    N_doc=N_doc,
                    method=method,
                    word_weights=word_sim_weights if "SIMWEIGHT" in method else None,
                    normalize=False
                )
                
                # Save scores
                output_file = output_dir / f"scores_{method}_{topic}.csv"
                score.to_csv(output_file, index=False)
                logging.info(f"Saved {method} scores for {topic} to {output_file}")
                
                # Save word contributions
                contrib_dir = Path(gl.OUTPUT_FOLDER_W2V, "scores", "word_contributions", topic)
                os.makedirs(contrib_dir, exist_ok=True)
                contrib_file = contrib_dir / f"word_contribution_{method}_{topic}.csv"
                pd.DataFrame.from_dict(contribution, orient="index").to_csv(contrib_file)
                logging.info(f"Saved word contributions for {topic} to {contrib_file}")
            
            # Log sample of scores
            logging.info(f"Sample of {method} scores for {topic} (first 5 documents):")
            logging.info("\n" + score.head().to_string())
    
    # After scoring all topics, aggregate results for each method
    logging.info("Aggregating scores across topics...")
    topics = list(expanded_df.columns)
    for method in methods:
        aggregate_topic_scores(method, topics)
    
    logging.info("Scoring and aggregation complete. Results saved in scores directory.")
    logging.info(f"Full log available at: {log_file}")