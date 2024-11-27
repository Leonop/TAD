# author: Zicheng(Leo) Xiao
# Date: 2024-10-24
# Description: This script scores the documents using different methods, including TF, TF-IDF, and WF-IDF.
# The scores are saved to the "scores" folder in the output directory.
# virtualenv: cuda_env

import itertools
import os
import pickle
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
import pandas as pd
from tqdm import tqdm as tqdm
import file_util
import global_options as gl
from narrativesBERT import attention_dictionary
import shelve


# @TODO: The scoring functions are not memory friendly. The entire pocessed corpus needs to fit in the RAM. Rewrite a memory friendly version.

def construct_doc_level_corpus(sent_corpus_file, sent_id_file):
    """Construct document-level corpus from sentence-level corpus and write to disk.
    Dump "corpus_doc_level.pickle" and "doc_ids.pickle" to Path(gl.OUTPUT_FOLDER, "scores", "temp").

    Arguments:
        sent_corpus_file {str or Path} -- The sentence corpus after parsing and cleaning, each line is a sentence
        sent_id_file {str or Path} -- The sentence ID file, each line corresponds to a line in the sent_corpus (docID_sentenceID)

    Returns:
        [str], [str], int -- a tuple of a list of documents, a list of document IDs, and the number of documents
    """
    print("Constructing document-level corpus")
    
    # Define the path to the temp directory
    temp_dir = os.path.join(gl.OUTPUT_FOLDER, "scores", "temp")
    # Check if the directory does not exist
    if not os.path.exists(temp_dir):
        # Create the directory
        os.makedirs(temp_dir)

    # Use shelve to store documents on disk
    shelve_path = os.path.join(temp_dir, 'doc_shelf.db')
    if os.path.exists(shelve_path):
        os.remove(shelve_path)

    with shelve.open(shelve_path, writeback=True) as id_doc_shelf:
        with open(sent_corpus_file, 'r') as corpus_f, open(sent_id_file, 'r') as id_f:
            for corpus_line, id_line in zip(corpus_f, id_f):
                corpus_line = corpus_line.strip()
                id_line = id_line.strip()
                doc_id = id_line.split("_")[0]

                # Append the sentence to the corresponding document
                if doc_id in id_doc_shelf:
                    id_doc_shelf[doc_id] += " " + corpus_line
                else:
                    id_doc_shelf[doc_id] = corpus_line

        # Ensure data is written to disk
        id_doc_shelf.sync()

        # Retrieve document IDs and texts
        doc_ids = list(id_doc_shelf.keys())
        corpus = [id_doc_shelf[doc_id] for doc_id in doc_ids]
        N_doc = len(doc_ids)

    # Save the corpus and doc_ids to disk
    with open(Path(temp_dir, "corpus_doc_level.pickle"), "wb") as out_f:
        pickle.dump(corpus, out_f)
    with open(Path(temp_dir, "doc_ids.pickle"), "wb") as out_f:
        pickle.dump(doc_ids, out_f)
    print("Constructing document-level corpus is done!")
    return corpus, doc_ids, N_doc


def calculate_df(corpus):
    """Calcualte and dump a document-freq dict for all the words.
    
    Arguments:
        corpus {[str]} -- a list of documents
    
    Returns:
        {dict[str: int]} -- document freq for each word
    """
    print("Calculating document frequencies.")
    # document frequency
    df_dict = defaultdict(int)
    for doc in tqdm(corpus):
        doc_splited = doc.split()
        words_in_doc = set(doc_splited)
        for word in words_in_doc:
            df_dict[word] += 1
    # save df dict
    with open(
        Path(gl.OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle"), "wb"
    ) as f:
        pickle.dump(df_dict, f)
    return df_dict


def load_doc_level_corpus():
    """load the corpus constructed by construct_doc_level_corpus()
    
    Returns:
        [str], [str], int -- a tuple of a list of documents, a list of document IDs, and the number of documents
    """
    print("Loading document level corpus.")
    with open(
        Path(gl.OUTPUT_FOLDER, "scores", "temp", "corpus_doc_level.pickle"),
        "rb",
    ) as in_f:
        corpus = pickle.load(in_f)
    with open(
        Path(gl.OUTPUT_FOLDER, "scores", "temp", "doc_ids.pickle"), "rb"
    ) as in_f:
        doc_ids = pickle.load(in_f)
    assert len(corpus) == len(doc_ids)
    N_doc = len(corpus)
    return corpus, doc_ids, N_doc


def score_tf(documents, doc_ids, expanded_dict, **kwargs):
    """
    Score documents using term freq. 
    """
    print("Scoring using Term-freq (tf).")
    score = attention_dictionary.score_tf(
        documents=documents,
        document_ids=doc_ids,
        expanded_words=expanded_dict,
        n_core=gl.N_CORES,
    )
    score.to_csv(
        Path(gl.OUTPUT_FOLDER, "scores", "TF", f"scores_TF_{kwargs['topic_name']}.csv"), index=False
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
        print("Scoring TF.")
        score_tf(documents, doc_ids, expanded_dict, **kwargs)
    else:
        print("Scoring TF-IDF.")
        # load document freq
        df_dict = pd.read_pickle(
            Path(gl.OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle")
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
        if not os.path.exists(str(Path(gl.OUTPUT_FOLDER, "scores", f"{method}"))):
            os.makedirs(str(Path(gl.OUTPUT_FOLDER, "scores", f"{method}")))
        # save the document level scores (without dividing by doc length)
        score.to_csv(
            str(
                Path(
                    gl.OUTPUT_FOLDER,
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
                gl.OUTPUT_FOLDER,
                "scores",
                "word_contributions",
                f"word_contribution_{method}.csv",
            )
        )


if __name__ == "__main__":
    current_dict_path = str(Path(gl.OUTPUT_FOLDER, "dict", f"expanded_dict_cleaned.csv"))
    
    culture_dict, all_dict_words = attention_dictionary.read_dict_from_csv(
        current_dict_path
    )
        # words weighted by similarity rank (optional)
    word_sim_weights = attention_dictionary.compute_word_sim_weights(current_dict_path)

    ## Pre-score ===========================
    # aggregate processed sentences to documents
    corpus, doc_ids, N_doc = construct_doc_level_corpus(
        sent_corpus_file=Path(
            gl.DATA_FOLDER, "input", "documents.txt"
        ),
        sent_id_file=Path(
            gl.DATA_FOLDER, "input", "document_ids.txt"
        ),
    )
    word_doc_freq = calculate_df(corpus)

    ## Score ========================
    # create document scores
    methods = ["TFIDF"]
    for method in methods:
        score_tf_idf(
            corpus,
            doc_ids,
            N_doc,
            method=method,
            expanded_dict=culture_dict,
            normalize=False,
            word_weights=word_sim_weights,
        )
