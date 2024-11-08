import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import global_options as gl
import ast
import datetime
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
import multiprocessing
from functools import partial
import file_util
from collections import defaultdict
from narrativesBERT import attention_dictionary as ad

class DataCleaner:
    def __init__(self):
        self.topics = "topic_dictionary.json"
        self.path = os.getcwd()
        self.subtopics = ['revenue_earnings', 'financial_position',
       'cash flow', 'productivity', 'guidance', 'growth', 'dividends',
       'impairment', 'expansion', 'merge and acquisition', 'investment',
       'cost control', 'governance', 'social', 'environmental',
       'capital structure', 'market trend', 'macroeconomic', 'industry trend',
       'competition', 'challenge', 'uncertainty', 'political risk', 'economic risk',
       'liquidity risk', 'sovereign risk', 'credit risk', 'operational risk',
       'legal risk', 'climate risk', 'cybersecurity risk', 'technology',
       'artificial intelligence', 'automation', 'urgency', 'emphasize',
       'long term']
        self.topic_table = pd.read_json(os.path.join(self.path, "data", self.topics))

    def extract_dict_values(self, _dict):
        col_list = []
        col_values = []
        if _dict == '' or pd.isna(_dict):
            return col_list, col_values
        try:
            _dict = ast.literal_eval(_dict)
            if len(_dict) != 0:
                col_list = list(_dict.keys())
                col_values = list(_dict.values())
        except (SyntaxError, ValueError):
            print(f"Invalid input dictionary string: {_dict}")
            # You can decide how to handle the invalid input here, such as logging the error or skipping the input
        return col_list, col_values

    
    def process_df(self, col_name):
        temp = self.topic_table
        topic_list = temp[col_name].unique().tolist()
        # create new columns for each topic in self.df, set default value to 0
        for topic in topic_list:
            self.df[topic] = 0
        for i in tqdm(range(len(self.df))):
            dictionary = self.df.loc[i, col_name]
            if str(dictionary) == "nan":
                continue
            col_list, col_values = self.extract_dict_values(dictionary)
            # print(col_list)
            # print(col_values)
            if len(col_list) != 0:
                for idx in range(len(col_list)): 
                    col = col_list[idx]
                    #idx is the index of the topic in the col_list
                    self.df.loc[i,col] = col_values[idx]
        return self.df
    
    def load_doc_level_corpus(self):
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


    # Define a function to process each chunk of lines
    def process_lines(self, lines):
        # Example: Return line length for each line, or any other processing you want
        return [len(line) for line in lines]
            
    # Function to read a chunk of the file
    def read_in_chunks(filename, start_line, num_lines):
        with open(filename, 'r') as f:
            # Move to the starting line
            for _ in range(start_line):
                f.readline()
            # Read a specified number of lines
            lines = [f.readline().strip() for _ in range(num_lines)]
        return lines

    # main function to read file in parallel
    def read_file_in_parallel(self, filename, chunk_size=1000):
        # Determine the number of lines in the file
        with open(filename, 'r') as f:
            total_lines = sum(1 for line in f)
        # Create a pool of workers
        pool = multiprocessing.Pool()
        # Prepare tasks for each worker
        tasks = [(filename, i, chunk_size) for i in range(0, total_lines, chunk_size)]
        read_chunk = partial(self.read_in_chunks)
        # Process each chunk in parallel
        results = pool.starmap(read_chunk, tasks)
        # Close the pool and wait for all workers to complete
        pool.close()
        pool.join()
        # Optionally, process each chunk further or combine results
        processed_data = []
        for lines in results:
            processed_data.extend(self.process_lines(lines))  # Process lines as needed
        return processed_data
        
    def construct_doc_level_corpus(self, sent_corpus_file, sent_id_file):
        """Construct document level corpus from sentence level corpus and write to disk.
        Dump "corpus_doc_level.pickle" and "doc_ids.pickle" to Path(gl.OUTPUT_FOLDER, "scores", "temp"). 
        
        Arguments:
            sent_corpus_file {str or Path} -- The sentence corpus after parsing and cleaning, each line is a sentence
            sent_id_file {str or Path} -- The sentence ID file, each line correspond to a line in the sent_co(docID_sentenceID)
        
        Returns:
            [str], [str], int -- a tuple of a list of documents, a list of document IDs, and the number of documents
        """
        print("Constructing doc level corpus")
        # sentence level corpus
        sent_corpus = file_util.file_to_list(sent_corpus_file)
        sent_IDs = file_util.file_to_list(sent_id_file)
        assert len(sent_IDs) == len(sent_corpus)
        # doc id for each sentence
        doc_ids = [x.split("_")[0] for x in sent_IDs]
        # concat all text from the same doc
        id_doc_dict = defaultdict(lambda: "")
        for i, id in enumerate(doc_ids):
            id_doc_dict[id] += " " + sent_corpus[i]
        # create doc level corpus
        corpus = list(id_doc_dict.values())
        doc_ids = list(id_doc_dict.keys())
        assert len(corpus) == len(doc_ids)
        with open(
            Path(gl.OUTPUT_FOLDER, "scores", "temp", "corpus_doc_level.pickle"),
            "wb",
        ) as out_f:
            pickle.dump(corpus, out_f)
        with open(
            Path(gl.OUTPUT_FOLDER, "scores", "temp", "doc_ids.pickle"), "wb"
        ) as out_f:
            pickle.dump(doc_ids, out_f)
        N_doc = len(corpus)
        return corpus, doc_ids, N_doc
    
    
    def calculate_df(self, corpus):
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
    

    def load_doc_level_corpus(self):
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
    
    def score_tf(self, documents, doc_ids, expanded_dict, **kwargs):
        """
        Score documents using term freq. 
        """
        print("Scoring using Term-freq (tf).")
        score = ad.score_tf(
            documents=documents,
            document_ids=doc_ids,
            expanded_words=expanded_dict,
            n_core=gl.N_CORES,
        )
        score.to_csv(
            Path(gl.OUTPUT_FOLDER, "scores", "TF", f"scores_TF_{kwargs['topic_name']}.csv"), index=False
        )

    def score_tf_idf(self, documents, doc_ids, N_doc, method, expanded_dict, **kwargs):
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
            self.score_tf(documents, doc_ids, expanded_dict, **kwargs)
        else:
            print("Scoring TF-IDF.")
            # load document freq
            df_dict = pd.read_pickle(
                Path(gl.OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle")
            )
            # score tf-idf
            score, contribution = ad.score_tf_idf(
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
                        f"scores_{method}_{kwargs['topic_name']}.csv",
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
                    f"word_contribution_{method}_{kwargs['topic_name']}.csv",
                )
            )
            
if __name__ == "__main__":
    dc = DataCleaner()
    current_dict_path = str(Path(gl.OUTPUT_FOLDER, "dict", f"expanded_dict.csv"))
    
    culture_dict, all_dict_words = ad.read_dict_from_csv(
        current_dict_path
    )
        # words weighted by similarity rank (optional)
    word_sim_weights = ad.compute_word_sim_weights(dc.topic_table)

    ## Pre-score ===========================
    # aggregate processed sentences to documents
    corpus, doc_ids, N_doc = ad.construct_doc_level_corpus(
        sent_corpus_file=Path(
            gl.DATA_FOLDER, "input", "documents.txt"
        ),
        sent_id_file=Path(
            gl.DATA_FOLDER, "input", "document_ids.txt"
        ),
    )
    word_doc_freq = ad.calculate_df(corpus)

    ## Score ========================
    # create document scores
    methods = ["TFIDF"]
    for method in methods:
        ad.score_tf_idf(
            corpus,
            doc_ids,
            N_doc,
            method=method,
            expanded_dict=culture_dict,
            normalize=False,
            word_weights=word_sim_weights,
        )