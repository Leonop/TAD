# Author: Zicheng Xiao
# Date: 2024-09-01
# Description: This script is used to preprocess the earnings call data.
# The data is stored in the data folder, and the preprocessed data is stored in the docword folder.
import codecs
import json
import re
import os
import string
import numpy as np
# import nltk
import spacy
import torch
from transformers import BertTokenizer, BertModel
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet
from datetime import datetime
import multiprocessing as mp
import pandas as pd
# ignore the warning
import warnings
warnings.filterwarnings("ignore")
import importlib,sys
importlib.reload(sys)
# sys.setdefaultencoding('utf-8')
path = sys.path[0]
import nltk
import datetime as dt
import global_options as gl
import warnings
import gensim
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import spacy
from tqdm import tqdm
tqdm.pandas()
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger')
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

class parallel_NLP_PreProcess:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.ps = PorterStemmer()
        self.sb = SnowballStemmer('english')
        self.stoplist = {word.strip().lower() for word in gl.stop_list}
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def remove_punct_and_digits(self, text):
        text = re.sub(r'[{}]'.format(string.punctuation), ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\d+', '', text)
        return text.strip()

    def remove_stopwords(self, tokens):
        return [word for word in tokens if isinstance(word, str) and word.lower() not in self.stoplist]

    def lemmatization(self, text, allowed_postags=['NOUN', 'ADJ', 'VERB']):
        doc = self.nlp(text)
        return [token.lemma_ for token in doc if token.pos_ in allowed_postags]

    def smart_ngrams(self, docs, min_count, threshold):
        bigram_phrases = Phrases(docs, min_count=min_count, threshold=threshold, delimiter='_')
        bigram = Phraser(bigram_phrases)
        trigram_phrases = Phrases(docs, min_count=min_count, threshold=threshold, delimiter='_')
        trigram = Phraser(trigram_phrases)

        bigram_trigram_docs = [
            bigram[doc] + [token for token in trigram[bigram[doc]] if '_' in token] for doc in docs
        ]
        return bigram_trigram_docs

    def process_chunk(self, chunk, col):
        """Apply preprocessing steps to a DataFrame chunk."""
        chunk[col] = chunk[col].astype(str).apply(self.remove_punct_and_digits)
        chunk[col] = chunk[col].apply(lambda x: [token.text for token in self.nlp(x) if not token.is_space])
        chunk[col] = chunk[col].apply(lambda x: self.remove_stopwords(x))
        chunk[col] = chunk[col].apply(lambda x: self.lemmatization(' '.join(x)))
        chunk[col] = pd.Series(self.smart_ngrams(chunk[col].tolist(), gl.MIN_COUNT, gl.THRESHOLD))
        chunk[col] = chunk[col].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        return chunk

    def parallel_preprocess(self, df, col, n_workers=4):
        """Parallelize the preprocessing using multiprocessing."""
        stime = datetime.now()

        # Split the DataFrame into chunks
        chunk_size = len(df) // n_workers
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        # Initialize multiprocessing Pool
        with mp.Pool(n_workers) as pool:
            processed_chunks = list(tqdm(pool.imap(lambda chunk: self.process_chunk(chunk, col), chunks),
                                         total=len(chunks), desc="Processing", ncols=80))

        # Concatenate all processed chunks back into a single DataFrame
        df_processed = pd.concat(processed_chunks).reset_index(drop=True)

        print(f"Processing completed in {datetime.now() - stime}")
        return df_processed[col]
