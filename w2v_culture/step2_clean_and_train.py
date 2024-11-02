import datetime
import functools
import logging
import sys
from pathlib import Path

import pandas as pd

import global_options as gl
import parse
from culture import culture_models, file_util, preprocess

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def clean_file(in_file, out_file):
    """clean the entire corpus (output from CoreNLP)
    
    Arguments:
        in_file {str or Path} -- input corpus, each line is a sentence
        out_file {str or Path} -- output corpus
    """
    a_text_clearner = preprocess.text_cleaner()
    parse.process_largefile(
        input_file=in_file,
        output_file=out_file,
        input_file_ids=[
            str(i) for i in range(file_util.line_counter(in_file))
        ],  # fake IDs (do not need IDs for this function).
        output_index_file=None,
        function_name=functools.partial(a_text_clearner.clean),
        chunk_size=200000,
    )

# clean the parsed text (remove POS tags, stopwords, etc.) ----------------
clean_file(
    in_file=Path(gl.DATA_FOLDER_W2V, "processed", "parsed", "documents.txt"),
    out_file=Path(gl.DATA_FOLDER_W2V, "processed", "unigram", "documents.txt"),
)

# train and apply a phrase model to detect 2-word phrases ----------------
culture_models.train_bigram_model(
    input_path=Path(
        gl.DATA_FOLDER_W2V, "data", "processed", "unigram", "documents.txt"
    ),
    model_path=Path(gl.DATA_FOLDER_W2V, "models", "phrases", "bigram.mod"),
)
culture_models.file_bigramer(
    input_path=Path(
        gl.DATA_FOLDER_W2V, "processed", "unigram", "documents.txt"
    ),
    output_path=Path(
        gl.DATA_FOLDER_W2V, "data", "processed", "bigram", "documents.txt"
    ),
    model_path=Path(gl.DATA_FOLDER_W2V, "phrases", "bigram.mod"),
    scoring="original_scorer",
    threshold=gl.PHRASE_THRESHOLD,
)

# train and apply a phrase model to detect 3-word phrases ----------------
culture_models.train_bigram_model(
    input_path=Path(gl.DATA_FOLDER, "processed", "bigram", "documents.txt"),
    model_path=Path(gl.DATA_FOLDER_W2V, "phrases", "trigram.mod"),
)
culture_models.file_bigramer(
    input_path=Path(gl.DATA_FOLDER, "processed", "bigram", "documents.txt"),
    output_path=Path(
        gl.DATA_FOLDER, "processed", "trigram", "documents.txt"
    ),
    model_path=Path(gl.DATA_FOLDER_W2V, "models", "phrases", "trigram.mod"),
    scoring="original_scorer",
    threshold=gl.PHRASE_THRESHOLD,
)

# train the word2vec model ----------------
print(datetime.datetime.now())
print("Training w2v model...")
culture_models.train_w2v_model(
    input_path=Path(
        gl.DATA_FOLDER, "processed", "trigram", "documents.txt"
    ),
    model_path=Path(gl.DATA_FOLDER_W2V, "models", "w2v" "w2v.mod"),
    size=gl.W2V_DIM,
    window=gl.W2V_WINDOW,
    workers=gl.N_CORES,
    iter=gl.W2V_ITER,
)
