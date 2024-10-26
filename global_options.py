"""Global options for analysis
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
# Hardware options
N_CORES: int = 8  # max number of CPU cores to use
RAM_CORENLP: str = "32G"  # max RAM allocated for parsing using CoreNLP; increase to speed up parsing
PARSE_CHUNK_SIZE: int = 100 # number of lines in the input file to process uing CoreNLP at once. Increase on workstations with larger RAM (e.g. to 1000 if RAM is 64G)  

PROJECT_DIR = os.getcwd()
# Input data options
file_path = os.path.join('..', 'narrativesBERT','data', 'earnings_calls_20231017.csv')

# Directory locations
UNIQUE_KEYS = ['companyid', 'keydevid', 'transcriptid', 'transcriptcomponentid', 'transcriptcomponenttypename'] # composite key in earnings call data

PROJECT_DIR = os.getcwd()
DATA_FOLDER = os.path.join(PROJECT_DIR, "data")
model_folder = os.path.join(PROJECT_DIR, "models")
output_folder = os.path.join(PROJECT_DIR, "outputs")
output_fig_folder = os.path.join(output_folder, "fig")
data_filename = 'earnings_calls_20231017.csv'
DATE_COLUMN = "transcriptcreationdate_utc"
TOPIC_SCATTER_PLOT = os.path.join(output_fig_folder, "topic_scatter_plot.pdf")
TEXT_COLUMN = "componenttext" # the column in the main earnings call data that contains the earnings transcript
START_ROWS = 2000000 # start row to read from the csv file
NROWS = 10000000 # number of rows to read from the csv file
CHUNK_SIZE = 1000 # number of rows to read at a time
YEAR_FILTER = 2013 # train the model on data from start year to this year
START_YEAR = 2011 # start year of the data
# Batch Size for Bert Topic Model Training in BERTopic_big_data_hpc.py
BATCH_SIZE = 1000