"""Global options for analysis
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

# Hardware options
N_CORES: int = 8  # max number of CPU cores to use
RAM_CORENLP: str = "32G"  # max RAM allocated for parsing using CoreNLP; increase to speed up parsing
PARSE_CHUNK_SIZE: int = 100 # number of lines in the input file to process uing CoreNLP at once. Increase on workstations with larger RAM (e.g. to 1000 if RAM is 64G)  

# Directory locations
os.environ[
    "CORENLP_HOME"
] = ""   
# location of the CoreNLP models; use / to seperate folders
DATA_FOLDER: str = "/home/zc_research/TAD/data"
MODEL_FOLDER: str = "/home/zc_research/TAD/models/" # will be created if does not exist
OUTPUT_FOLDER: str = "/home/zc_research/TAD/outputs/" # will be created if does not exist; !!! WARNING: existing files will be removed !!!
