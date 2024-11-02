# clean_expanded_dict.py
# The reason for this script is to clean the dictionary from the pattern of the words that are not real words.
# author: Zicheng Xiao 
# date: 2024.09.02
# python version: 3.8
# usage: python clean_expanded_dict.py

# import packages
import os
import json
import pandas as pd
from tqdm import tqdm
import global_options as gl
import re
from pathlib import Path


# read the csv file with name with pattern as expanded_dict_{name}.csv
# name is the value of the list of keys of gl.SEED_WORDS
def read_csv(file_path: str) -> "pd.DataFrame":
    """Read the csv file with the file_path.
    Returns: df {pd.DataFrame}
    """
    df = pd.read_csv(file_path)
    return df

def get_file_path(name: str) -> str:
    """Get the file path of the csv file.
    Returns: file_path {str}
    """
    file_path = str(Path(gl.OUTPUT_FOLDER, "dict", "expanded_dict.csv"))
    return file_path

def clean_dict(expanded_words) -> "dict[str: list]":
    """Remove certain pattern that are in the seed words from the expanded dictionary.
    Returns: expanded_words_cleaned {dict[str: list]}

    remove pattern: word contains "_n" or "_\" or "\_n" or "``" or "_\_n" or "_u\d4" or"\__" or any word less than 2 letters
    """
    # Define the patterns to remove
    patterns = [r'_n', r'_\\', r'\\_n', r'``', r'_\\_n', r'_u\\d{4}', r'\\__', r'\\\d{4}']
    
    # Function to clean a word by removing the patterns
    def clean_word(word):
        for pattern in patterns:
            word = re.sub(pattern, '', word)
        return word

    # Clean the expanded words dictionary
    expanded_words_cleaned = {}
    for key, words in expanded_words.items():
        cleaned_words = [clean_word(word.replace('\\', '')) for word in words if len(word) >= 3 and not re.match(r'\d+k$', word)]
        expanded_words_cleaned[key] = [word for word in cleaned_words if word != '']
    return expanded_words_cleaned

def write_dict_to_csv(_dict, file_name):
    """write the expanded dictionary to a csv file, each dimension is a column, the header includes dimension names
    
    Arguments:
        culture_dict {dict[str, list[str]]} -- an expanded dictionary {dimension: [words]}
        file_name {str} -- where to save the csv file?
    """
    pd.DataFrame.from_dict(_dict, orient="index").transpose().to_csv(
        file_name, index=None
    )


if __name__ == "__main__":
    # use tqdm to show progress bar
    file_path_topic = get_file_path()
    expanded_words = read_csv(file_path_topic)
    expanded_words_cleaned = clean_dict(expanded_words)
    write_dict_to_csv(expanded_words_cleaned, str(Path(gl.OUTPUT_FOLDER, "dict", "expanded_dict_cleaned.csv")))
        
