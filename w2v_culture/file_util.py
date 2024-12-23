import datetime
import itertools
import os
import sys
import csv
from multiprocessing import Pool, freeze_support, cpu_count
from pathlib import Path
import global_options as gl
import pandas as pd
from tqdm import tqdm
from functools import partial


def line_counter(a_file):
    """Count the number of lines in a text file
    
    Arguments:
        a_file {str or Path} -- input text file
    
    Returns:
        int -- number of lines in the file
    """
    n_lines = 0
    with open(a_file, "rb") as f:
        n_lines = sum(1 for _ in f)
    return n_lines


def file_to_list(a_file):
    """Read a text file to a list, each line is an element
    
    Arguments:
        a_file {str or path} -- path to the file
    
    Returns:
        [str] -- list of lines in the input file, can be empty
    """
    file_content = []
    with open(a_file, "rb") as f:
        for l in f:
            file_content.append(l.decode(encoding="utf-8").strip())
    return file_content


def list_to_file(list, a_file, validate=True):
    """Write a list to a file, each element in a line
    The strings needs to have no line break "\n" or they will be removed
    
    Keyword Arguments:
        validate {bool} -- check if number of lines in the file
            equals to the length of the list (default: {True})
    """
    with open(a_file, "w", 8192000, encoding="utf-8", newline="\n") as f:
        for e in list:
            e = str(e).replace("\n", " ").replace("\r", " ")
            f.write("{}\n".format(e))
    if validate:
        assert line_counter(a_file) == len(list)


def read_large_file(a_file, block_size=10000):
    """A generator to read text files into blocks
    Usage: 
    for block in read_large_file(filename):
        do_something(block)
    
    Arguments:
        a_file {str or path} -- path to the file
    
    Keyword Arguments:
        block_size {int} -- [number of lines in a block] (default: {10000})
    """
    block = []
    with open(a_file) as file_handler:
        for line in file_handler:
            block.append(line)
            if len(block) == block_size:
                yield block
                block = []
    # yield the last block
    if block:
        yield block

def load_chunk(chunk):
    """Process each chunk: filter, reset index, sort, and remove duplicates."""
    filtered_chunk = chunk[
        (chunk["year"] <= gl.YEAR_FILTER) & 
        (chunk["year"] >= gl.START_YEAR)
    ]
    filtered_chunk = filtered_chunk.reset_index(drop=True)
    return filtered_chunk

def load_data(input_file):
    """Load large CSV data in parallel using multiprocessing."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The file at path {input_file} does not exist.")

    chunk_size = gl.CHUNK_SIZE
    num_chunks = gl.NROWS // chunk_size + (gl.NROWS % chunk_size > 0)

    try:
        chunk_reader = pd.read_csv(
            input_file, 
            quoting= csv.QUOTE_ALL,
            escapechar='\\',
            quotechar='"',
            error_bad_lines=False,  # Skip problematic rows
            chunksize=chunk_size, 
            skiprows=range(1, gl.START_ROWS + 1), 
            nrows=gl.NROWS,
            usecols=gl.SELECTED_COLS  # Load only selected columns
        )
    except OSError as e:
        print(f"Error reading the file: {e}")
        raise

    # ANSI color for green progress bar
    GREEN = '\033[92m'
    RESET = '\033[0m'

    # Use a multiprocessing pool to process chunks in parallel
    with Pool(cpu_count()) as pool:
        # Process chunks with tqdm to track progress
        chunks = list(tqdm(
            pool.imap(load_chunk, chunk_reader), 
            total=num_chunks, 
            bar_format=f'{GREEN}{{l_bar}}{{bar:20}}{{r_bar}}{RESET}'
        ))
    # Concatenate all processed chunks
    meta = pd.concat(chunks, ignore_index=True)
    # drop rows that keydevid is null
    print(f"Number of rows in the data Before dropna: {meta.shape}")
    meta = meta.dropna(subset=['gvkey', 'transcriptcomponentid'])
    print(f"Number of rows in the data After dropna: {meta.shape}")
    # drop the duplicated earnings calls and keep the first one
    meta = meta.sort_values(by=gl.UNIQUE_KEYS, ascending=True).drop_duplicates(subset=gl.UNIQUE_KEYS, keep='first')
    # Process date and quarter
    meta['date'] = pd.to_datetime(meta['mostimportantdateutc'])
    meta['quarter'] = meta['date'].dt.quarter
    meta['sentenceid'] = meta['transcriptcomponentid'].astype(str) 
    # write the processed data to files in input folder, documents.txt and document_ids.txt
    write_df_to_files(meta, 'sentenceid', 'componenttext', 2, 1000)
    save_id2firm(meta, 'all')
    return None

def save_id2firm(df, filename):
    output_dir = os.path.join(gl.DATA_FOLDER, 'input', f'id2firms_{filename}.txt')
    id2firm = df[['transcriptid', 'companyid', 'gvkey', 'year', 'quarter', 'date', 'transcriptcomponenttypename', 'sentenceid','componentorder', 'proid', 'transcriptpersonname', 'word_count']]
    # Save the DataFrame to a text file without the index
    id2firm.to_csv(output_dir, sep='\t', index=False, header=True)

def process_chunk(chunk_df, id_column, text_column):
    """Process a chunk of DataFrame."""
    try:
        cleaned_texts = chunk_df[text_column].astype(str).str.replace('\n', ' ', regex=False)
        return (
            '\n'.join(chunk_df[id_column].astype(str)),
            '\n'.join(cleaned_texts)
        )
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return "", ""

def write_df_to_files(df, id_column, text_column, n_workers=2, chunk_size=1000):
    id_file = os.path.join(gl.DATA_FOLDER, "input",  "document_ids.txt")
    text_file = os.path.join(gl.DATA_FOLDER, "input",  "documents.txt")
    
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Create partial function with fixed columns
    process_func = partial(process_chunk, id_column=id_column, text_column=text_column)
    
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, chunks),
            total=len(chunks),
            desc="Processing chunks"
        ))
        pool.close()  # Properly close the pool
        pool.join()   # Wait for all workers to finish
    
    # Write results to files
    with open(id_file, 'a', encoding='utf-8') as f_id, \
         open(text_file, 'a', encoding='utf-8') as f_text:
        for ids, texts in results:
            f_id.write(ids + '\n')
            f_text.write(texts + '\n')
