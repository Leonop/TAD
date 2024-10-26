# read the bitcoin_submission
# interpreter is Python 3.9.12
import pandas as pd
import numpy as np
import json
import os
import global_options as gl
from tqdm import tqdm

def load_data():
    # Check if the file exists
    if not os.path.exists(gl.file_path):
        raise FileNotFoundError(f"The file at path {gl.file_path} does not exist.")

    # Define the file path and the number of rows to read as a subsample
    chunk_size = gl.CHUNK_SIZE  # Adjust this number to read a subsample
    # Use chunksize to limit rows number per iteration
    meta = pd.DataFrame()
    try:
        chunk_reader = pd.read_csv(
                                    gl.file_path, 
                                    chunksize=chunk_size, 
                                    skiprows=range(1, gl.START_ROWS+1),
                                    nrows=gl.NROWS  # Adjust this number to read a subsample
                                    )
    except OSError as e:
        print(f"Error reading the file: {e}")
        raise

    # ANSI escape codes for green color
    GREEN = '\033[92m'
    RESET = '\033[0m'
    # Wrap the chunk reader with tqdm to track progress
    for chunk in tqdm(chunk_reader, total=gl.NROWS//chunk_size, bar_format=f'{GREEN}{{l_bar}}{{bar:20}}{{r_bar}}{RESET}'):
        filtered_chunk = chunk[(chunk["year"] <= gl.YEAR_FILTER) & (chunk["year"] >= gl.START_YEAR)] # Filter by START_YEAR and YEAR_FILTER
        filtered_chunk = filtered_chunk.reset_index()
        filtered_chunk = filtered_chunk.sort_values(by='isdelayed_flag', ascending=False).drop_duplicates(subset=gl.UNIQUE_KEYS, keep='first')
        meta = pd.concat([meta, filtered_chunk], ignore_index=True)       
    meta['date'] = pd.to_datetime(meta['mostimportantdateutc'])
    meta['quarter'] = meta['date'].dt.quarter
    return meta
    
def save_file(df, filename):
    output_dir = os.path.join(gl.DATA_FOLDER, 'input', f'document_{filename}.txt')
    with open(output_dir , 'w') as f:
        for _, group in df.groupby('transcriptid'):  # Group by transcript
            sentences = '    '.join(group['sentence'])  # Join sentences with 4 spaces
            f.write(sentences + '\n')  # Add a new line after each transcript

def save_file_id(df, filename):
    output_dir = os.path.join(gl.DATA_FOLDER, 'input', f'document_{filename}_id.txt')
    with open(output_dir, 'w') as f:
        for id, _ in df.groupby('transcriptid'):  # Group by transcript
            f.write(id + '\n')  # Add a new line after each transcript
            
def save_id2firm(df, filename):
    output_dir = os.path.join(gl.DATA_FOLDER, 'input', f'id2firm_{filename}.txt')
    id2firm = df[['transcriptid', 'companyid', 'gvkey', 'year', 'quarter']]
    # Save the DataFrame to a text file without the index
    id2firm.to_csv(output_dir, sep='\t', index=False, header=True)
        
def create_input_data():
    #load the data
    df = load_data()
    # @earnings calls has duplicates text for given company quarter and 
    # sort based on gl.UNIQUE_KEYS
    df = df.sort_values(by=gl.UNIQUE_KEYS)
    # deduplicate based on gl.UNIQUE_KEYS
    df = df.drop_duplicates(subset=gl.UNIQUE_KEYS, keep='last').reset_index(drop=True)
    # compute the length of the text
    df['PorQ'] = df['transcriptcomponenttypename'].apply(
        lambda x: 1 if x == 'Presenter Speech' 
                else (np.nan if x == 'Question and Answer Operator Message' else 0)
    )
    temp = df["word_count"].groupby([['companyid', 'keydevid', 'transcriptid', 'transcriptcomponentid', 'transcriptcomponenttypename']]).transform('sum').reset_index()
    df = pd.merge(df, temp, on=gl.UNIQUE_KEYS, how='left')
    pre = df[df['PorQ'] == 1]
    qa = df[df['PorQ'] == 0]
    que =  df[df['transcriptcomponenttypename'] == 'Question']
    ans = df[df['transcriptcomponenttypename'] == 'Answer']
    # save the transcript text by type
    save_file(pre, 'pre')
    save_file(qa, 'qa')
    save_file(que, 'q')
    save_file(ans, 'a')
    # save the transcript id by type
    save_file_id(pre, 'pre')
    save_file_id(qa,'qa')
    save_file_id(que, 'q')
    save_file_id(ans, 'a')
    # save the id2firm
    save_id2firm(df)
    

if __name__ == "__main__":
    create_input_data()