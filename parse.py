# read the bitcoin_submission
# interpreter is Python 3.9.12
import pandas as pd
import numpy as np
import json
import os
import global_options as gl
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def load_chunk(chunk):
    """Process each chunk: filter, reset index, sort, and remove duplicates."""
    filtered_chunk = chunk[
        (chunk["year"] <= gl.YEAR_FILTER) & 
        (chunk["year"] >= gl.START_YEAR)
    ]
    filtered_chunk = filtered_chunk.reset_index(drop=True)
    return filtered_chunk

def load_data():
    """Load large CSV data in parallel using multiprocessing."""
    if not os.path.exists(gl.INPUT_file):
        raise FileNotFoundError(f"The file at path {gl.INPUT_file} does not exist.")

    chunk_size = gl.CHUNK_SIZE
    num_chunks = gl.NROWS // chunk_size + (gl.NROWS % chunk_size > 0)

    try:
        chunk_reader = pd.read_csv(
            gl.INPUT_file, 
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
    # drop the duplicated earnings calls and keep the first one
    meta = meta.sort_values(by=gl.UNIQUE_KEYS, ascending=True).drop_duplicates(subset=gl.UNIQUE_KEYS, keep='first')
    # Process date and quarter
    meta['date'] = pd.to_datetime(meta['mostimportantdateutc'])
    meta['quarter'] = meta['date'].dt.quarter
    # Generate unique sentence IDs for each group
    meta['sentenceid'] = meta.groupby(gl.UNIQUE_KEYS).cumcount()

    # Ensure sentenceid is a 16-digit integer by padding with leading zeros
    meta['sentenceid'] = meta['sentenceid'].apply(lambda x: f"{x:016d}").astype('int64')
    return meta

    
def save_DOC(df, filename):
    output_dir = os.path.join(gl.DATA_FOLDER, 'input', f'document_{filename}.txt')
    with open(output_dir , 'w') as f:
        for _, group in df.groupby(gl.UNIQUE_KEYS):  # Group by transcript
            sentences = '    '.join(group[gl.TEXT_COLUMN])  # Join sentences with 4 spaces
            f.write(str(sentences))  # Add a new line after each transcript
        f.write('\n')
        
def save_SENT(df, filename):
    output_dir = os.path.join(gl.DATA_FOLDER, 'input', f'sentence_{filename}.txt')
    with open(output_dir, 'w') as f:
        for _, row in df.iterrows():
            f.write(row[gl.TEXT_COLUMN] + '\n')  # Add a new line after each sentence
        
def save_file_DOC_id(df, filename):
    output_dir = os.path.join(gl.DATA_FOLDER, 'input', f'document_{filename}_id.txt')
    with open(output_dir, 'w') as f:
        for id, _ in df.groupby(gl.UNIQUE_KEYS):  # Group by earnings call transcript
            f.write(str(id) + '\n')  # Add a new line after each transcript

def save_file_SENT_id(df, filename):
    output_dir = os.path.join(gl.DATA_FOLDER, 'input', f'document_{filename}_sent_id.txt')
    with open(output_dir, 'w') as f:
        for sid, _ in df.groupby('sentenceid'):  # Group by earnings call transcript
            f.write(str(sid) + '\n')  # Add a new line after each transcript
            
def save_id2firm(df, filename):
    output_dir = os.path.join(gl.DATA_FOLDER, 'input', f'id2firms_{filename}.txt')
    id2firm = df[['transcriptid', 'companyid', 'gvkey', 'year', 'quarter']]
    # Save the DataFrame to a text file without the index
    id2firm.to_csv(output_dir, sep='\t', index=False, header=True)
        
def create_input_data():
    #load the data
    result_df = pd.DataFrame()
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
    composite_key = gl.UNIQUE_KEYS
    temp = df.groupby(composite_key, as_index=False)['word_count'].sum().reset_index().rename(columns={'word_count': 'total_word_count'})
    temp_p = df[df['PorQ']==1].groupby(composite_key, as_index=False)['word_count'].sum().reset_index().rename(columns={'word_count': 'pres_word_count'})
    temp_q = df[df['PorQ']==0].groupby(composite_key, as_index=False)['word_count'].sum().reset_index().rename(columns={'word_count': 'qa_word_count'})
    df.columns = df.columns.str.strip()
    temp.columns = temp.columns.str.strip()
    result_df = pd.merge(df, temp, on=composite_key, how='left')
    result_df = pd.merge(df, temp_p, on=composite_key, how='left')
    result_df = pd.merge(df, temp_q, on=composite_key, how='left')
    pre = df[df['PorQ'] == 1]
    qa = df[df['PorQ'] == 0]
    que =  df[df['transcriptcomponenttypename'] == 'Question']
    ans = df[df['transcriptcomponenttypename'] == 'Answer']
    # save the transcript text by type
    save_DOC(pre, 'pre')
    save_DOC(qa, 'qa')
    save_DOC(que, 'q')
    save_DOC(ans, 'a')
    # save the transcript id by type
    save_file_DOC_id(pre, 'pre')
    save_file_DOC_id(qa,'qa')
    save_file_DOC_id(que, 'q')
    save_file_DOC_id(ans, 'a')
    # save the id2firm
    save_id2firm(df, 'all')
    return result_df

if __name__ == "__main__":
    create_input_data()
    # The input data is completed!
    print("The input data is completed!")
    # Please check the /data/input folder for the output files.
    print("Please check the /data/input folder for the output files.")