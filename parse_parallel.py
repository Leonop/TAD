# read the bitcoin_submission
# interpreter is Python 3.9.12
import pandas as pd
import numpy as np
import json
import os
import global_options as gl
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import datetime
import itertools
import os
from multiprocessing import Pool
from pathlib import Path
import file_util as fu
from preprocess_parallel import parallel_NLP_PreProcess
    
def process_largefile(
    input_file,
    output_file,
    input_file_ids,
    output_index_file,
    function_name,
    chunk_size=100,
    start_index=None,
):
    """ A helper function that transforms an input file + a list of IDs of each line (documents + document_IDs) to two output files (processed documents + processed document IDs) by calling function_name on chunks of the input files. Each document can be decomposed into multiple processed documents (e.g. sentences). 
    Supports parallel with Pool.

    Arguments:
        input_file {str or Path} -- path to a text file, each line is a document
        ouput_file {str or Path} -- processed linesentence file (remove if exists)
        input_file_ids {str]} -- a list of input line ids
        output_index_file {str or Path} -- path to the index file of the output
        function_name {callable} -- A function that processes a list of strings, list of ids and return a list of processed strings and ids.
        chunk_size {int} -- number of lines to process each time, increasing the default may increase performance
        start_index {int} -- line number to start from (index starts with 0)

    Writes:
        Write the ouput_file and output_index_file
    """
    try:
        if start_index is None:
            # if start from the first line, remove existing output file
            # else append to existing output file
            os.remove(str(output_file))
            os.remove(str(output_index_file))
    except OSError:
        pass
    assert fu.line_counter(input_file) == len(
        input_file_ids
    ), "Make sure the input file has the same number of rows as the input ID file. "

    with open(input_file, newline="\n", encoding="utf-8", errors="ignore") as f_in:
        line_i = 0
        # jump to index
        if start_index is not None:
            # start at start_index line
            for _ in range(start_index):
                next(f_in)
            input_file_ids = input_file_ids[start_index:]
            line_i = start_index
        for next_n_lines, next_n_line_ids in zip(
            itertools.zip_longest(*[f_in] * chunk_size),
            itertools.zip_longest(*[iter(input_file_ids)] * chunk_size),
        ):
            line_i += chunk_size
            print(datetime.datetime.now())
            print(f"Processing line: {line_i}.")
            next_n_lines = list(filter(None.__ne__, next_n_lines))
            next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))
            output_lines = []
            output_line_ids = []
            with Pool(gl.N_CORES) as pool:
                for output_line, output_line_id in pool.starmap(
                    function_name, zip(next_n_lines, next_n_line_ids)
                ):
                    output_lines.append(output_line)
                    output_line_ids.append(output_line_id)
            output_lines = "\n".join(output_lines) + "\n"
            output_line_ids = "\n".join(output_line_ids) + "\n"
            with open(output_file, "a", newline="\n") as f_out:
                f_out.write(output_lines)
            if output_index_file is not None:
                with open(output_index_file, "a", newline="\n") as f_out:
                    f_out.write(output_line_ids)
        
# def create_input_data():
#     #load the data
#     result_df = pd.DataFrame()
#     df = fu.load_data()
#     # @earnings calls has duplicates text for given company quarter and 
#     # sort based on gl.UNIQUE_KEYS
#     df = df.sort_values(by=gl.UNIQUE_KEYS)
#     # deduplicate based on gl.UNIQUE_KEYS
#     df = df.drop_duplicates(subset=gl.UNIQUE_KEYS, keep='last').reset_index(drop=True)
#     # compute the length of the text
#     df['PorQ'] = df['transcriptcomponenttypename'].apply(
#         lambda x: 1 if x == 'Presenter Speech' 
#                 else (np.nan if x == 'Question and Answer Operator Message' else 0)
#     )
#     composite_key = gl.UNIQUE_KEYS
#     temp = df.groupby(composite_key, as_index=False)['word_count'].sum().reset_index().rename(columns={'word_count': 'total_word_count'})
#     temp_p = df[df['PorQ']==1].groupby(composite_key, as_index=False)['word_count'].sum().reset_index().rename(columns={'word_count': 'pres_word_count'})
#     temp_q = df[df['PorQ']==0].groupby(composite_key, as_index=False)['word_count'].sum().reset_index().rename(columns={'word_count': 'qa_word_count'})
#     df.columns = df.columns.str.strip()
#     temp.columns = temp.columns.str.strip()
#     result_df = pd.merge(df, temp, on=composite_key, how='left')
#     result_df = pd.merge(df, temp_p, on=composite_key, how='left')
#     result_df = pd.merge(df, temp_q, on=composite_key, how='left')
#     pre = df[df['PorQ'] == 1]
#     qa = df[df['PorQ'] == 0]
#     que =  df[df['transcriptcomponenttypename'] == 'Question']
#     ans = df[df['transcriptcomponenttypename'] == 'Answer']
#     # save the transcript text by type
#     save_DOC(pre, 'pre')
#     save_file(qa, 'qa')
#     save_file(que, 'q')
#     save_file(ans, 'a')
#     # save the transcript id by type
#     save_file_id(pre, 'pre')
#     save_file_id(qa,'qa')
#     save_file_id(que, 'q')
#     save_file_id(ans, 'a')
#     # save the id2firm
#     save_id2firm(df, 'all')
#     return result_df

if __name__ == "__main__":
    # pnp = parallel_NLP_PreProcess()
    fu.load_data(gl.INPUT_file)
 
    # # Configure paths
    # DATA_FOLDER = "data"
    # input_file = gl.INPUT_file
    # output_file = Path(DATA_FOLDER, "processed", "parsed", "documents.txt")
    # output_index_file = Path(DATA_FOLDER, "processed", "parsed", "document_ids.txt")
    
    # # Read document IDs
    # with open(Path(DATA_FOLDER, "input", "document_ids1.txt")) as f:
    #     input_file_ids = f.read().splitlines()
    
    # # Process documents
    # process_largefile(
    #     input_file=str(input_file),
    #     output_file=str(output_file),
    #     input_file_ids=input_file_ids,
    #     output_index_file=str(output_index_file),
    #     n_workers=4,
    #     chunk_size=100
    # )
    # create_input_data()
    # # The input data is completed!
    # print("The input data is completed!")
    # # Please check the /data/input folder for the output files.
    # print("Please check the /data/input folder for the output files.")