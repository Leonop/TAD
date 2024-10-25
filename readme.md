# Measuring management behavioral bias using machine learning

## Introduction
The repository implements a machine learning model to measure management behavioral bias. The model is based on the paper "Measuring Information Quality by Topic Attention Divergence: Evidence from Earnings Calls" by Zicheng Xiao, Tengfei Zhang, Jared Williams. 
The paper uses a dataset of earnings conference calls from 2005 - 2020.  The detail content structure is quantified, which is used to measure the bias. The model uses a machine learning model to evaluate the bias of the CEO based on the their attention divergence with investors.

The code is tested on Ubuntu 20.04 with Python 3.8.5.

## Requirements
- `Python 3.6+`
- The required Python packages can be installed via `pip install -r requirements.txt`


## Data

We included some example data in the `data/input/` folder. The three files are

- `documents.txt`: Each line is a document (e.g., each earnings call). Each document needs to have line breaks remvoed. The file has no header row.
- `document_ids.txt`: Each line is document ID (e.g., unique identifier for each earnings call). A document ID cannot have `_` or whitespaces. The file has no header row.
- (Optional) `id2firms.csv`: A csv file with three columns (`document_id`:str, `firm_id`:str, `time`:int). The file has a header row.