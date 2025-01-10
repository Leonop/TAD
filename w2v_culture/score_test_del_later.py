import os
import pickle
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import from w2v_culture package
from w2v_culture import step5_score
import global_options as gl
from narrativesBERT import attention_dictionary

def create_test_data():
    """Create sample test data"""
    # Sample documents
    test_corpus = [
        "cash flow increased revenue growth profit",
        "market share competitive pricing strategy",
        "dividend payment cash return profit",
        "revenue market growth strategy"
    ]
    test_doc_ids = ["doc1", "doc2", "doc3", "doc4"]
    
    # Sample dictionary with topics
    test_dict = {
        'Cash_Flows': ['cash', 'flow', 'revenue', 'profit'],
        'Market_Share': ['market', 'share', 'competitive', 'strategy'],
        'Growth': ['growth', 'increase', 'strategy'],
        'Dividend': ['dividend', 'payment', 'return', 'cash']
    }
    
    # Create temporary directory
    temp_dir = Path(gl.OUTPUT_FOLDER_W2V, "scores", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save test corpus and doc_ids
    with open(temp_dir / "corpus_doc_level.pickle", "wb") as f:
        pickle.dump(test_corpus, f)
    with open(temp_dir / "doc_ids.pickle", "wb") as f:
        pickle.dump(test_doc_ids, f)
    
    # Save test dictionary
    dict_dir = Path(gl.OUTPUT_FOLDER, "dict")
    os.makedirs(dict_dir, exist_ok=True)
    
    # Convert dictionary to DataFrame with a different approach
    rows = []
    max_len = max(len(v) for v in test_dict.values())
    
    # Create rows for the DataFrame
    for i in range(max_len):
        row = {}
        for topic in test_dict:
            row[topic] = test_dict[topic][i] if i < len(test_dict[topic]) else None
        rows.append(row)
    
    # Create DataFrame from rows and save
    df = pd.DataFrame(rows)
    df.to_csv(dict_dir / "expanded_dict.csv", index=False)
    
    return test_corpus, test_doc_ids

def test_scoring():
    """Test the scoring functionality"""
    print("Creating test data...")
    test_corpus, test_doc_ids = create_test_data()
    
    print("\nTest corpus:")
    for doc_id, doc in zip(test_doc_ids, test_corpus):
        print(f"{doc_id}: {doc}")
    
    # Calculate document frequencies (needed for TF-IDF)
    print("\nCalculating document frequencies...")
    df_dict = step5_score.calculate_df(test_corpus)
    N_doc = len(test_corpus)
    
    # Get expanded dictionary
    expanded_dict_path = str(Path(gl.OUTPUT_FOLDER, "dict", "expanded_dict.csv"))
    culture_dict, all_dict_words = attention_dictionary.read_dict_from_csv(expanded_dict_path)
    
    # Calculate word similarity weights (needed for SIMWEIGHT methods)
    word_sim_weights = attention_dictionary.compute_word_sim_weights(expanded_dict_path)
    
    # Test all scoring methods
    methods = ["TF", "TFIDF", "WFIDF", "TFIDF+SIMWEIGHT", "WFIDF+SIMWEIGHT"]
    
    for method in methods:
        print(f"\nTesting {method} scoring method...")
        if method == "TF":
            # Use TF scoring
            score = attention_dictionary.score_tf(
                documents=test_corpus,
                document_ids=test_doc_ids,
                expanded_words=culture_dict,
                n_core=1
            )
            print(f"\n{method} scoring results:")
            print(score.to_string())
            
            # Save results
            output_dir = Path(gl.OUTPUT_FOLDER_W2V, "scores", method)
            os.makedirs(output_dir, exist_ok=True)
            score.to_csv(output_dir / f"scores_{method}_test.csv", index=False)
        else:
            # Use TF-IDF variants
            score, contribution = attention_dictionary.score_tf_idf(
                documents=test_corpus,
                document_ids=test_doc_ids,
                expanded_words=culture_dict,
                df_dict=df_dict,
                N_doc=N_doc,
                method=method,
                word_weights=word_sim_weights if "SIMWEIGHT" in method else None,
                normalize=False
            )
            print(f"\n{method} scoring results:")
            print(score.to_string())
            
            # Save results
            output_dir = Path(gl.OUTPUT_FOLDER_W2V, "scores", method)
            os.makedirs(output_dir, exist_ok=True)
            score.to_csv(output_dir / f"scores_{method}_test.csv", index=False)
            
            # Save word contributions
            contrib_dir = Path(gl.OUTPUT_FOLDER_W2V, "scores", "word_contributions")
            os.makedirs(contrib_dir, exist_ok=True)
            pd.DataFrame.from_dict(contribution, orient="index").to_csv(
                contrib_dir / f"word_contribution_{method}_test.csv"
            )
    
    print("\nAll scoring methods completed. Results saved in scores directory.")

if __name__ == "__main__":
    test_scoring()



"""
scores/
├── TF/
│   └── scores_TF_test.csv
├── TFIDF/
│   └── scores_TFIDF_test.csv
├── WFIDF/
│   └── scores_WFIDF_test.csv
├── TFIDF+SIMWEIGHT/
│   └── scores_TFIDF+SIMWEIGHT_test.csv
├── WFIDF+SIMWEIGHT/
│   └── scores_WFIDF+SIMWEIGHT_test.csv
└── word_contributions/
    ├── word_contribution_TFIDF_test.csv
    ├── word_contribution_WFIDF_test.csv
    ├── word_contribution_TFIDF+SIMWEIGHT_test.csv
    └── word_contribution_WFIDF+SIMWEIGHT_test.csv
"""

"""
Example output:

Importing dict: /scrfs/storage/zichengx/home/Research/TAD/outputs/dict/expanded_dict.csv
Number of words in Cash_Flows dimension: 4
Number of words in Market_Share dimension: 4
Number of words in Growth dimension: 3
Number of words in Dividend dimension: 4

Testing TF scoring method...

TF scoring results:
   Cash_Flows  Dividend  Growth  Market_Share  document_length Doc_ID
0           4         1       1             0                6   doc1
1           0         0       1             4                5   doc2
2           2         4       0             0                5   doc3
3           1         0       2             2                4   doc4

Testing TFIDF scoring method...
Scoring using TFIDF
100%|████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 38746.46it/s]

TFIDF scoring results:
   Cash_Flows  Dividend    Growth  Market_Share  document_length Doc_ID
0    3.465736  0.693147  0.693147      0.000000              6.0   doc1
1    0.000000  0.000000  0.693147      4.158883              5.0   doc2
2    1.386294  4.852030  0.000000      0.000000              5.0   doc3
3    0.693147  0.000000  1.386294      1.386294              4.0   doc4

Testing WFIDF scoring method...
Scoring using WFIDF
100%|████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 33420.75it/s]

WFIDF scoring results:
   Cash_Flows  Dividend    Growth  Market_Share  document_length Doc_ID
0    3.465736  0.693147  0.693147      0.000000              6.0   doc1
1    0.000000  0.000000  0.693147      4.158883              5.0   doc2
2    1.386294  4.852030  0.000000      0.000000              5.0   doc3
3    0.693147  0.000000  1.386294      1.386294              4.0   doc4

Testing TFIDF+SIMWEIGHT scoring method...
Scoring using TFIDF+SIMWEIGHT
100%|████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 33420.75it/s]

TFIDF+SIMWEIGHT scoring results:
   Cash_Flows  Dividend  Growth  Market_Share  document_length Doc_ID
0    2.623213  0.430677     1.0       0.00000              6.0   doc1
1    0.000000  0.000000     0.5       3.76186              5.0   doc2
2    0.861353  4.692536     0.0       0.00000              5.0   doc3
3    0.500000  0.000000     1.5       1.50000              4.0   doc4

Testing WFIDF+SIMWEIGHT scoring method...
Scoring using WFIDF+SIMWEIGHT
100%|████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 32017.59it/s]

WFIDF+SIMWEIGHT scoring results:
   Cash_Flows  Dividend  Growth  Market_Share  document_length Doc_ID
0    2.623213  0.430677     1.0       0.00000              6.0   doc1
1    0.000000  0.000000     0.5       3.76186              5.0   doc2
2    0.861353  4.692536     0.0       0.00000              5.0   doc3
3    0.500000  0.000000     1.5       1.50000              4.0   doc4

All scoring methods completed. Results saved in scores directory.
"""