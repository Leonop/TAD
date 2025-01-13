# write a code to compute the Topic Attention Divergence for each firm in id2firms (TAD)
# design the optimal algorithm to compute the 
# write a function to load the narratives data by sentence_id, 
# filepath = /outputs/scores/TF/combined_scores_TF.csv narratives_path = os.path.join(os.getcwd(), "w2v_culture", "outputs", "scores", "TF", "combined_scores_TF.csv")
# load the narratives data in the function in chunks of 100000 rows
# write a function to compute the TAD for each firm in id2firms
# the function should take the narratives data and the firm_id as input
# the function should return the TAD for the firm

# standardize the narratives data by divide the score based on the 
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys
import global_options as gl

def setup_logging():
    """Setup logging configuration."""
    try:
        # Create logs directory with absolute path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(current_dir, "outputs", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"TAD_computation_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info(f"Started logging to: {log_file}")
        return log_file
        
    except Exception as e:
        print(f"Error in setup_logging: {str(e)}")
        raise

class TAD:
    def __init__(self, model='TFIDF', analyst_feature="GenExp"):
        logging.info(f"Initializing TAD with model={model}, analyst_feature={analyst_feature}")
        
        self.narratives_path = os.path.join(os.getcwd(), "w2v_culture", "outputs", "scores", f"{model}", f"combined_scores_{model}.csv")
        logging.info(f"Narratives path: {self.narratives_path}")
        self.model_type = model
        self.analyst_feature = analyst_feature
        self.topics_ = []  # Initialize empty list
        self.id2firms_path = os.path.join(os.getcwd(), "data", "id2firms_alyst.csv")
        self.UNIQUE_KEYS_TAD = [
            'Doc_ID', 
            'gvkey', 
            'year', 
            'quarter', 
            'transcriptcomponenttypename'
        ]
        try:
            # Load narratives and get topics
            logging.info("Reading first chunk of data...")
            first_chunk = next(pd.read_csv(self.narratives_path, chunksize=100000))
            logging.info(f"First chunk columns: {first_chunk.columns.tolist()}")
            
            # Debug: Check first few rows
            logging.info(f"First few rows of data:\n{first_chunk.head()}")
            
            self.topics_ = [col for col in first_chunk.columns 
                        if col not in ['Doc_ID', 'sentence_id', 'gvkey', 'year', 'quarter', 
                                    'Ptranscriptcomponenttypename', 'document_length', 'GenExp']]
            logging.info(f"Found {len(self.topics_)} topics: {self.topics_}")
            
            # Initialize narratives iterator
            self.narratives = self.load_narratives()
        except Exception as e:
            logging.error(f"Error in initialization: {str(e)}")
            logging.error(f"Current working directory: {os.getcwd()}")
            raise

    def load_narratives(self):
        logging.info("Loading narratives in chunks...")
        try:
            narratives = pd.read_csv(self.narratives_path, chunksize=100000)
            return narratives
        except Exception as e:
            logging.error(f"Error loading narratives: {str(e)}")
            raise

    def merge_narratives_with_id2firms(self):
        """Merge narratives with id2firms data, processing all chunks."""
        logging.info("Loading and merging id2firms data...")
        try:
            # Load the id2firms data and log its structure
            id2firms = pd.read_csv(self.id2firms_path)
            logging.info(f"id2firms columns: {id2firms.columns.tolist()}")
            logging.info(f"id2firms first few rows:\n{id2firms.head()}")
            
            # Convert sentence_id to int in id2firms
            id2firms['sentenceid'] = id2firms['sentenceid'].astype(int)
            
            # Initialize list to store merged chunks
            all_merged_data = []
            chunk_count = 0
            
            # Create a fresh iterator for processing chunks
            chunks = pd.read_csv(self.narratives_path, chunksize=100000)
            
            # Process each chunk
            for chunk in tqdm(chunks, desc="Processing chunks"):
                chunk_count += 1
                logging.info(f"Processing chunk {chunk_count}")
                
                # Check for Doc_ID column
                if 'Doc_ID' not in chunk.columns:
                    logging.warning(f"Doc_ID not found in chunk {chunk_count}. Columns: {chunk.columns.tolist()}")
                    continue
                
                # Convert Doc_ID to int in chunk
                chunk['Doc_ID'] = chunk['Doc_ID'].astype(int)
                
                # Merge current chunk with id2firms
                merged_chunk = pd.merge(
                    id2firms,
                    chunk,
                    left_on='sentenceid',
                    right_on='Doc_ID',
                    how='inner'
                )
                
                if not merged_chunk.empty:
                    all_merged_data.append(merged_chunk)
                    logging.info(f"Chunk {chunk_count} merged successfully. Shape: {merged_chunk.shape}")
                else:
                    logging.warning(f"Chunk {chunk_count} produced no matches")
            
            if not all_merged_data:
                raise ValueError("No data was merged successfully")
            
            try:
                # Combine all chunks with progress information
                logging.info(f"Attempting to concatenate {len(all_merged_data)} chunks")
                final_merged_data = pd.concat(all_merged_data, ignore_index=True)
                logging.info(f"Final merged data shape: {final_merged_data.shape}")
                
                # Log the actual columns we got
                logging.info(f"Columns in final merged data: {final_merged_data.columns.tolist()}")
                
                # Verify the merged data has all required columns
                required_cols = self.UNIQUE_KEYS_TAD + self.topics_
                logging.info(f"Required columns: {required_cols}")
                
                missing_cols = [col for col in required_cols if col not in final_merged_data.columns]
                if missing_cols:
                    logging.error(f"Missing columns in merged data: {missing_cols}")
                    logging.error(f"Available columns: {final_merged_data.columns.tolist()}")
                    raise ValueError(f"Final merged data is missing columns: {missing_cols}")
                
                # Additional verification of data
                logging.info(f"Number of unique firms: {final_merged_data['gvkey'].nunique()}")
                logging.info(f"Date range: {final_merged_data['year'].min()}-{final_merged_data['year'].max()}")
                
                return final_merged_data
                
            except Exception as e:
                logging.error(f"Error in final data processing: {str(e)}")
                logging.error(f"Current working directory: {os.getcwd()}")
                logging.error("Stack trace:", exc_info=True)
                raise
                
        except Exception as e:
            logging.error(f"Error in merge_narratives_with_id2firms: {str(e)}")
            logging.error(f"Current working directory: {os.getcwd()}")
            logging.error("Stack trace:", exc_info=True)
            raise
    
    def cpt_firm_TAD(self, df, firm_id='gvkey'):
        """Compute TAD for firms."""
        try:
            logging.info(f"Computing TAD for firms using {firm_id} as identifier")
            
            # Initialize results list
            results = []
            
            # Group by firm, year, and quarter
            keys = [firm_id, 'year', 'quarter']
            logging.info(f"Grouping by keys: {keys}")
            
            grouped = df.groupby(keys)
            total_groups = len(grouped)
            logging.info(f"Found {total_groups} groups")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output file
            output_file = os.path.join(output_dir, f"TAD_score_{self.model_type}_{self.analyst_feature}.csv")
            
            # Write headers
            with open(output_file, "w") as f:
                f.write(f"{firm_id},year,quarter,tad_ps_q,tad_ps_a,tad_q_a\n")
            
            # Process each group
            for name, group in tqdm(grouped):
                try:
                    # Compute TAD scores
                    tad_ps_q, tad_ps_a, tad_q_a = self.get_v_a_con(group)
                    
                    # Store results
                    result = {
                        'firm_id': name[0],
                        'year': name[1],
                        'quarter': name[2],
                        'tad_ps_q': tad_ps_q,
                        'tad_ps_a': tad_ps_a,
                        'tad_q_a': tad_q_a
                    }
                    results.append(result)
                    
                    # Write to file
                    with open(output_file, "a") as f:
                        f.write(f"{name[0]},{name[1]},{name[2]},{tad_ps_q},{tad_ps_a},{tad_q_a}\n")
                    
                except Exception as e:
                    logging.warning(f"Error processing group {name}: {str(e)}")
                    continue
            
            logging.info(f"Successfully processed {len(results)} firms")
            return results
            
        except Exception as e:
            logging.error(f"Error in cpt_firm_TAD: {str(e)}")
            logging.error(f"DataFrame info:\n{df.info()}")
            logging.error("Stack trace:", exc_info=True)
            raise
            

    def add_filter_analyst_feature(self, df, feature):
        """Add weights based on analyst features."""
        try:
            logging.info(f"Adding weights for feature: {feature}")
            
            # Check if feature exists
            if feature not in df.columns:
                logging.info(f"Feature {feature} found in DataFrame")
                return df
            else:
                # Fill NaN values with 0
                df[feature] = df[feature].fillna(0)
                
                # Normalize feature values
                max_val = df[feature].max()
                if max_val > 0:
                    weights = df[feature] / max_val
                else:
                    weights = df[feature]
                
                # Apply weights to topic columns
                for topic in self.topics_:
                    df[topic] = df[topic] * weights
                
                logging.info("Successfully added analyst feature weights")
                return df
            
        except Exception as e:
            logging.error(f"Error in add_filter_analyst_feature: {str(e)}")
            raise

    def get_v_a_con(self, row_doc, section_id='transcriptcomponenttypename'):
        """Compute TAD scores for a group of data."""
        try:
            logging.info(f"Computing TAD for group with shape: {row_doc.shape}")
            
            # Add weights based on analyst features
            row_doc = self.add_filter_analyst_feature(row_doc, self.analyst_feature)
            
            # Group by section type and compute mean topic vectors
            section_vectors = row_doc.groupby(section_id)[self.topics_].mean()
            logging.info(f"Created section vectors with shape: {section_vectors.shape}")
            
            # Check for required sections
            required_sections = ['Presenter Speech', 'Question', 'Answer']
            missing_sections = set(required_sections) - set(section_vectors.index)
            if missing_sections:
                raise ValueError(f"Missing required sections: {missing_sections}")
            
            # Extract vectors for each section
            ps_vector = section_vectors.loc['Presenter Speech'].values
            q_vector = section_vectors.loc['Question'].values
            a_vector = section_vectors.loc['Answer'].values
            
            # Compute TAD scores (1 - cosine similarity)
            tad_ps_q = 1 - self.compute_cosine_similarity(ps_vector, q_vector)
            tad_ps_a = 1 - self.compute_cosine_similarity(ps_vector, a_vector)
            tad_q_a = 1 - self.compute_cosine_similarity(q_vector, a_vector)
            
            logging.info(f"Computed TAD scores - PS-Q: {tad_ps_q:.4f}, PS-A: {tad_ps_a:.4f}, Q-A: {tad_q_a:.4f}")
            
            return tad_ps_q, tad_ps_a, tad_q_a
            
        except Exception as e:
            logging.error(f"Error in get_v_a_con: {str(e)}")
            logging.error(f"Group data columns: {row_doc.columns.tolist()}")
            raise

    def compute_cosine_similarity(self, vector1, vector2):
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vector1 (numpy.ndarray): First vector
            vector2 (numpy.ndarray): Second vector
            
        Returns:
            float: Cosine similarity between the vectors
        """
        try:
            # Convert to numpy arrays if they aren't already
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # Check if vectors are 1D and convert if necessary
            if len(v1.shape) > 1:
                v1 = v1.flatten()
            if len(v2.shape) > 1:
                v2 = v2.flatten()
            
            # Compute cosine similarity
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logging.error(f"Error in compute_cosine_similarity: {str(e)}")
            logging.error(f"Vector1 shape: {np.array(vector1).shape}")
            logging.error(f"Vector2 shape: {np.array(vector2).shape}")
            raise

# In main:
if __name__ == "__main__":
    try:
        # Set up logging first
        setup_logging()
        logging.info("Starting TAD computation script")
        
        # Initialize TAD
        tad = TAD(model='TFIDF', analyst_feature="GenExp")
        
        # Process all chunks and merge with id2firms
        merged_data = tad.merge_narratives_with_id2firms()
        logging.info(f"Processing complete dataset with shape: {merged_data.shape}")
        
        # Compute TAD for all data
        results = tad.cpt_firm_TAD(merged_data)
        
        # Check if results is not None before getting length
        if results is not None:
            logging.info(f"Processed {len(results)} firms")
        else:
            logging.warning("No results were returned from TAD computation")
        
        logging.info("Script completed successfully")
        
    except Exception as e:
        logging.error(f"Script failed with error: {str(e)}")
        raise