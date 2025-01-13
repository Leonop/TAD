from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys
import traceback  # Add this import
import global_options as gl

def create_sample_data():
    """Create a small sample dataset for testing."""
    # Create 9 records (3 companies x 3 section types)
    sample_data = {
        'sentence_id': range(1, 10),
        'gvkey': [1, 1, 1, 2, 2, 2, 3, 3, 3],  # 3 companies
        'year': [2020] * 9,
        'quarter': [1] * 9,
        'Ptranscriptcomponenttypename': [
            'Presenter Speech', 'Question', 'Answer',  # Company 1
            'Presenter Speech', 'Question', 'Answer',  # Company 2
            'Presenter Speech', 'Question', 'Answer'   # Company 3
        ],
        'document_length': [100] * 9,
        'GenExp': [5, 4, 3, 5, 4, 3, 5, 4, 3]
    }
    
    # Add topic columns with random scores
    topics = ['Cash Flows', 'Profit Margin', 'Revenue', 'Dividend', 'Growth']
    np.random.seed(42)  # For reproducibility
    for topic in topics:
        sample_data[topic] = np.random.random(9)  # Random scores between 0 and 1
    
    df = pd.DataFrame(sample_data)
    logging.info(f"Created sample data with shape: {df.shape}")
    logging.info(f"Section types per company:\n{df.groupby('gvkey')['Ptranscriptcomponenttypename'].value_counts()}")
    
    return df


class TAD:
    def __init__(self, model='TFIDF', analyst_feature="GenExp"):
        self.setup_logging()
        logging.info(f"Initializing TAD with model={model}, analyst_feature={analyst_feature}")
        
        # For testing, use sample data instead of reading from file
        self.sample_data = create_sample_data()
        
        self.model_type = model
        self.analyst_feature = analyst_feature
        self.topics_ = [col for col in self.sample_data.columns 
                       if col in ['Cash Flows', 'Profit Margin', 'Revenue', 'Dividend', 'Growth']]
        
        logging.info(f"Using topics: {self.topics_}")
        logging.info("Initialization complete")
        
    def setup_logging(self):
        """Setup logging configuration."""
        try:
            print("Setting up logging...")  # Debug print 1
            
            # Create logs directory with absolute path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(current_dir, "outputs", "logs")
            print(f"Log directory path: {log_dir}")  # Debug print 2
            
            os.makedirs(log_dir, exist_ok=True)
            print(f"Log directory created/exists: {os.path.exists(log_dir)}")  # Debug print 3
            
            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"TAD_computation_{timestamp}.log")
            print(f"Attempting to create log file at: {log_file}")  # Debug print 4
            
            # Clear any existing handlers
            logging.getLogger().handlers = []
            
            # Configure logging with both file and console output
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            
            print(f"Log file created at: {log_file}")  # Debug print 5
            logging.info(f"Started logging to: {log_file}")
            return log_file
            
        except Exception as e:
            print(f"Error in setup_logging: {str(e)}")
            print(traceback.format_exc())  # Print full traceback
            raise

    def compute_cosine_similarity(self, v1, v2):
        """Compute cosine similarity between two vectors."""
        try:
            # Add small epsilon to avoid division by zero
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_product == 0:
                return 0
            similarity = np.dot(v1, v2) / norm_product
            logging.info(f"Computed cosine similarity: {similarity:.4f}")
            return similarity
        except Exception as e:
            logging.error(f"Error in compute_cosine_similarity: {str(e)}")
            raise

    def get_v_a_con(self, group_data):
        """
        Compute TAD scores for a group of data.
        Returns: TAD scores between (PS-Q, PS-A, Q-A)
        """
        try:
            logging.info(f"Computing TAD for group with shape: {group_data.shape}")
            
            # Group by section type and compute mean topic vectors
            section_vectors = group_data.groupby('Ptranscriptcomponenttypename')[self.topics_].mean()
            logging.info(f"Created section vectors with shape: {section_vectors.shape}")
            
            # Extract vectors for each section
            try:
                ps_vector = section_vectors.loc['Presenter Speech'].values
                q_vector = section_vectors.loc['Question'].values
                a_vector = section_vectors.loc['Answer'].values
            except KeyError as e:
                logging.error(f"Missing section type: {e}")
                logging.info(f"Available sections: {section_vectors.index.tolist()}")
                raise
            
            # Compute TAD scores (1 - cosine similarity)
            tad_ps_q = 1 - self.compute_cosine_similarity(ps_vector, q_vector)
            tad_ps_a = 1 - self.compute_cosine_similarity(ps_vector, a_vector)
            tad_q_a = 1 - self.compute_cosine_similarity(q_vector, a_vector)
            
            logging.info(f"Computed TAD scores - PS-Q: {tad_ps_q:.4f}, PS-A: {tad_ps_a:.4f}, Q-A: {tad_q_a:.4f}")
            
            return tad_ps_q, tad_ps_a, tad_q_a
            
        except Exception as e:
            logging.error(f"Error in get_v_a_con: {str(e)}")
            logging.error(f"Group data columns: {group_data.columns.tolist()}")
            logging.error(f"Group data head:\n{group_data.head()}")
            raise

    def process_sample_data(self):
        """Process the sample data for testing."""
        try:
            logging.info("Starting sample data processing")
            logging.info(f"Sample data shape: {self.sample_data.shape}")
            logging.info(f"Sample data columns: {self.sample_data.columns.tolist()}")
            
            results = []
            # Group by firm and compute TAD
            for name, group in self.sample_data.groupby(['gvkey', 'year', 'quarter']):
                logging.info(f"\nProcessing group: {name}")
                try:
                    tad_ps_q, tad_ps_a, tad_q_a = self.get_v_a_con(group)
                    
                    # Store results
                    results.append({
                        'gvkey': name[0],
                        'year': name[1],
                        'quarter': name[2],
                        'TAD_PS_Q': tad_ps_q,
                        'TAD_PS_A': tad_ps_a,
                        'TAD_Q_A': tad_q_a
                    })
                    
                    logging.info(f"Group {name}:")
                    logging.info(f"  TAD PS-Q: {tad_ps_q:.4f}")
                    logging.info(f"  TAD PS-A: {tad_ps_a:.4f}")
                    logging.info(f"  TAD Q-A:  {tad_q_a:.4f}")
                except Exception as e:
                    logging.error(f"Error processing group {name}: {str(e)}")
                    continue
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            logging.info("\nFinal Results:")
            logging.info(f"\n{results_df}")
            
            logging.info("Sample processing completed")
            return results_df
        except Exception as e:
            logging.error(f"Error in process_sample_data: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        print("Script starting...")
        
        # Initialize TAD with sample data
        tad = TAD(model='TFIDF', analyst_feature="GenExp")
        print("TAD instance created successfully")
        
        # Show sample data
        print("\nSample data head:")
        print(tad.sample_data.head())
        
        # Process sample data
        print("\nProcessing sample data...")
        tad.process_sample_data()
        
        print("\nScript completed successfully")
        
    except Exception as e:
        print(f"Script failed with error: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)