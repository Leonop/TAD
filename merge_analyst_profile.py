# Python environment
# python 3.8.19
# virtualenv: 3.8.19 bertopic_env 
import pandas as pd
import os 
import numpy as np
from tqdm import tqdm
import gc  # For garbage collection
import logging
from datetime import datetime
import global_options as glo

class MAP:
    def __init__(self, output_path, input_filename):
        # Define all paths in __init__
        self.base_path = os.getcwd()
        self.id2firms_path = os.path.join(self.base_path, "data", "input", "id2firm_2006_2020.txt")
        self.id2firms_2021_2024_path = os.path.join(self.base_path, "data", "input", "id2firm_2021_2024.txt")
        self.proid_unique_path = os.path.join(self.base_path, "data", "proid_unique_20231017.dta")
        self.all_star_path = os.path.join(self.base_path, "data", "Allstar.dta")
        self.analyst_exp_path = os.path.join(self.base_path, "data", "analyst_experience_fyear_20220611.dta")
        self.FAFA_path = os.path.join(self.base_path, "data", "Forecast_Accuracy_1Q_20240112.dta")
        self.link_path = os.path.join(self.base_path, "data", "CCM_linktable.dta")
        
        # Initialize data containers
        self.proid_unique = None
        self.all_star = None
        self.analyst_exp = None
        self.FAFA = None
        self.link = None
        self.output_path = output_path
        self.input_path = os.path.join(self.base_path, "data", "input", input_filename)
        # Set up logging
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(self.base_path, "log_files")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"merge_analyst_profile_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # This will also print to console
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, path):
        """Load data from various file formats"""
        if path.endswith(".dta"):
            return pd.read_stata(path)
        elif path.endswith(".txt"):
            return pd.read_csv(path, sep='\t', header=0)
        else:
            raise ValueError(f"Unsupported file type: {path}")

    def color_text(self, text):
        """Helper function to color entire text in red"""
        # ANSI escape codes for colors
        RED = '\033[91m'
        RESET = '\033[0m'
        
        # Color the entire text red
        return f"{RED}{text}{RESET}"
    
    def load_and_prepare_reference_data(self):
        """Load and prepare reference datasets"""
        # Load proid_unique
        self.proid_unique = self.load_data(self.proid_unique_path)
        self.logger.info(f"proid_unique descriptive: {self.proid_unique.describe()}")
        self.logger.info(f"proid_unique columns: {self.proid_unique.columns.tolist()}")
        self.logger.info(f"proid_unique sample:\n{self.proid_unique.head()}")
        # Load analyst experience data
        self.analyst_exp = self.load_data(self.analyst_exp_path)
        self.logger.info(f"analyst_exp descriptive: {self.analyst_exp.describe()}")
        # Load all star data
        self.all_star = self.load_data(self.all_star_path)
        self.logger.info(f"all_star descriptive: {self.all_star.describe()}")
        # Load FAFA data
        self.FAFA = self.load_data(self.FAFA_path)
        self.logger.info(f"FAFA descriptive: {self.FAFA.describe()}")
        
    def convert_cusip_to_8_digits(self, df):
        """Convert cusip to 8 digits"""
        # Add debugging
        self.logger.info("Converting cusips to 8 digits")
        self.logger.info(f"Original cusip sample: {df['cusip'].head()}")
        # Remove any leading/trailing whitespace
        df['cusip'] = df['cusip'].str.strip()
        # Extract first 8 characters if longer
        df['cusip'] = df['cusip'].str[:8]
        # Fill with leading zeros if shorter
        df['cusip'] = df['cusip'].str.zfill(8)
        return df

    def process_link_table(self):
        """Process link table"""
        self.link = self.load_data(self.link_path)
        
        self.link['LINKDT'] = pd.to_datetime(self.link['LINKDT'])
        self.link['LINKENDDT'] = pd.to_datetime(self.link['LINKENDDT'])
         # Convert cusip to 8 digits and ensure it's properly formatted
        self.link = self.convert_cusip_to_8_digits(self.link)       
        # Sort and keep only the most recent link for each gvkey-cusip pair
        self.link.sort_values(by=['gvkey', 'cusip', 'LINKDT'], ascending=[True, True, False], inplace=True)
        self.link = self.link[['gvkey', 'cusip']].drop_duplicates(subset=['cusip'], keep='first')
                
        # Ensure gvkey is integer
        self.link['gvkey'] = pd.to_numeric(self.link['gvkey'], errors='coerce')
        self.link = self.link.dropna(subset=['gvkey'])
        self.link['gvkey'] = self.link['gvkey'].astype('int64')
        return self.link.sort_values(by=['cusip'])

    def process_with_validation(self):
        """Process data with input and output validation"""
        try:
            # Get the link table
            link = self.link
            if link is None:
                self.logger.error("Link table not initialized")
                return None

            # Get the FAFA data
            df = self.FAFA
            if df is None:
                df = self.load_data(self.FAFA_path)
                if df is None:
                    self.logger.error("Failed to load FAFA data")
                    return None

            self.logger.info(f"Processing {len(df):,} records...")
            df = pd.merge(df, link, on=['cusip'], how='left', indicator=True)
            self.logger.info(f"value counts of _merge: {df['_merge'].value_counts()}")
            self.logger.info(f"Merge rate: {self.compute_merge_success_rate(df):.2%}")
            result = self.process_financial_analyst_forecast_accuracy(df)
            
            if result is not None:
                self.logger.info(f"Processed {len(result):,} records")
                self.logger.info(f"Final shape: {result.shape}")
                self.logger.info(f"Final columns: {result.columns.tolist()}")
            
            return result

        except Exception as e:
            self.logger.error(f"Error in process_with_validation: {str(e)}")
            self.logger.error("Traceback:", exc_info=True)
            return None

    def process_financial_analyst_forecast_accuracy(self, df):
        """
        Process financial analyst forecast accuracy data efficiently.
        Args:
            df: DataFrame with required columns ["cusip", "gvkey", "analys", "anndats", "fpedats", 
                analyst_following, Accuracy_mean, Accuracy_first, Accuracy_last, Accuracy_most]
        """
        # try:
        # 1. Convert dates and sort
        for col in ['anndats', 'fpedats']:
            df[col] = pd.to_datetime(df[col])
            # Drop rows with missing group keys
        df = df.dropna(subset=['gvkey', 'analys'])
        # Create year and quarter columns from anndats
        df['year'] = df['anndats'].dt.year
        df['fqtr'] = df['anndats'].dt.quarter
        
        df = df.sort_values(['gvkey', 'analys', 'anndats'])
        
        # 2. Rolling window calculations
        metrics = ['analyst_following', 'Accuracy_mean', 'Accuracy_first', 
                    'Accuracy_last', 'Accuracy_most']
        
        # Compute rolling aggregates per gvkey, analys, year, and quarter.
        # Note: An analyst may have forecasts for different quarters in the same firm/year,
        # so we group by those keys.
        rolled = (df.groupby(['gvkey', 'analys', 'year', 'fqtr'])
                    .rolling(window=8, min_periods=1, on='anndats')
                    .agg({
                        'analyst_following': 'mean',
                        'Accuracy_mean': lambda x: x.abs().mean(),
                        'Accuracy_first': lambda x: x.iloc[0],  # first value in the window
                        'Accuracy_last': lambda x: x.iloc[-1],  # last value in the window
                        'Accuracy_most': lambda x: x.abs().min()
                    })
                    .reset_index())
        
        # 3. Sort by date so that shifting works correctly
        rolled = rolled.sort_values('anndats')
        
        # 4. Shift values by one observation (assuming one quarter per row)
        #    This assumes that within each firm/analyst group, each row represents one quarter.
        rolled[metrics] = (
            rolled.groupby(['gvkey', 'analys'])[metrics].shift(1)
        )
        
        # 5. Convert the quarterly data to annual data by aggregating (using mean for some metrics)
        result = rolled.groupby(['gvkey', 'analys', 'year']).agg({
            'analyst_following': 'last',
            'Accuracy_mean': 'mean',
            'Accuracy_first': 'first',
            'Accuracy_last': 'last',
            'Accuracy_most': 'min'
        }).reset_index()
        
        # Rename columns for consistency
        result = result.rename(columns={
            'analys': 'amaskcd',
            'Accuracy_mean': 'accuracy_mean',
            'Accuracy_first': 'accuracy_first',
            'Accuracy_last': 'accuracy_last'
        })
        
        return result.sort_values(['gvkey', 'amaskcd', 'year'])
        
        
    def process_all_star(self):
        """Process all star data"""
        self.logger.info("Processing all_star data")
        self.logger.info(f"Original columns: {self.all_star.columns.tolist()}")
        
        # Rename columns if needed
        self.all_star.columns = ['year', 'emaskcd', 'amaskcd']
        
        # Ensure numeric types
        for col in ['year', 'emaskcd', 'amaskcd']:
            self.all_star[col] = pd.to_numeric(self.all_star[col], errors='coerce')
        
        # Add all_star indicator
        self.all_star['all_star'] = 1
        
        self.logger.info(f"Processed all_star shape: {self.all_star.shape}")
        self.logger.info(f"Processed all_star columns: {self.all_star.columns.tolist()}")
        
        return self.all_star
    
    def process_analyst_exp(self):
        """Process analyst experience data"""
        # self.analyst_exp = self.optimize_dtypes(self.analyst_exp)
        # drop column gvkey and drop duplicates 
        print(self.analyst_exp.columns)
        self.analyst_exp = self.analyst_exp.drop(columns=['gvkey'])
        self.analyst_exp = self.analyst_exp.drop_duplicates().reset_index(drop=True)
        # rename column names
        self.analyst_exp.rename(columns={'fyear': 'year', 'AMASKCD': 'amaskcd', 'EMASKCD': 'emaskcd'}, inplace=True)
        # convert the year, amaskcd, emaskcd to numeric
        merge_columns = ['year', 'amaskcd', 'emaskcd']
        for col in merge_columns:
            self.analyst_exp[col] = pd.to_numeric(self.analyst_exp[col], errors='coerce')
        # drop duplicates in year, amaskcd, emaskcd
        self.analyst_exp = self.analyst_exp.drop_duplicates(subset = merge_columns).reset_index(drop=True)
        return self.analyst_exp

    def save_chunk(self, chunk, first_chunk=False):
        """
        Save chunk to CSV file
        
        Parameters:
        -----------
        chunk : pandas DataFrame
            The chunk to save
        output_path : str
            Path to save the CSV file
        first_chunk : bool
            Whether this is the first chunk (to write headers)
        """
        try:
            chunk.to_csv(self.output_path, 
                        mode='w' if first_chunk else 'a',
                        index=False, 
                        header=first_chunk)
        except Exception as e:
            self.logger.error(f"Error saving chunk: {str(e)}")
            self.logger.error("Traceback:", exc_info=True)

    def compute_merge_success_rate(self, chunk):
        """
        Compute merge success rate
        
        Parameters:
        -----------
        chunk : pandas DataFrame
            DataFrame containing the merge results
            
        Returns:
        --------
        float
            Merge success rate (0.0 to 1.0)
        """
        return chunk['_merge'].value_counts().get('both', 0) / len(chunk)
    
    def validate_merge(self, chunk, original_len, merge_name):
        """Validate merge didn't create unexpected duplicates"""
        new_len = len(chunk)
        if new_len > original_len:
            self.logger.warning ("*******************************************************")
            self.logger.warning(f"* \n *WARNING: {merge_name} merge increased rows from {original_len:,} to {new_len:,}*")
            self.logger.warning(f"* Increase: {new_len - original_len:,} rows ({((new_len/original_len)-1)*100:.2f}%)*")
            self.logger.warning("*******************************************************")
        return new_len

    def analyze_merge_keys_before_merge(self, df1, df2, keys, name1="chunk", name2="FAFA"):
        """Analyze merge keys before performing merge"""
        self.logger.info(f"\n{'='*20} Merge Analysis {'='*20}")
        
        for key in keys:
            self.logger.info(f"\nAnalyzing key: {key}")
            # Check data types
            self.logger.info(f"{name1} {key} dtype: {df1[key].dtype}")
            self.logger.info(f"{name2} {key} dtype: {df2[key].dtype}")
            
            # Check value ranges
            self.logger.info(f"{name1} {key} range: {df1[key].min()} to {df1[key].max()}")
            self.logger.info(f"{name2} {key} range: {df2[key].min()} to {df2[key].max()}")
            
            # Check for nulls
            self.logger.info(f"{name1} {key} null count: {df1[key].isnull().sum()}")
            self.logger.info(f"{name2} {key} null count: {df2[key].isnull().sum()}")
            
            # Show sample of non-matching values
            df1_vals = set(df1[key].unique())
            df2_vals = set(df2[key].unique())
            non_matching = df1_vals - df2_vals
            if len(non_matching) > 0:
                self.logger.info(f"Sample of values in {name1} not in {name2}: {list(non_matching)[:5]}")

    def process_large_data(self, chunk_size=100000):
        """Process and merge large datasets in chunks"""
        try:
            if self.proid_unique is None:
                self.load_and_prepare_reference_data()
            
            # Initialize counters
            total_rows = 0
            first_chunk = True
            
            # Process link table before using it
            self.link = self.process_link_table()
            
            # Process reference data with checks
            self.analyst_exp = self.process_analyst_exp()
            if self.analyst_exp is None:
                self.logger.error("Analyst experience processing failed")
                return None
            
            self.FAFA = self.process_with_validation()
            if self.FAFA is None:
                self.logger.error("FAFA processing failed")
                return None
            
            self.logger.info(f"FAFA processed successfully. Shape: {self.FAFA.shape}")
            self.logger.info(f"FAFA columns: {self.FAFA.columns.tolist()}")
            
            # Verify FAFA has required columns before proceeding
            required_cols = ['gvkey', 'year', 'amaskcd']
            if not all(col in self.FAFA.columns for col in required_cols):
                self.logger.error(f"FAFA missing required columns. Has: {self.FAFA.columns.tolist()}")
                return None
            
            self.all_star = self.process_all_star()
            
            self.logger.info(f"Processing file: {os.path.basename(self.input_path)}")
            
            # Modify the CSV reader configuration
            csv_reader = pd.read_csv(
                self.input_path, 
                sep='\t', 
                chunksize=chunk_size,
                dtype={'date': str},
                low_memory=False
            )
            
            # Process chunks
            for chunk_num, chunk in enumerate(tqdm(csv_reader, desc="Processing chunks", colour="green")):
                try:
                    self.logger.info(f"***************************************Processing chunk {chunk_num + 1}***************************************")
                    
                    # Convert date column after reading
                    try:
                        # First try parsing with standard format
                        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
                        
                        # If most dates are NaT, try alternative format
                        if chunk['date'].isna().sum() > len(chunk) * 0.9:
                            self.logger.warning("Standard date parsing failed, trying alternative format...")
                            chunk['date'] = pd.to_datetime(chunk['date'], format='%b %d %Y', errors='coerce')
                    except Exception as e:
                        self.logger.error(f"Error parsing dates: {str(e)}")
                        self.logger.error("Date column sample:")
                        self.logger.error(chunk['date'].head())
                    
                    # Log date parsing results
                    self.logger.info(f"Date parsing sample results: {chunk['date'].head()}")
                    self.logger.info(f"Number of valid dates: {chunk['date'].notna().sum()}")
                    self.logger.info(f"Number of invalid dates: {chunk['date'].isna().sum()}")
                    
                    self.logger.info(f"{self.color_text('Main data sample description')}")
                    self.logger.info(f"chunk description: {chunk.describe()}")
                    chunk['proid'] = pd.to_numeric(chunk['proid'], errors='coerce').fillna(-999).astype(int)
                    self.logger.info(f"Chunk size before merges: {chunk.shape}")
                    
                    # Track original length before merges
                    original_len = len(chunk)
                    
                    # First merge with proid_unique
                    self.logger.info("Merging with proid_unique")
                    self.logger.info(f"proid_unique columns: {self.proid_unique.columns.tolist()}")
                    chunk = pd.merge(chunk, self.proid_unique, on='proid', how='left', indicator=True)
                    new_len = self.validate_merge(chunk, original_len, "proid_unique")
                    self.logger.info(f"proid_unique merge rate: {self.compute_merge_success_rate(chunk):.2%} and after merge data length: {new_len}")
                    
                    # Log merge results
                    if '_merge' in chunk.columns:
                        merge_rate = self.compute_merge_success_rate(chunk)
                        if merge_rate < 0.9:
                            self.logger.warning(f"Low proid_unique merge rate: {merge_rate:.2%}")
                        chunk.drop(columns=['_merge'], inplace=True)
                    
                    # Merge with Allstar
                    self.logger.info("3rd merge on all star")
                    original_len = len(chunk)
                    if all(col in chunk.columns for col in ['year', 'emaskcd', 'amaskcd']):
                        chunk = pd.merge(
                            chunk,
                            self.all_star,
                            on=['year', 'emaskcd', 'amaskcd'],
                            how='left',
                            indicator=True
                        )
                        new_len = self.validate_merge(chunk, original_len, "all_star")
                        
                        # Fill missing values in emaskcd and amaskcd
                        chunk['emaskcd'] = chunk['emaskcd'].fillna(-999).astype(int)
                        chunk['amaskcd'] = chunk['amaskcd'].fillna(-998).astype(int)
                        
                        # Merge with all_star if we have the required columns
                        if '_merge' in chunk.columns:
                            chunk['all_star'] = chunk['_merge'].map({'both': 1, 'left_only': 0, 'right_only': 0})
                            merge_rate = self.compute_merge_success_rate(chunk)
                            if merge_rate < 0.9:
                                self.logger.warning(f"Low all_star merge rate: {merge_rate:.2%}")
                            self.logger.info(f"value counts of _merge: {chunk['_merge'].value_counts('_merge')}")
                            chunk.drop(columns=['_merge'], inplace=True)
                    else:
                        self.logger.warning("Skipping all_star merge due to empty all_star data")
                        chunk['all_star'] = 0
                    
                    # Merge with analyst_exp
                    self.logger.info("4th merge on analyst_exp")
                    if not self.analyst_exp.empty:
                        original_len = len(chunk)
                        chunk = pd.merge(
                            chunk,
                            self.analyst_exp,
                            on=['year', 'amaskcd', 'emaskcd'],
                            how='left',
                            indicator=True
                        )
                        new_len = self.validate_merge(chunk, original_len, "analyst_exp")
                        
                        if '_merge' in chunk.columns:
                            merge_rate = self.compute_merge_success_rate(chunk)
                            if merge_rate < 0.9:
                                self.logger.warning(f"Low analyst_exp merge rate: {merge_rate:.2%}")
                            self.logger.info(f"value counts of _merge: {chunk['_merge'].value_counts('_merge')}")                            
                            chunk.drop(columns=['_merge'], inplace=True)
                    else:
                        self.logger.warning("Skipping analyst_exp merge due to empty analyst_exp data")
                    
                    # Before the merge
                    self.logger.info("\nPreparing for FAFA merge...")
                    merge_keys = ['gvkey', 'year', 'amaskcd']

                    # Analyze before type conversion
                    self.analyze_merge_keys_before_merge(chunk, self.FAFA, merge_keys)

                    # Ensure consistent types for merge keys
                    for col in merge_keys:
                        # Convert both DataFrames to same type
                        if col in chunk.columns and col in self.FAFA.columns:
                            # Convert to numeric first to handle any string values
                            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                            self.FAFA[col] = pd.to_numeric(self.FAFA[col], errors='coerce')
                            
                            # Fill NA values with a sentinel value
                            chunk[col] = chunk[col].fillna(-999).astype('int64')
                            self.FAFA[col] = self.FAFA[col].fillna(-999).astype('int64')

                    # Analyze after type conversion
                    self.logger.info("\nAfter type conversion:")
                    self.analyze_merge_keys_before_merge(chunk, self.FAFA, merge_keys)

                    # Check for any matching keys
                    chunk_keys = set(chunk[merge_keys].apply(tuple, axis=1))
                    fafa_keys = set(self.FAFA[merge_keys].apply(tuple, axis=1))
                    common_keys = chunk_keys.intersection(fafa_keys)
                    self.logger.info(f"\nMerge key analysis:")
                    self.logger.info(f"Unique keys in chunk: {len(chunk_keys)}")
                    self.logger.info(f"Unique keys in FAFA: {len(fafa_keys)}")
                    self.logger.info(f"Common keys: {len(common_keys)}")

                    # Perform the merge
                    self.logger.info("\nPerforming FAFA merge...")
                    chunk = pd.merge(
                        chunk,
                        self.FAFA,
                        on=merge_keys,
                        how='left',
                        indicator=True,
                        suffixes=('', '_y')
                    )

                    # Analyze merge results
                    merge_stats = chunk['_merge'].value_counts()
                    self.logger.info(f"\nMerge results:")
                    self.logger.info(f"Total rows: {len(chunk)}")
                    self.logger.info(f"Merge statistics:\n{merge_stats}")
                    self.logger.info(f"Merge rate: {self.compute_merge_success_rate(chunk):.2%}")

                    # Sample of unmatched rows
                    if 'left_only' in merge_stats:
                        unmatched = chunk[chunk['_merge'] == 'left_only'][merge_keys].head()
                        self.logger.info(f"\nSample of unmatched rows:\n{unmatched}")
                    
                    # After the merge, ensure the date column is properly handled
                    if 'date_y' in chunk.columns:  # If there was a date column from both sides
                        chunk.drop(columns=['date_y'], inplace=True)
                    
                    # Add these logging statements in process_large_data
                    self.logger.info(f"Date column present in FAFA: {'date' in self.FAFA.columns}")
                    self.logger.info(f"Chunk columns after merge: {chunk.columns.tolist()}")
                    self.logger.info(f"Merge rate: {self.compute_merge_success_rate(chunk):.2%}")
                    # Save chunk and update total_rows
                    self.save_chunk(chunk, first_chunk)
                    first_chunk = False
                    total_rows += len(chunk)
                    
                    # Clear memory
                    del chunk
                    gc.collect()
                    
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_num + 1}: {str(e)}")
                    self.logger.error("Traceback:", exc_info=True)
                    continue
            
            self.logger.info(f"Processing completed. Total rows processed: {total_rows}")
            self.logger.info(f"Output saved to: {self.output_path}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in process_large_data: {str(e)}")
            self.logger.error("Traceback:", exc_info=True)
            return None

    def run(self, chunk_size=100000):
        """Main execution method"""
        try:
            self.logger.info("Starting merge process")

            # Process the data
            self.process_large_data(chunk_size)
            
            if self.output_path is not None:
                self.logger.info(f"Data successfully processed and saved to {self.output_path}")
            else:
                self.logger.error("Failed to process data")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in processing: {str(e)}")
            self.logger.error("Traceback:", exc_info=True)
            return None

if __name__ == "__main__":
    # check if the output path exist, create the folder if not
    if not os.path.exists(os.path.join(os.getcwd(), "outputs", "analyst_profile")):
        os.makedirs(os.path.join(os.getcwd(), "outputs", "analyst_profile"))    
    output_path = os.path.join(os.getcwd(), "outputs", "analyst_profile", "id2firms_anlys.csv")
    
    # Use the actual filename
    input_filename = "id2firm_2006_2020.txt"  # or "id2firm_2021_2024.txt"
    processor = MAP(output_path, input_filename)
    chunk_size = 1000000
    processor.run(chunk_size)


