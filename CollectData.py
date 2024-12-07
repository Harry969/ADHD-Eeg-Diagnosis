import pathlib
import numpy as np

import pathlib
import numpy as np

class CollectData:
    
    def __init__(self):
        """
        Initialize the paths to different data directories.
        Each path is hardcoded for raw, filtered, ASR-filtered, and ICA-filtered data.
        """
        # Directory for raw data files
        self.raw_dir = "DATA/Raw"
        # Directory for filtered data files
        self.filter_dir = "DATA/Filtered"
        # Directory for ASR-filtered data files
        self.filter_ASR_dir = "DATA/ASR"
        # Directory for ICA-filtered data files
        self.ICA_dir = "DATA/ICA"

    def read_file(self, dir):
        """
        Reads all CSV files from the provided directory and loads them into a list.
        
        Args:
            dir (str): The directory containing CSV files to be read.
        
        Returns:
            tuple: A tuple containing:
                - list of numpy arrays (one per file/subject), where each array represents the data.
                - array of channel names (from the first row of the first file/subject).
        """
        # Create a Path object for the provided directory
        current_dir = pathlib.Path(dir)
        
        # List to store the data arrays read from the files
        returned_list = []
        col_names = None
        
        # Iterate over each file in the directory
        for file in current_dir.iterdir():
            # Only process files with a '.csv' extension
            if file.is_file() and file.suffix == '.csv':  
                # Load the CSV file as a numpy array (with dtype as string initially)
                arr = np.loadtxt(file, delimiter=',', dtype=str)
                
                # Extract the data part (excluding the first row, which contains channel names)
                data_arr = arr[1:, :].astype('float64')  # Convert data to float64
                
                # Extract the column/channel names from the first row (only once)
                if col_names is None:  
                    col_names = arr[0, :]
                
                # Append the data array to the list
                returned_list.append(data_arr)
        
        return returned_list, col_names
    
    def load_data(self):
        """
        Loads raw, filtered, ASR-filtered, and ICA-filtered data into instance variables.
        """
        # Load raw data and store the channel names
        self.list_raw, self.chan_names = self.read_file(self.raw_dir)
        # Load filtered data
        self.list_filtered, _ = self.read_file(self.filter_dir)
        # Load ASR-filtered data
        self.list_asr, _ = self.read_file(self.filter_ASR_dir)
        # Load ICA-filtered data
        self.list_ica, _ = self.read_file(self.ICA_dir)
