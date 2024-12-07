import math

class CreateChunks:
    """
    This class processes EEG data by splitting it into chunks.
    It returns a list of lists where each sublist represents a chunked version of the data
    across all individuals.
    """

    def __init__(self):
        pass

    def cut_eeg(self, data, n_chunks):
        """
        Splits the EEG data into `n_chunks` chunks for each individual.
        
        Args:
            data (list of np.arrays): List of individuals' EEG data. Each individual's data is a 2D array (time x channels).
            n_chunks (int): Number of chunks to split each individual's data into.
        
        Returns:
            list of lists: A list containing `n_chunks` lists, where each sublist contains 
                           the chunked data of all individuals.
        """
        # List to store the chunked data
        lista_chunks = []
        
        # Iterate over the number of chunks
        for chunk in range(n_chunks):
            lista_indiv = []  # List to store chunked data for all individuals
            
            # Iterate over each individual's data
            for indiv in data:
                n_time = indiv.shape[0]  # Total time steps in the individual's data

                # Ensure there are enough time points to split into chunks
                if n_chunks > n_time:
                    raise ValueError(f"Cannot split {n_time} time steps into {n_chunks} chunks.")

                # Time length for each chunk
                time_per_chunk = math.floor(n_time / n_chunks)  

                # Extract the specific chunk of data for the individual
                chnk_indv = indiv[time_per_chunk * chunk : time_per_chunk * (chunk + 1)]
                lista_indiv.append(chnk_indv)  # Add the chunk to the individual's chunk list
            
            lista_chunks.append(lista_indiv)  # Add the chunked data for all individuals
        
        return lista_chunks

    def get_more_chunks(self, data, list_n_chunks=None):
        """
        Splits the EEG data into multiple chunk configurations based on a predefined or custom list of chunk sizes.
        
        Args:
            data (list of np.arrays): List of individuals' EEG data.
            list_n_chunks (list of int, optional): A list of chunk sizes to apply. Defaults to [1, 2, 3, 4, 5, 20].
        
        Returns:
            list of lists: A list of different chunk configurations. Each element is a list of chunked EEG data
                           where the data is split into different numbers of chunks.
        """
        # Default chunk sizes if none provided
        if list_n_chunks is None:
            list_n_chunks = [1, 2, 3, 4, 5, 20]  

        lista_final = []  # List to store chunked data for each configuration
        
        # Iterate over different chunk sizes and split the data accordingly
        for n_chunks in list_n_chunks:
            cut_eeg_data = self.cut_eeg(data, n_chunks)  # Split data into `n_chunks`
            lista_final.append(cut_eeg_data)  # Append the chunked data to the final list
        
        return lista_final
