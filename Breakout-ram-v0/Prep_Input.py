import numpy as np

class Prep_Input:
    
    # Return binary representation of the input
    def binary(x):

        batch_size = x.shape[0]
        x = x.reshape(-1)
        bin_val = np.unpackbits(x.astype(np.uint8)).reshape((batch_size, -1))
        '''
        batch_size = x.shape[0]
        x = x.reshape(-1)
        bin_val = np.unpackbits(x.astype(np.uint8)).reshape((batch_size,-1, 1))
        '''
        return bin_val
    
    # Return the input itself
    def identity(x):
        return x
