import numpy as np
import mne

class preprocessor():
    def __init__(self,fs=120,band_pass=[2.0,30.0],tmin=0.0,tmax=31.5):
        # Configuration for the preprocessing
        self.fs = 120  # target sampling frequency in Hz for resampling
        self.band_pass = [2.0, 30.0]  # bandpass cutoffs in Hz for spectral filtering
        self.tmin = 0.0  # trial onset in seconds for slicing
        self.tmax = 31.5  # trial end in seconds for slicing
    
        self.FR = 60
           
    def process(self,inp):  #Inp has shape channels * measurements * trials  = (1,3780,N)
        
        """
        I should instead be doing only the spectral band-pass and downsampling pbb
        Pbb loop over the first and last dimension of the simulated data? 
        Then band-pass and downsample
        """
        mne.set_log_level(verbose=False)
        outp = np.zeros(inp.shape)
        
        for i1 in range(inp.shape[0]):
            for i2 in range(inp.shape[2]):
                current = inp[i1,:,i2]
                current = mne.filter.filter_data(current,sfreq=self.fs,l_freq=self.band_pass[0],h_freq=self.band_pass[1])
                outp[i1,:,i2] = current
        return outp