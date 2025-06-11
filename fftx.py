import numpy as np
class DummyFFT:
    def fftn(self, x):
        return np.fft.fftn(x)
    def ifftn(self, x):
        return np.fft.ifftn(x)
fft = DummyFFT()
