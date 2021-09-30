import numpy as np

def pad_flipbook(flipbook, size):
    assert flipbook.ndim == 3
    
    h, w = flipbook.shape[1:]
    assert size[0] >= h and size[1] >= w
    
    ph, pw = size[0] - h, size[1] - w
    
    flipbook = np.pad(flipbook, ((0, 0), (0, ph), (0, pw)))
    assert flipbook.shape[1] == size[0]
    assert flipbook.shape[2] == size[1]
    
    return flipbook
    
    
def unpad_flipbook():
    raise NotImplementedError