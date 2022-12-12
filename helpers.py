import numpy as np

__all__ = [
    'pad_flipbook',
    'pad_image'
]

def pad_flipbook(flipbook, size):
    """Zero pads a flipbook (h, w) dimensions only."""
    assert flipbook.ndim == 3
    
    h, w = flipbook.shape[1:]
    assert size[0] >= h and size[1] >= w
    
    ph, pw = size[0] - h, size[1] - w
    
    flipbook = np.pad(flipbook, ((0, 0), (0, ph), (0, pw)))
    assert flipbook.shape[1] == size[0]
    assert flipbook.shape[2] == size[1]
    
    return flipbook

def pad_image(image, size):
    """Zero pads an image (h, w) dimensions only."""
    assert image.ndim == 2
    
    h, w = image.shape
    assert size[0] >= h and size[1] >= w
    
    ph, pw = size[0] - h, size[1] - w
    
    image = np.pad(image, ((0, ph), (0, pw)))
    assert image.shape[0] == size[0]
    assert image.shape[1] == size[1]
    
    return image
    
    
def unpad_flipbook():
    raise NotImplementedError