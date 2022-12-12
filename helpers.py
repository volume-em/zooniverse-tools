import numpy as np
from skimage import draw

__all__ = [
    'pad_flipbook',
    'pad_image',
    'polygon_to_array',
    'poly2segmentation'
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
    
def polygon_to_array(polygon):
    """
    Converts zooniverse annotation polygon
    into an array of x,y coordinates of shape
    (n_vertices, 2).
    
    """
    # empty array to fill
    object_array = np.zeros((len(polygon), 2), dtype='float')
    
    # each vertex is a dict with x, y
    for i, vertex in enumerate(polygon):
        object_array[i, 0] = vertex['x']
        object_array[i, 1] = vertex['y']
        
    return object_array

def poly2mask(vertices, shape):
    """
    Converts a list of vertices created by polygon_to_array
    into a segmentation mask.
    
    Arguments:
    ----------
    vertices: Array of (n_vertices, 2).
    
    shape: Tuple of (height, width). Shape of the segmentation mask 
    (should match the shape of the annotated image).
    
    Returns:
    --------
    mask: Binary segmentation mask of shape (height, width).
    
    """
    y, x = vertices[:, 0], vertices[:, 1]
    fill_row_coords, fill_col_coords = draw.polygon(x, y, shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[fill_row_coords, fill_col_coords] = 1
    
    return mask

def poly2segmentation(vertex_set, shape):
    """
    Converts a list of vertices created by polygon_to_array
    into a segmentation mask.
    
    Arguments:
    ----------
    vertices: Array of (n_vertices, 2).
    
    shape: Tuple of (height, width). Shape of the segmentation mask 
    (should match the shape of the annotated image).
    
    Returns:
    --------
    mask: Binary segmentation mask of shape (height, width).
    
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for label, vertices in enumerate(vertex_set, 1):
        y, x = vertices[:, 0], vertices[:, 1]
        fill_row_coords, fill_col_coords = draw.polygon(x, y, shape)
        mask[fill_row_coords, fill_col_coords] = label
    
    return mask