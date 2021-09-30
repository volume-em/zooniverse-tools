import numpy as np
from skimage import draw

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

def iou(gt, pred):
    """
    Calculates IoU between ground truth (gt)
    and user generated annotation (pred).
    """
    # binarize
    gt = gt > 0
    pred = pred > 0
    
    gt_max = gt.max()
    pred_max = pred.max()
    
    if gt_max == 0 and pred_max == 0:
        return 1
    elif gt_max > 0 and pred_max == 0:
        return 0
    elif gt_max == 0 and pred_max > 0:
        return 0
    else:
        intersection = np.count_nonzero(np.logical_and(gt, pred))
        union = np.count_nonzero(np.logical_or(gt, pred))
        
        if union == 0:
            return 1
        else:
            return intersection / union

def average_precision(gt_mask, pred_mask, thr=0.75, return_counts=False):
    """
    Calculates the average precision of a user generated annotation (pred_mask)
    against expert ground truth (gt_mask).
    
    Arguments:
    ----------
    gt_mask: (n, m) array where each mito is a different label
    as created by an expert.
    
    pred_mask: (n, m) array where each mito is a different label
    as created by a user.
    
    thr: Float. Threshold at which to consider a detection a true
    positive. Default 0.75 for measuring AP@75.
    
    return_counts: Bool. If True, return count of TP, FP, FN detections.
    Default False.
    
    """
    # number of objects in each
    n_gt = len(np.unique(gt_mask))
    n_pred = len(np.unique(pred_mask))
    
    if n_gt == 1 and n_pred == 1:  # only background
        ap = 1
        tp, fp, fn = 0, 0, 0
        output = (ap,)
        if return_counts:
            output = (ap, tp, fp, fn)
        return output
    elif n_gt > 1 and n_pred == 1:  # pred only background
        ap = 0
        tp, fp, fn = 0, 0, n_gt
        output = (ap,)
        if return_counts:
            output = (ap, tp, fp, fn)
        return output
    elif n_gt == 1 and n_pred > 1:  # gt only background 
        ap = 0
        tp, fp, fn = 0, n_pred, 0
        output = (ap,)
        if return_counts:
            output = (ap, tp, fp, fn)
        return output
    else: 
        # multiple instances to compare
        # histogram2d to calculate intersections
        intersections, _, _ = np.histogram2d(
            gt_mask.ravel(), pred_mask.ravel(), bins=(n_gt, n_pred)
        )
        
        # clip out background (label 0) to match instances
        intersections = intersections[1:, 1:]
        
        gt_counts = np.histogram(gt_mask.ravel(), bins=n_gt)[0][1:]
        pred_counts = np.histogram(pred_mask.ravel(), bins=n_pred)[0][1:]
        
        unions = gt_counts[:, None] + pred_counts[None, :]
        if np.sum(unions) == 0:
            if return_counts:
                return (1, 0, 0, 0)
            else:
                return (1,)
        else:
            ious = intersections / (unions - intersections)
        
        n_fp = np.count_nonzero(np.max(ious, axis=0) <= thr)
        n_fn = np.count_nonzero(np.max(ious, axis=1) <= thr)
        n_tp = np.count_nonzero(np.max(ious, axis=1) >= thr)
        ap = (n_tp) / (n_tp + n_fn + n_fp)
        
        output = (ap,)
        if return_counts:
            output = (ap, n_tp, n_fp, n_fn)
        return output