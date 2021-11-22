import numpy as np
import networkx as nx
from skimage import measure
from itertools import combinations

def overlap_labels(masks):
    gt = measure.label(gt)
    new_labels = measure.label(new_labels)
    
    n_gt = len(np.unique(gt))
    n_new = len(np.unique(new_labels))
    hist_array = np.histogram2d(gt.ravel(), new_labels.ravel(), bins=(range(n_gt + 1), range(n_new + 1)))[0]
    gt_new_labels = np.where(hist_array.sum(1) > 0)[0]
    
    out = np.zeros_like(gt)
    for l in gt_new_labels:
        if l > 0:
            out += gt == l
        
    return out > 0

def remove_labels(gt, remove):
    gt = measure.label(gt)
    remove = measure.label(remove)
    
    n_gt = len(np.unique(gt))
    n_remove = len(np.unique(remove))
    if n_remove > 1:
        hist_array = np.histogram2d(gt.ravel(), remove.ravel(), bins=(range(n_gt + 1), range(n_remove + 1)))[0]
        in_back = hist_array[:, 0]
        in_fore = hist_array[:, 1:].sum(1)
        gt_remove = np.where(np.logical_and(in_fore > in_back, in_fore > 0))[0]

        out = np.zeros_like(gt)
        for l in np.unique(gt):
            if l > 0:
                if l not in gt_remove:
                    out += gt == l
    else:
        out = gt
        
    return out > 0

def box_area(boxes):
    """
    Calculates the area of an array of boxes
    
    Arguments:
    ----------
    boxes: Array of shape (n, 4). Where coordinates
    are (y1, x1, y2, x2).
    
    Returns:
    --------
    box_areas. Array of shape (n,).
    
    """
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    return height * width

def pairwise_box_intersection(boxes):
    """
    Calculates the pairwise overlaps with a set of
    bounding boxes.
    
    Arguments:
    ----------
    boxes: Array of shape (n, 4). Where coordinates
    are (y1, x1, y2, x2).
    
    Returns:
    --------
    box_overlaps. Array of shape (n, n).
    
    """
    # separate boxes into coordinates arrays
    [y_min, x_min, y_max, x_max] = np.split(boxes, 4, axis=1)

    # find top and bottom coordinates of overlapping area
    all_pairs_min_ymax = np.minimum(y_max, np.transpose(y_max))
    all_pairs_max_ymin = np.maximum(y_min, np.transpose(y_min))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape), 
        all_pairs_min_ymax - all_pairs_max_ymin
    )
    
    # find left and right coordinates of the overlapping area
    all_pairs_min_xmax = np.minimum(x_max, np.transpose(x_max))
    all_pairs_max_xmin = np.maximum(x_min, np.transpose(x_min))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape), 
        all_pairs_min_xmax - all_pairs_max_xmin
    )
    
    return intersect_heights * intersect_widths

def pairwise_box_iou(boxes):
    """
    Calculates the pairwise intersection-over-union 
    within a set of bounding boxes.
    
    Arguments:
    ----------
    boxes: Array of shape (n, 4). Where coordinates
    are (y1, x1, y2, x2).
    
    Returns:
    --------
    box_ious. Array of shape (n, n).
    
    """
    intersect = pairwise_box_intersection(boxes) # (n, n)
    
    # union is the difference between the sum of 
    # areas and the intersection
    area = box_area(boxes)
    pairwise_area = area[:, None] + area[None, :] # (n, n)
    union = pairwise_area - intersect
    
    return intersect / union

def merge_boxes(box1, box2):
    """
    Merge boxes into a single large box.
    """
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    return np.array([
        min(ymin1, ymin2), 
        min(xmin1, xmin2), 
        max(ymax1, ymax2), 
        max(xmax1, xmax2)
    ])

def crop_and_binarize(mask, box, label):
    """
    Crop and binarize a mask
    """
    ymin, xmin, ymax, xmax = box
    return mask[ymin:ymax, xmin:xmax] == label

def mask_iou(mask1, mask2):
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    union = np.count_nonzero(np.logical_or(mask1, mask2))
    return (intersection + 1) / (union + 1)

def mask_aggregation(masks, overlap_thr=0.1):
    # consensus segmentation generation
    # generate bounding boxes for all instances
    mask_indices = []
    mask_labels = []
    detection_boxes = []
    for i, mask in enumerate(masks):
        rps = measure.regionprops(mask)
        mask_indices.extend([i] * len(rps))
        mask_labels.extend([rp.label for rp in rps])
        detection_boxes.extend([rp.bbox for rp in rps])
        
    # return mask of all zeros if no detections
    if not detection_boxes:
        return [np.zeros_like(masks[0])]
        
    mask_indices = np.array(mask_indices)
    mask_labels = np.array(mask_labels)
    detection_boxes = np.array(detection_boxes)
    n_detections = len(detection_boxes)
    
    # calculate ious between pairs of boxes
    # and return indices of matching pairs
    box_matches = np.array(pairwise_box_iou(detection_boxes).nonzero()).T
    
    # filter out boxes from the same annotator
    r1_match_ann = mask_indices[box_matches[:, 0]]
    r2_match_ann = mask_indices[box_matches[:, 1]]
    box_matches = box_matches[r1_match_ann != r2_match_ann]
    
    # remove duplicates (because order of items in pair doesn't matter)
    box_matches = np.sort(box_matches, axis=-1)
    box_matches = np.unique(box_matches, axis=0)
    
    graph = nx.Graph()
    for node_id in range(len(mask_labels)):
        graph.add_node(node_id)
        
    # iou to weighted edges
    for r1, r2 in zip(box_matches[:, 0], box_matches[:, 1]):
        # determine instance labels by mask
        mask1 = masks[mask_indices[r1]]
        l1 = mask_labels[r1]
        box1 = detection_boxes[r1]
            
        mask2 = masks[mask_indices[r2]]
        l2 = mask_labels[r2]
        box2 = detection_boxes[r2]
            
        # large enclosing box for both instances
        box = merge_boxes(box1, box2)
        mask1 = crop_and_binarize(mask1, box, l1)
        mask2 = crop_and_binarize(mask2, box, l2)
            
        pair_iou = mask_iou(mask1, mask2)
        if pair_iou >= overlap_thr:
            graph.add_edge(r1, r2, iou=pair_iou)
            
    # return separated masks with confidence
    # maps for each
    instance_scores = []
    for comp in nx.connected_components(graph):
        comp = list(comp)
        
        instance = np.zeros_like(masks[0]).astype(np.float32)
        # add all masks in the group together
        # to get a confidence mask for each
        for r in comp:
            mask = masks[mask_indices[r]]
            label = mask_labels[r]
            instance += (mask == label).astype(np.float32) / len(masks)
            
        instance_scores.append(instance)
            
    return instance_scores
            

def aggregated_instance_segmentation(aggregated_masks, vote_thr=0.5):
    """
    Merges a list of masks into an instance segmentation.
    
    Arguments:
    ----------
    aggregated_masks: List of n (h, w) confidence aggregated masks.
    E.g. output from mask_aggregation function.
    
    vote_thr: Integer. Threshold number of votes over which
    to mark a pixel as part of a segmentation. Default 0.4.
    
    Returns:
    --------
    instance_segmentation: Array of (h, w) with each detected mitochondrion
    given a different label.
    
    """
    mask_shape = aggregated_masks[0].shape
    instance_segmentation = np.zeros(mask_shape, dtype=np.uint8)
    
    # add each detection with an new label
    label = 1
    for mask in aggregated_masks:
        # threshold the mask
        mask = mask >= vote_thr
        
        # relabel in case new connected
        # components fall out
        mask = measure.label(mask).astype(np.uint8)
        
        # number and values of new labels
        # excluding background value of 0
        mask_labels = np.unique(mask)[1:]
        
        for ml in mask_labels:
            detection_seg = ((mask == ml) * label).astype(instance_segmentation.dtype)
            instance_segmentation += detection_seg
            label += 1
            
            # will probably never have more
            # than 255 instances for small images
            # like these, but let's avoid any headaches
            if label >= 255:
                raise Exception('Unsigned 8-bit is invalid for this mask!')
        
    return instance_segmentation
    