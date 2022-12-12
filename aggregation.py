import numpy as np
import networkx as nx
from skimage import measure
from itertools import combinations

__all__ = [
    'mask_aggregation', 
    'aggregated_instance_segmentation'
]

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
    Crops and binarizes an instance mask
    around a particular label.
    """
    ymin, xmin, ymax, xmax = box
    return mask[ymin:ymax, xmin:xmax] == label

def mask_iou(mask1, mask2):
    """
    Computes the intersection-over-union between two 
    binary segmentation masks.
    """
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    union = np.count_nonzero(np.logical_or(mask1, mask2))
    return intersection / union

def mask_ioa(mask1, mask2):
    """
    Computes the intersection-over-area between two 
    binary segmentation masks.
    """
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    area = np.count_nonzero(mask1)
    return intersection / area

def calculate_clique_ious(G, clique1, clique2):
    """
    Computes the average IoU between all instances in clique1
    and clique 2.
    """
    all_ious = []
    for node1 in clique1:
        for node2 in clique2:
            if G.has_edge(node1, node2):
                all_ious.append(G[node1][node2]['iou'])
            else:
                # iou too small to have an edge, so it's 0
                all_ious.append(0.)

    return sum(all_ious) / len(all_ious)

def create_clique_graph(G, iou_threshold, min_clique_iou=0.1):
    """
    Creates a graph where each node represents a clique
    of overlapping objects in instance segmentations of
    the same image.
    """
    # get a list of edges to drop from the graph
    drop_edges = []
    for (u, v, d) in G.edges(data=True):
        if d['iou'] < iou_threshold:
            drop_edges.append((u, v))

    # create a new graph with edges removed
    H = G.copy()
    for edge in drop_edges:
        H.remove_edge(*edge)

    # make each connected component (i.e. clique)
    # in H a node in a new graph
    clique_graph = nx.Graph()
    for i,clique in enumerate(nx.connected_components(H)):
        clique_graph.add_node(i, clique=clique)

    # edge weights are average IoUs between pairs within
    # separate cliques
    clique_nodes = list(clique_graph.nodes)
    for i,node1 in enumerate(clique_nodes):
        for j,node2 in enumerate(clique_nodes[i+1:]):
            clique1 = clique_graph.nodes[node1]['clique']
            clique2 = clique_graph.nodes[node2]['clique']
            clique_iou = calculate_clique_ious(G, clique1, clique2)
            if clique_iou >= min_clique_iou:
                clique_graph.add_edge(node1, node2, iou=clique_iou)

    return clique_graph

def pull_clique(G, src, dst):
    """
    Pulls instances from one clique into another clique
    and then removes the edge connecting the cliques.
    """
    # merge clique from src to dst
    src_clique = G.nodes[src]['clique']
    G.nodes[dst]['clique'] = G.nodes[dst]['clique'].union(src_clique)
    G.remove_edge(src, dst)

    return G

def merge_cliques(G):
    """
    Merges or splits instances within
    a clique of overlapping object labelmaps.
    """
    H = G.copy()
    while len(H.edges()) > 0:
        # sorted nodes by the number of neighbors
        most_connected = sorted(
            H.nodes, key=lambda x: len(list(H.neighbors(x))), reverse=True
        )[0]

        # get neighbors of the most connected node
        neighbors = list(H.neighbors(most_connected))

        # sort neighbors by the size of their cliques
        neighbors = sorted(
            neighbors, key=lambda x: len(H.nodes[x]['clique']), reverse=True
        )

        most_connected_clique = H.nodes[most_connected]['clique']
        is_pushed = False
        for neighbor in neighbors:
            if len(H.nodes[neighbor]['clique']) > len(most_connected_clique):
                pull_clique(H, most_connected, neighbor)
                is_pushed = True
            elif is_pushed:
                pull_clique(H, most_connected, neighbor)
            else:
                break

        if is_pushed:
            H.remove_node(most_connected)
        else:
            # push to neighbors with larger cliques
            neighbors = list(H.neighbors(most_connected))
            neighbors = sorted(
                neighbors, key=lambda x: len(H.nodes[x]['clique'])
            )
            # pull from neighbors with smaller or equal cliques
            for neighbor in neighbors:
                most_connected_clique = H.nodes[most_connected]['clique']
                pull_clique(H, neighbor, most_connected)
                H.remove_node(neighbor)

    return H

def mask_aggregation(masks, overlap_thr=0.1):
    """
    Takes a list of instance segmentation masks
    and returns a list of vote count maps. There is
    one map generated for each potential object instance.
    
    Arguments:
    -----------
    masks (List[np.ndarray]): List of (h, w) instance
    segmentation masks (each object has a different label).
    
    overlap_thr (Float): Maximum overlap (from 0-1) allowed
    between potential objects. Any pair of instances that exceed
    this overlap threshold will be put into the same clique.
    
    Returns:
    ---------
    instance_scores (List[np.ndarray]): List of (h, w) where
    each array has values from 0 to len(masks). Values represent
    the number of votes that a given pixel received.
    
    """
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

    instance_scores = []
    for comp in nx.connected_components(graph):
        sg = graph.subgraph(comp)
        clique_graph = create_clique_graph(sg, 0.75)
        clique_graph = merge_cliques(clique_graph)

        for node in clique_graph.nodes:
            clique = clique_graph.nodes[node]['clique']

            instance = np.zeros_like(masks[0]).astype(np.float32)
            # add all masks in the group together
            # to get a confidence mask for each
            for r in clique:
                mask = masks[mask_indices[r]]
                label = mask_labels[r]
                instance += (mask == label).astype(np.float32) / len(masks)

            instance_scores.append(instance)

    return instance_scores


def aggregated_instance_segmentation(aggregated_masks, vote_thr=0.5, start=1):
    """
    Merges a list of masks into an instance segmentation.

    Arguments:
    ----------
    aggregated_masks: List of n x (h, w) confidence aggregated masks.
    E.g. output from mask_aggregation function.

    vote_thr: Integer. Threshold number of votes over which
    to mark a pixel as part of a segmentation. Default 0.5.
    
    start: Integer. The label_id to start at for labeling the
    instances sequentially.

    Returns:
    --------
    instance_segmentation: Array of (h, w) with each detected mitochondrion
    given a different label.

    """
    mask_shape = aggregated_masks[0].shape
    instance_segmentation = np.zeros(mask_shape, dtype=np.int32)

    # add each detection with an new label
    for mask in aggregated_masks:
        # threshold the mask
        mask = mask >= vote_thr

        # relabel in case new connected
        # components fall out
        mask = measure.label(mask)

        # number and values of new labels
        # excluding background value of 0
        mask_labels = np.unique(mask)[1:]

        for ml in mask_labels:
            ml_mask = mask == ml
            instance_segmentation[ml_mask] = start
            start += 1

    return instance_segmentation
