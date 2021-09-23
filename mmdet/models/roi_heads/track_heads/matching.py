import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from tracking_utils import kalman_filter
import time

def fuse_motion(kf, cost_matrix, means, convariances, det, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    ### tlbr_to_tlwh
    det[:, 2:] -= det[:, :2]    
    ### tlwh_to_xyah
    det[:, :2] += det[:, 2:] / 2
    det[:, 2] /= det[:, 3]
    
    for i in range(len(means)):
        gating_distance = kf.gating_distance(
            means[i], covariances[i], det, only_position, metric='maha')
        cost_matrix[gating_distance > gating_threshold, i] = np.inf
        cost_matrix[:, i] = lambda_ * cost_matrix[:, i] + (1 - lambda_) * gating_distance
    return cost_matrix
