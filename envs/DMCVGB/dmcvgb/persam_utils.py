import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

def calculate_dice_loss(inputs, targets, num_masks = 1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks = 1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
    
    return topk_xy, topk_label


def negative_point_selection(mask_sim, topk=1, box=None):
	if box is None:
		box = np.array([0, 0, mask_sim.shape[0]-1, mask_sim.shape[1]-1])
	
	n_mask_sim = mask_sim.clone()
	n_mask_sim = n_mask_sim[box[1]:box[3], box[0]:box[2]]
	if n_mask_sim.shape[0] == 0 or n_mask_sim.shape[1] == 0:
		return np.array([[0, 0]]), np.array([0])
	n_topk_xy = n_mask_sim.flatten(0).topk(topk, largest=False)[1]
	w, h = n_mask_sim.shape
	n_topk_x = (n_topk_xy // h).unsqueeze(0)
	n_topk_y = (n_topk_xy - n_topk_x * h)
	n_topk_x = n_topk_x + box[1]
	n_topk_y = n_topk_y + box[0]
	n_topk_xy = torch.cat((n_topk_y, n_topk_x), dim=0).permute(1, 0)
	n_topk_label = np.array([0] * topk)
	n_topk_xy = n_topk_xy.cpu().numpy()

	return n_topk_xy, n_topk_label