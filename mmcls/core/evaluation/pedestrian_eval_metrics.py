# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch

upper_classes = ['LongSleeve', 'ShortSleeve', 'NoSleeve',
                 'Solidcolor', 'multicolour', 'lattice',
                 'Long', 'middle', 'Short', 'Bald']

upper_colors_classes = ['upperBlack',
                        'upperBrown', 'upperBlue', 'upperGreen', 'upperGray', 'upperOrange',
                        'upperPink', 'upperPurple', 'upperRed', 'upperWhite', 'upperYellow']

def nd_format(pred_class, classes, format_string ='{0:.3f}'):
    pred_class = [format_string.format(v,i) for i,v in enumerate(pred_class)]
    result = []
    for (index,cur) in enumerate(classes):
        result.append({
            cur: pred_class[index]
        })
    return result

def upper_average_performance(pred, target, thr=None, k=None):
    """Calculate CP, CR, CF1, OP, OR, OF1, MF1, where C stands for per-class
    average, O stands for overall average, P stands for precision, R stands for
    recall and F1 stands for F1-score, MF1 stands for MacroF1.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.
        thr (float): The confidence threshold. Defaults to None.
        k (int): Top-k performance. Note that if thr and k are both given, k
            will be ignored. Defaults to None.

    Returns:
        tuple: (CP, CR, CF1, OP, OR, OF1, MF1)
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'

    eps = np.finfo(np.float32).eps
    target[target == -1] = 0
    # special predicted positive for pedestrian upper
    
    # 'upperLength', 'clothesStyles', 'hairStyles'
    # CLASSES = ('LongSleeve', 'ShortSleeve', 'NoSleeve',
    #            'Solidcolor', 'multicolour', 'lattice',
    #            'Long', 'middle', 'Short', 'Bald')
    pos_inds = np.zeros_like(pred)
    upperLength_max_col_indexes = pred[:,:3].argmax(axis=1)
    for (row, col) in enumerate(upperLength_max_col_indexes):
        pos_inds[row][col] = 1
    clothesStyles_max_col_indexes = pred[:,3:6].argmax(axis=1) + 3
    for (row, col) in enumerate(clothesStyles_max_col_indexes):
        pos_inds[row][col] = 1
    hairStyles_max_col_indexes = pred[:,6:].argmax(axis=1) + 6
    for (row, col) in enumerate(hairStyles_max_col_indexes):
        pos_inds[row][col] = 1

    tp = (pos_inds * target) == 1
    fp = (pos_inds * (1 - target)) == 1
    fn = ((1 - pos_inds) * target) == 1

    precision_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fp.sum(axis=0), eps)
    recall_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fn.sum(axis=0), eps)
    print('precision_class: ', nd_format(precision_class, upper_classes))
    print('recall_class: ', nd_format(recall_class, upper_classes))
    # calculate MacroF1
    f1_class = 2 * precision_class * recall_class / np.maximum(precision_class + recall_class, eps)
    print('f1_class: ', nd_format(f1_class, upper_classes))
    MF1 = f1_class.mean() * 100.0
    CP = precision_class.mean() * 100.0
    CR = recall_class.mean() * 100.0
    CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)
    OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps) * 100.0
    OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps) * 100.0
    OF1 = 2 * OP * OR / np.maximum(OP + OR, eps)
    return CP, CR, CF1, MF1, OP, OR, OF1

def upper_colors_average_performance(pred, target, thr=0.5, k=3):
    """Calculate CP, CR, CF1, OP, OR, OF1, MF1, where C stands for per-class
    average, O stands for overall average, P stands for precision, R stands for
    recall and F1 stands for F1-score, MF1 stands for MacroF1.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.
        thr (float): The confidence threshold. Defaults to None.
        k (int): Top-k performance. Note that if thr and k are both given, k
            will be ignored. Defaults to None.

    Returns:
        tuple: (CP, CR, CF1, OP, OR, OF1, MF1)
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'

    eps = np.finfo(np.float32).eps
    target[target == -1] = 0
    
    # special predicted positive for pedestrian upper
    if k is None:
        k = 3
    # top-k labels will be predicted positive for any example
    sort_inds = np.argsort(-pred, axis=1)
    sort_inds_ = sort_inds[:, :k]
    inds = np.indices(sort_inds_.shape)
    pos_inds = np.zeros_like(pred)
    pos_inds[inds[0], sort_inds_] = pred[inds[0], sort_inds_]
    if thr is not None:
        # a label is predicted positive if the confidence is no lower than thr
        pos_inds = np.where(pos_inds >= thr, 1, 0)
    else:
        pos_inds[inds[0], sort_inds_] = 1

    tp = (pos_inds * target) == 1
    fp = (pos_inds * (1 - target)) == 1
    fn = ((1 - pos_inds) * target) == 1

    precision_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fp.sum(axis=0), eps)
    recall_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fn.sum(axis=0), eps)
    np.set_printoptions(linewidth=400)
    print('precision_class: ', nd_format(precision_class, upper_colors_classes))
    print('recall_class: ', nd_format(recall_class, upper_colors_classes))
    # calculate MacroF1
    f1_class = 2 * precision_class * recall_class / np.maximum(precision_class + recall_class, eps)
    print('f1_class: ', nd_format(f1_class, upper_colors_classes))
    MF1 = f1_class.mean() * 100.0
    CP = precision_class.mean() * 100.0
    CR = recall_class.mean() * 100.0
    CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)
    OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps) * 100.0
    OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps) * 100.0
    OF1 = 2 * OP * OR / np.maximum(OP + OR, eps)
    return CP, CR, CF1, MF1, OP, OR, OF1
