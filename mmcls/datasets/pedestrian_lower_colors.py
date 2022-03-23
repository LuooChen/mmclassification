# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
import csv
from mmcv.utils import print_log
import logging

from .builder import DATASETS
from .multi_label import MultiLabelDataset
from mmcls.core import pedestrian_colors_average_performance, mAP


@DATASETS.register_module()
class PedestrianLowerColors(MultiLabelDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset."""

    # lower_colors
    CLASSES = ('lowerBlack',
               'lowerBrown', 'lowerBlue', 'lowerGreen', 'lowerGray', 'lowerOrange',
               'lowerPink', 'lowerPurple', 'lowerRed', 'lowerWhite', 'lowerYellow')

    def __init__(self, **kwargs):
        super(PedestrianLowerColors, self).__init__(**kwargs)

    def get_lower_colors_labels(self, row) -> list:
        labels = []
        for (index, color) in enumerate(self.CLASSES):
            if row[color] != '':
                labels.append(index)
        return labels

    def load_annotations(self):
        """Load annotations.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        with open(self.ann_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels = self.get_lower_colors_labels(row)
                gt_label = np.zeros(len(self.CLASSES))
                gt_label[labels] = 1
                info = dict(
                    img_prefix=self.data_prefix,
                    img_info=dict(filename=row['name']),
                    gt_label=gt_label.astype(np.int8))
                data_infos.append(info)
        return data_infos

    def nd_format(self, pred_class, format_string ='{0:.3f}'):
        _classes = list(self.CLASSES)
        pred_class = [format_string.format(v,i) for i,v in enumerate(pred_class)]
        result = []
        for (index, cur) in enumerate(_classes):
            result.append({
                cur: pred_class[index]
            })
        return str(result)

    def evaluate(self,
                 results,
                 metric='MF1',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is 'mAP'. Options are 'mAP', 'CP', 'CR', 'CF1',
                'MF1', 'OP', 'OR' and 'OF1'.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'k' and 'thr'. Defaults to None
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.

        Returns:
            dict: evaluation results
        """
        if metric_options is None or metric_options == {}:
            metric_options = {'thr': 0.5}

        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['mAP', 'CP', 'CR', 'CF1', 'MF1', 'OP', 'OR', 'OF1']
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        if 'mAP' in metrics:
            mAP_value = mAP(results, gt_labels)
            eval_results['mAP'] = mAP_value
        if len(set(metrics) - {'mAP'}) != 0:
            # performance_keys = ['CP', 'CR', 'CF1', 'MF1', 'OP', 'OR', 'OF1']
            performance_keys = ['CP', 'CR', 'CF1', 'MF1', 'OP', 'OR', 'OF1', 'precision_class', 'recall_class', 'f1_class']
            performance_values = pedestrian_colors_average_performance(results, gt_labels,
                                                     **metric_options)
            for k, v in zip(performance_keys, performance_values):
                if k in metrics:
                    eval_results[k] = v
                if k == 'precision_class':
                    print_log('precision_class: ' + self.nd_format(v), logger=logger, level=logging.INFO)
                if k == 'recall_class':
                    print_log('recall_class: ' + self.nd_format(v), logger=logger, level=logging.INFO)
                if k == 'f1_class':
                    print_log('f1_class: ' + self.nd_format(v), logger=logger, level=logging.INFO)

        return eval_results