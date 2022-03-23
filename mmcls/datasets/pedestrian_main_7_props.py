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
from mmcls.core import pedestrian_main_7_props_average_performance, mAP

@DATASETS.register_module()
class PedestrianMain7Props(MultiLabelDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset."""

    # 7 main props
    # 'upperLength', 'clothesStyles', 'hairStyles', 'lowerLength',
    # 'lowerStyles', 'shoesStyles', 'towards'
    CLASSES = ('LongSleeve', 'ShortSleeve', 'NoSleeve',
               'Solidcolor', 'multicolour', 'lattice',
               'Long', 'middle', 'Short', 'Bald',
               'Skirt', 'Trousers', 'Shorts',
               'multicolour', 'Solidcolor', 'lattice',
               'Sandals', 'LeatherShoes', 'Sneaker', 'else',
               'right', 'left', 'front', 'back')

    def __init__(self, **kwargs):
        super(PedestrianMain7Props, self).__init__(**kwargs)

    def load_annotations(self):
        """Load annotations.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        #'upperLength', 'clothesStyles', 'hairStyles', 'lowerLength',
        # 'lowerStyles', 'shoesStyles', 'towards'
        data_infos = []
        with open(self.ann_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels = [self.class_to_idx[row['upperLength']],
                          self.class_to_idx[row['clothesStyles']],
                          self.class_to_idx[row['hairStyles']],
                          self.class_to_idx[row['lowerLength']],
                          self.class_to_idx[row['lowerStyles']],
                          self.class_to_idx[row['shoesStyles']],
                          self.class_to_idx[row['towards']]]
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
            performance_values = pedestrian_main_7_props_average_performance(results, gt_labels,
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