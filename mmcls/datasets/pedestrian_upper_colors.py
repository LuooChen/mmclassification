# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
import csv

from .builder import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class PedestrianUpperColors(MultiLabelDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset."""

    # upper_colors
    CLASSES = ('upperBlack',
               'upperBrown', 'upperBlue', 'upperGreen', 'upperGray', 'upperOrange',
               'upperPink', 'upperPurple', 'upperRed', 'upperWhite', 'upperYellow')

    def __init__(self, **kwargs):
        super(PedestrianUpperColors, self).__init__(**kwargs)
    
    def get_upper_colors_labels(self, row) -> list:
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
                labels = self.get_upper_colors_labels(row)
                gt_label = np.zeros(len(self.CLASSES))
                gt_label[labels] = 1
                info = dict(
                    img_prefix=self.data_prefix,
                    img_info=dict(filename=row['name']),
                    gt_label=gt_label.astype(np.int8))
                data_infos.append(info)
        return data_infos
