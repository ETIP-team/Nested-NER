# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-02-05

import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import NestedLabelCNN.ACE.config as cfg

# from NestedLabelCNN.ACE.neww2vmodel import geniaDataset

empty_wv = np.zeros((1, cfg.WORD_EMBEDDING_DIM))
sentence_length = cfg.SENTENCE_LENGTH

gt_contain_matrix = np.zeros((len(cfg.LABEL), len(cfg.LABEL)), dtype=np.int32)


class NestedFlagCNNClissifier(nn.Module):
    def __init__(self, word_embedding_dim=cfg.WORD_EMBEDDING_DIM, max_length=cfg.MAX_CLS_LEN,
                 feature_maps_number=cfg.FEATURE_MAPS_NUM, kernel_length=cfg.KERNEL_LENGTH,
                 pooling_size=cfg.POOLING_SIZE,
                 classes_num=cfg.NESTED_CLASSES_NUM):
        super(NestedFlagCNNClissifier, self).__init__()

        # self.wordEmb = nn.Embedding.from_pretrained(weight)
        # default configuration is true
        # self.wordEmb.weight.requires_grad = False

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=feature_maps_number,
                               kernel_size=(kernel_length, word_embedding_dim),  # kernal size
                               stride=1)
        # pooling in forward
        # self.size = pooling_out_size
        self.pooling_size = pooling_size
        # fully connected
        self.output_length = (max_length - int(kernel_length / 2) * 2) / pooling_size
        self.fc1 = nn.Linear(int(self.output_length * feature_maps_number),
                             int(self.output_length * feature_maps_number / 2))
        self.fc2 = nn.Linear(self.output_length * feature_maps_number / 2, classes_num)
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=Variable(t.Tensor(np.array(cfg.CLASS_WEIGHT)).cuda()))

        self.optimizer = None

    def forward(self, x):
        # x = self.wordEmb(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(self.pooling_size, 1))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

    def calc_loss(self, pred, labels):
        return self.cross_entropy_loss(pred, labels)


def str2wv(ls, word_vector_model):  # modify in Arguments
    arr = empty_wv
    for i in ls:
        try:
            arr = np.vstack((arr, word_vector_model[i]))
        except KeyError:
            arr = np.vstack((arr, empty_wv))
    arr = np.delete(arr, 0, axis=0)
    row, column = arr.shape
    try:
        arr = np.pad(arr, ((0, sentence_length - row), (0, 0)), 'constant', constant_values=0)
    except Exception:
        print(ls)
        exit("Above sentence length is " + str(len(ls)) + " which was too long")
    return arr


def make_nested_label(region_proposal, region_proposal_cls, gt_classes, gt_boxes):
    """compare the region proposal with the ground truth classes.
        return the one hot label list [0,1,2,3]
    """
    without_self_gt_index = ~(gt_boxes == region_proposal).all(axis=1)  # get index without the self
    boxes_without_self_gt = gt_boxes[without_self_gt_index]
    cls_without_self_gt = np.array(gt_classes)[without_self_gt_index]
    contain_index = (region_proposal[0] <= boxes_without_self_gt[:, 0]) * (
            region_proposal[1] >= boxes_without_self_gt[:, 1])

    contain_different_flag = False
    contain_same_flag = False
    for cls in cls_without_self_gt[contain_index]:
        gt_contain_matrix[region_proposal_cls, cls] += 1
        if cls == region_proposal_cls:
            contain_same_flag = True
        else:
            contain_different_flag = True
    if contain_different_flag and contain_same_flag:
        return 3
    elif contain_different_flag:
        return 2
    elif contain_same_flag:
        return 1
    else:
        return 0


def make_nested_label_nested_cancel(region_proposal, region_proposal_cls, gt_classes, gt_boxes):
    """compare the region proposal with the ground truth classes.
        return the one hot label list [0,1,2,3]
    """
    without_self_gt_index = ~(gt_boxes == region_proposal).all(axis=1)  # get index without the self
    boxes_without_self_gt = gt_boxes[without_self_gt_index]
    cls_without_self_gt = np.array(gt_classes)[without_self_gt_index]
    contain_index = (region_proposal[0] <= boxes_without_self_gt[:, 0]) * (
            region_proposal[1] >= boxes_without_self_gt[:, 1])

    contained_index = (region_proposal[0] >= boxes_without_self_gt[:, 0]) * (
            region_proposal[1] <= boxes_without_self_gt[:, 1])
    # contain by others
    if contained_index.any():  # not training!
        return -1
    contain_different_flag = False
    contain_same_flag = False
    for cls in cls_without_self_gt[contain_index]:
        gt_contain_matrix[region_proposal_cls, cls] += 1
        if cls == region_proposal_cls:
            contain_same_flag = True
        else:
            contain_different_flag = True
    if contain_different_flag and contain_same_flag:
        return 3
    elif contain_different_flag:
        return 2
    elif contain_same_flag:
        return 1
    else:
        return 0


def make_contain_label_2_cls(region_proposal, region_proposal_cls, gt_classes, gt_boxes):
    """compare the region proposal with the ground truth classes.
        return the one hot label list [0,1,2,3]
    """
    without_self_gt_index = ~(gt_boxes == region_proposal).all(axis=1)  # get index without the self
    boxes_without_self_gt = gt_boxes[without_self_gt_index]
    cls_without_self_gt = np.array(gt_classes)[without_self_gt_index]
    contain_index = (region_proposal[0] <= boxes_without_self_gt[:, 0]) * (
            region_proposal[1] >= boxes_without_self_gt[:, 1])

    # contain_same_flag = False
    # for cls in cls_without_self_gt[contain_index]:
    #     gt_contain_matrix[region_proposal_cls - 1, cls - 1] += 1
    #     if cls == region_proposal_cls:
    #         contain_same_flag = True
    # if contain_same_flag:
    #     return 1
    # else:
    #     return 0

    contain_different_flag = False
    contain_same_flag = False
    for cls in cls_without_self_gt[contain_index]:
        gt_contain_matrix[region_proposal_cls, cls] += 1
        if cls == region_proposal_cls:
            contain_same_flag = True
        else:
            contain_different_flag = True
    if contain_different_flag and contain_same_flag:
        return 1
    elif contain_different_flag:
        return -1
    elif contain_same_flag:
        return 1
    else:
        return 0


def padding_entity(entity_2d_ndarray):
    padding_length = cfg.MAX_CLS_LEN - entity_2d_ndarray.shape[0]
    assert padding_length >= 0

    return np.pad(entity_2d_ndarray, ((0, padding_length), (0, 0)), 'constant', constant_values=0)


if __name__ == '__main__':
    # word_dic = geniaDataset()
    # cnn = NestedFlagCNNClissifier(word_dic.weight)
    cnn = NestedFlagCNNClissifier()

    for name, parameter in cnn.named_parameters():
        print(name)
        print(parameter)
