# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-02-06


import os

import torch as t
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.autograd import Variable
from gensim.models import KeyedVectors

from NestedLabelCNN.GENIA.BaseCNN import str2wv, make_nested_label, padding_entity, make_contain_label_2_cls, \
    make_nested_label_nested_cancel
from NestedLabelCNN.GENIA.BaseCNN import NestedFlagCNNClissifier
from NestedLabelCNN.GENIA.neww2vmodel import geniaDataset
import NestedLabelCNN.GENIA.BaseCNN as BaseCNN
import NestedLabelCNN.GENIA.config as cfg

entity_length = cfg.MAX_CLS_LEN
max_epoch_time = 50
start_test_epoch = 5

# model_dictionary = "../model/genia/cnn_nested_classifier/kernel_size_" + str(cfg.KERNEL_LENGTH) + "/"
# model_dictionary = "../model/genia/cnn_nested_classifier/kernel_size_class_weight" + str(cfg.KERNEL_LENGTH) + "/"

# model_dictionary = "../model/genia/new_wv_cnn_nested_classifier/kernel_size_" + str(
#     cfg.KERNEL_LENGTH) + "class_weight_" + "1-50-50-50" + "/"

# model_dictionary = "../model/genia/new_wv_cnn_nested_classifier/" + str(cfg.NESTED_CLASSES_NUM) + \
#                         "_cls_kernel_size_" + str(
#     cfg.KERNEL_LENGTH) + "class_weight_" + "1-50" + "/"

model_dictionary = "../../model/genia/cancel_nested_cnn_nested_classifier/" + str(cfg.NESTED_CLASSES_NUM) + \
                   "_cls_kernel_size_" + str(
    cfg.KERNEL_LENGTH) + "_len_" + str(cfg.MAX_CLS_LEN) + "_class_weight_" + "-".join(
    [str(i) for i in cfg.CLASS_WEIGHT]) + "/"


# model_dictionary = "../model/genia/new_wv_cnn_nested_classifier/" + str(cfg.NESTED_CLASSES_NUM) + \
#                         "_cls_kernel_size_" + str(
#     cfg.KERNEL_LENGTH) + "_len_" + str(cfg.MAX_CLS_LEN) + "_class_weight_" + "-".join(
#     [str(i) for i in cfg.CLASS_WEIGHT]) + "/"

def sentence2id(word_dictionary, word_ls):
    """:return ndarray of ids."""
    result_ls = []
    for word in word_ls:
        result_ls.append(word_dictionary[word])
    return np.array(result_ls)


def prepare_test_data():  # word_dic):
    word_vector_model = KeyedVectors.load_word2vec_format(cfg.BIO_NLP_VEC, binary=True)

    # path = "../dataset/train/train.data"
    path = "../../dataset/test/test.data"
    text = open(path, encoding="utf-8").read()

    ls_data = text.strip('\n').split("\n\n")
    for index in range(len(ls_data)):
        ls_data[index] = [item.strip().strip("|") for item in ls_data[index].strip('\n').split("\n")]

    train_data = []
    train_label = []
    for sub_ls in ls_data:
        if len(sub_ls) > 2:
            print(sub_ls)
            sub_ls = sub_ls[len(sub_ls) - 2:]
        if len(sub_ls) < 1:
            continue

        # whole_sentence_word_ids = sentence2id(word_dic.vocabToNew, sub_ls[0].split())
        whole_sentence_ndarray = str2wv(sub_ls[0].split(), word_vector_model)  # make sentence sample.

        labels = sub_ls[1].split("|")
        gt_bboxs = []
        gt_classes = []

        for item in labels:
            tuple_ = tuple([int(item) for item in item.split()[0].split(",")])
            if tuple_[1] - tuple_[0] <= cfg.MAX_CLS_LEN:
                gt_bboxs.append([tuple_[0], tuple_[1]])
                gt_classes.append(cfg.LABEL.index(item[item.find("#") + 1:]) + 1)

        for index in range(len(gt_bboxs)):
            one_bbox = gt_bboxs[index]
            label = gt_classes[index]
            # one_train_data = whole_sentence_word_ids[one_bbox[0]: one_bbox[1]].tolist()
            one_train_data = padding_entity(whole_sentence_ndarray[one_bbox[0]: one_bbox[1], :])  # todo padding_entity
            for i in range(entity_length - len(one_train_data)):
                one_train_data.append(0)

            # nested_label = make_nested_label(np.array(one_bbox), label, gt_classes, np.array(gt_bboxs))
            if cfg.NESTED_CLASSES_NUM == 2:
                nested_label = make_contain_label_2_cls(np.array(one_bbox), label, gt_classes, np.array(gt_bboxs))
            if cfg.NESTED_CLASSES_NUM == 4:
                nested_label = make_nested_label_nested_cancel(np.array(one_bbox), label, gt_classes,
                                                               np.array(gt_bboxs))
            if nested_label != -1:
                train_data.append(one_train_data)
                train_label.append(nested_label)

    # train_data = np.array(train_data)
    del word_vector_model
    return train_data, train_label


def load_model(model_path):
    # cnn = NestedFlagCNNClissifier(word_dic.weight).cuda()
    cnn = NestedFlagCNNClissifier().cuda()
    cnn.load_state_dict(t.load(model_path))
    return cnn


def _test_one_sentence(cnn, one_test_entity, one_entity_label, confusion_matrix):
    one_test_entity = Variable(t.Tensor(one_test_entity)).cuda()
    # one_test_entity = Variable(t.Tensor(one_test_entity).long()).cuda()
    one_entity_label = Variable(t.Tensor(one_entity_label).long()).cuda()
    predict_result = cnn.forward(one_test_entity)
    batch_loss = cnn.calc_loss(predict_result, one_entity_label)

    predict_cls = np.argmax(predict_result.cpu().data.numpy(), axis=1)

    loss_ = batch_loss.cpu().data.numpy()
    confusion_matrix[int(predict_cls[0]), int(one_entity_label[0])] += 1
    return loss_


def _test(cnn, test_data, test_labels):
    print("Testing---------------------")
    losses = []
    confusion_matrix = initialize_confusion_matrix()
    for entity_index in range(len(test_data)):
        one_test_entity = np.array([test_data[entity_index].copy()]).reshape(-1, 1, cfg.MAX_CLS_LEN,
                                                                             cfg.WORD_EMBEDDING_DIM)
        # one_test_entity = np.array([test_data[entity_index].copy()]).reshape(-1, 1, cfg.MAX_CLS_LEN)

        one_test_label = np.array([test_labels[entity_index]])

        one_loss = _test_one_sentence(cnn, one_test_entity, one_test_label, confusion_matrix)

        losses.append(one_loss)
    print("Loss: ", np.mean(np.array(losses)))
    print(confusion_matrix)
    metrics_df = make_contain_metric_df(confusion_matrix.copy())
    print(metrics_df)
    return


def make_contain_metric_df(contain_confusion_matrix):
    classes_columns = cfg.NESTED_LABEL
    contain_df = pd.DataFrame(contain_confusion_matrix, columns=classes_columns)

    contain_df.loc([0, 0])
    metric_ls = []
    for i in range(len(classes_columns)):
        recall = contain_df[classes_columns[i]][i] / contain_df[classes_columns[i]].sum() if contain_df[
                                                                                                 classes_columns[
                                                                                                     i]].sum() > 0 else 0
        precision = contain_df[classes_columns[i]][i] / contain_df.iloc[i, :].sum() if contain_df.iloc[i,
                                                                                       :].sum() > 0 else 0
        F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metric_ls.append([precision, recall, F1])
    return pd.DataFrame(metric_ls, columns=["Precision", "Recall", "F1"], index=classes_columns)


def initialize_confusion_matrix():
    confusion_matrix = np.zeros((cfg.NESTED_CLASSES_NUM, cfg.NESTED_CLASSES_NUM), dtype=np.int32)
    return confusion_matrix


def main():
    # word_dic = geniaDataset()
    for epoch_index in range(start_test_epoch + 1, max_epoch_time):
        model_path = model_dictionary + "model_epoch" + str(epoch_index)
        cnn = load_model(model_path)  # , word_dic)
        print("Load Model from: ", model_path)
        test_data, test_labels = prepare_test_data()
        # del word_dic
        print("In Test Data:")
        print(sum(BaseCNN.gt_contain_matrix))
        print(BaseCNN.gt_contain_matrix)
        _test(cnn, test_data, test_labels)


if __name__ == '__main__':
    main()
