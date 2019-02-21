# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-02-05

import os

import torch as t
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from gensim.models import KeyedVectors

from NestedLabelCNN.ACE.BaseCNN import str2wv, make_nested_label, padding_entity, make_contain_label_2_cls, \
    make_nested_label_nested_cancel
from NestedLabelCNN.ACE.BaseCNN import NestedFlagCNNClissifier
# from NestedLabelCNN.ACE.neww2vmodel import geniaDataset
import NestedLabelCNN.ACE.config as cfg
import NestedLabelCNN.ACE.BaseCNN as BaseCNN

entity_length = cfg.MAX_CLS_LEN
word_embedding_dim = cfg.WORD_EMBEDDING_DIM
batch_size = 4
max_epoch_time = 300
start_save_epoch = 5

entity_distribution = [0] * (len(cfg.LABEL))
# model_save_dictionary = "../model/genia/new_wv_cnn_nested_classifier/" + str(cfg.NESTED_CLASSES_NUM) + \
#                         "_cls_kernel_size_" + str(
#     cfg.KERNEL_LENGTH) + "_len_" + str(cfg.MAX_CLS_LEN) + "_class_weight_" + "-".join(
#     [str(i) for i in cfg.CLASS_WEIGHT]) + "/"

model_save_dictionary = "../../model/ace05/cancel_nested_cnn_nested_classifier/" + str(cfg.NESTED_CLASSES_NUM) + \
                        "_cls_kernel_size_" + str(
    cfg.KERNEL_LENGTH) + "_len_" + str(cfg.MAX_CLS_LEN) + "_class_weight_" + "-".join(
    [str(i) for i in cfg.CLASS_WEIGHT]) + "/"

if not os.path.exists(model_save_dictionary):
    os.makedirs(model_save_dictionary)


def sentence2id(word_dictionary, word_ls):
    """:return ndarray of ids."""
    result_ls = []
    for word in word_ls:
        result_ls.append(word_dictionary[word])
    return np.array(result_ls)


def prepare_train_data():  # word_dic):  # todo need modity the sentence to idx.
    word_vector_model = KeyedVectors.load_word2vec_format(cfg.BIO_NLP_VEC, binary=True)

    write_file = open("train_data_samples.txt", "w")
    path = "../../dataset/ace05/extent/train.data"

    text = open(path).read()

    ls_data = text.strip('\n').split("\n\n")
    for index in range(len(ls_data)):
        ls_data[index] = [item.strip().strip("|") for item in ls_data[index].strip('\n').split("\n")]

    train_data = []
    train_label = []
    for sub_ls in ls_data:
        if len(sub_ls) > 3:
            print(sub_ls)
            sub_ls = sub_ls[len(sub_ls) - 2:]
        if len(sub_ls) < 3:
            wait = True
            continue

        whole_sentence_ndarray = str2wv(sub_ls[0].split(), word_vector_model)  # make sentence sample.

        # whole_sentence_word_ids = sentence2id(word_dic.vocabToNew, sub_ls[0].split())

        labels = sub_ls[2].split("|")
        gt_bboxs = []
        gt_classes = []

        for item in labels:
            tuple_ = tuple([int(item) for item in item.split()[0].split(",")])
            entity_distribution[cfg.LABEL.index(item[item.find("#") + 1:])] += 1
            if tuple_[1] - tuple_[0] <= cfg.MAX_CLS_LEN:
                gt_bboxs.append([tuple_[0], tuple_[1]])
                gt_classes.append(cfg.LABEL.index(item[item.find("#") + 1:]))

        write_flag = False

        for index in range(len(gt_bboxs)):
            one_bbox = gt_bboxs[index]
            label = gt_classes[index]
            one_train_data = padding_entity(whole_sentence_ndarray[one_bbox[0]: one_bbox[1], :])  # todo padding_entity
            # one_train_data = whole_sentence_word_ids[one_bbox[0]: one_bbox[1]].tolist()
            for i in range(entity_length - len(one_train_data)):  # padding zero.
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
                # if nested_label == 1 or nested_label == 2:
                write_flag = True
        if write_flag:
            write_file.write(" ".join(sub_ls[0].split()) + "\n")
            for index in range(len(gt_bboxs)):
                one_bbox = gt_bboxs[index]
                label = gt_classes[index]
                # write_file.write(sub_ls[0] + "\n")
                # write_file.write("[" + str(one_bbox[0]) + "," + str(one_bbox[1]) + "]  |  ")
                # write_file.write(str(label) + "  |")
                write_file.write(cfg.LABEL[label] + "  |")
                write_file.write(" ".join(sub_ls[0].split()[one_bbox[0]:one_bbox[1]]) + "\n")
            write_file.write("-------------------------------------------------------------------\n")

    write_file.close()
    del word_vector_model
    return train_data, train_label


def make_batch_data(train_data, train_labels):
    for index in range(0, len(train_data), batch_size):
        left_boundary = index
        right_boundary = len(train_data) if index + batch_size > len(train_data) else index + batch_size
        train_batch_data = []
        train_batch_labels = []
        for i in range(left_boundary, right_boundary):
            train_batch_data.append(train_data[i])
            train_batch_labels.append(train_labels[i])

        yield np.array(train_batch_data).reshape(-1, 1, cfg.MAX_CLS_LEN, cfg.WORD_EMBEDDING_DIM), np.array(
            train_batch_labels)  # input word vector
        # yield np.array(train_batch_data).reshape(-1, 1, cfg.MAX_CLS_LEN), np.array(
        #     train_batch_labels)  # input word ids.


def train_batch(cnn, batch_train_data, batch_train_labels):
    batch_train_data = Variable(t.Tensor(batch_train_data)).cuda()  # input word vector
    # batch_train_data = Variable(t.Tensor(batch_train_data).long()).cuda()  # input word idx.
    batch_train_labels = Variable(t.Tensor(batch_train_labels).long()).cuda()
    predict_result = cnn.forward(batch_train_data)
    batch_loss = cnn.calc_loss(predict_result, batch_train_labels)

    predict_cls = np.argmax(predict_result.cpu().data.numpy(), axis=1)

    # BP
    cnn.optimizer.zero_grad()
    batch_loss.backward()
    cnn.optimizer.step()
    loss_ = batch_loss.cpu().data.numpy()
    return loss_, np.sum(predict_cls == batch_train_labels.cpu().data.numpy())


def train_epoch(cnn, train_data, train_labels):
    losses = []
    right_amount = 0
    for batch_train_data, batch_train_labels in make_batch_data(train_data, train_labels):  # todo
        loss_, right_entity = train_batch(cnn, batch_train_data, batch_train_labels)
        losses.append(loss_)
        right_amount += right_entity

    return np.mean(np.array(losses)), right_amount


def start_training(cnn, train_data, train_labels):
    print("Start Training---------------------")
    for epoch_index in range(max_epoch_time):
        print("Epoch:  ", epoch_index + 1)
        epoch_loss, right_amount = train_epoch(cnn, train_data, train_labels)
        print("Loss: ", epoch_loss)
        print("Accuracy: ", right_amount / len(train_data))
        if epoch_index + 1 > start_save_epoch:
            model_save_path = model_save_dictionary + "model_epoch" + str(epoch_index + 1)
            t.save(cnn.state_dict(), model_save_path)
            print("Model save in ", model_save_path)
        print("\n\n")
    return


def main():
    # word_dic = geniaDataset()
    train_data, train_labels = prepare_train_data()
    # print(entity_distribution)
    # exit()
    print("In Train Data:")
    print(sum(BaseCNN.gt_contain_matrix))
    print(BaseCNN.gt_contain_matrix)
    # cnn = NestedFlagCNNClissifier(word_dic.weight).cuda()
    cnn = NestedFlagCNNClissifier().cuda()

    # using updated word vector!

    # cnn_parameter_dict = cnn.state_dict()
    # model = t.load(
    #     "../model/genia/base_new_wv_max_cls_len32_kernal_size_5_pooling_out_2norm_centre_mse_dropout_train_iou1.0_lambda0.5/model_epoch39.pth")
    # # model_dict = model.state_dict()
    # # cnn.wordEmb = model["wordEmb.weight"]
    # pretrained_dict = {k: v for k, v in model.items() if k in ["wordEmb.weight"]}
    # cnn_parameter_dict.update(pretrained_dict)
    # cnn.load_state_dict(cnn_parameter_dict)

    cnn.optimizer = optim.Adam(filter(lambda p: p.requires_grad, cnn.parameters()), lr=cfg.LEARNING_RATE,
                               weight_decay=cfg.L2_BETA)

    start_training(cnn, train_data, train_labels)


if __name__ == '__main__':
    main()
