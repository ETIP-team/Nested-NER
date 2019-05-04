# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-04-03


import os
import time
import torch as t


class Config:
    def __init__(self):
        # global config
        self.running_mode = None
        self.cuda = True  # False
        self.WORD_VEC_MODEL_PATH = "../model/word_vector_model/wikipedia-pubmed-and-PMC-w2v.bin"  # ACE05
        # self.WORD_VEC_MODEL_PATH = "../model/word_vector_model/gloved_200d_word2vec.txt"  # GLOVE FOR ACE
        # model config.

        self.attention_method = "general"  # "general",  "dot",  "concate", "PLQ", "concate_before_attention"
        self.encoder_decoder_connection = True
        self.level_connection = True
        self.fill_label_max = True
        self.add_control_flag = True
        self.encode_bi_flag = True
        self.decode_bi_flag = True
        self.train_empty_entity = False

        self.embedding_dim = 200
        # self.num_embeddings = 5
        self.hidden_units = 100  # embedding size must equals hidden_units * 2
        self.linear_hidden_units = 150  # 70
        self.encode_num_layers = 1
        self.decode_num_layers = 1

        self.learning_rate = 3e-4
        self.dropout_rate = 0.5  # 0.5  # Dropout!!!
        self.l2_penalty = 1e-4

        self.dataset_type = "JNLPBA"  # ACE05_Lu  # ACE05  # ACE2004   # ACE04+05

        if self.dataset_type.startswith("ACE"):
            self.labels = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
        elif self.dataset_type == "GENIA":
            self.labels = ['DNA', 'RNA', 'cell_type', 'protein', 'cell_line']
        elif self.dataset_type == "JNLPBA":
            self.labels = ['DNA', 'RNA', 'cell_type', 'protein', 'cell_line']
        else:
            raise KeyError("Should not be here, no such dataset!")
        self.bio_labels = ["O"]
        for one_label in self.labels:
            self.bio_labels.extend(["B-" + one_label, "I-" + one_label])

        self.classes_num = len(self.bio_labels)  # Begin, Inside, Out of entity
        self.max_nested_level = 1

        # train config
        self.num_batch = 4

        self.max_epoch = 40
        self.start_save_epoch = 1

        self.start_test_epoch = 3

        self.train_data = None
        self.train_label = None
        self.train_str = None
        self.dev_data = None
        self.dev_label = None

        self.test_data = None
        self.test_str = None
        self.test_label = None
        self.metric_dicts = None

        self.output_path = "../result/result.data"

    def list_all_member(self):
        print(time.asctime(time.localtime(time.time())))
        for name, value in vars(self).items():
            if value is not None:
                print('%s=%s' % (name, value))

    def model_save_path(self, epoch, create_flag=True):
        final_model_path = "../model/" + self.dataset_type + "/"
        # final_model_path += "bi_" if self.encode_bi_flag else ""
        # final_model_path += "previous_s_i_"
        final_model_path += "bi_de_" if self.decode_bi_flag else ""
        final_model_path += "glove_" if "glove" in self.WORD_VEC_MODEL_PATH else ""
        final_model_path += "pre_s_i_"
        final_model_path += "control" + str(self.add_control_flag)[0]
        final_model_path += self.attention_method
        # final_model_path += "_max_nested_level_" + str(self.max_nested_level)
        final_model_path += "_max_level_" + str(self.max_nested_level)[0]
        final_model_path += "_en_de" + str(self.encoder_decoder_connection)[0]
        final_model_path += "_level" + str(self.level_connection)[0]
        # final_model_path += "_train_empty_entity" + str(self.train_empty_entity)
        final_model_path += "_train_emptyF"
        final_model_path += "F" if not self.train_empty_entity else ""  # empty word
        final_model_path += "_fill_label" + str(self.fill_label_max)[0]
        # final_model_path += "_hidden_units_" + str(self.hidden_units)
        # final_model_path += "_learning_rate_" + str(self.learning_rate)
        # final_model_path += "_num_batch_" + str(self.num_batch)
        final_model_path += "_l2_" + str(self.l2_penalty)
        final_model_path += "drop" + str(self.dropout_rate) if self.dropout_rate > 0 else ""
        # final_model_path += "_1_linear"
        # todo add.

        final_model_path += "/"
        if create_flag and not os.path.exists(final_model_path):
            os.makedirs(final_model_path)
            print("create model dir " + final_model_path + " successfully")

        return final_model_path + "model_epoch_" + str(epoch + 1) + ".pth"

    def load_model(self, model, epoch):
        model_path = self.model_save_path(epoch, False)
        model.load_state_dict(t.load(model_path))

        print("load model from " + model_path)
        return model

    def save_model(self, model, epoch):
        model_path = self.model_save_path(epoch, True)

        t.save(model.state_dict(), model_path)

        print("model saved in " + model_path + " successfully")
        return

    def get_train_path(self):
        if self.dataset_type == "ACE2004":
            return "../data/dataset_layer/ACE2004/layer_train.data"
        elif self.dataset_type == "ACE05":
            return "../data/big_first/layer_train.data"
            # return "../data/big_first/layer_train_sample.data"
        elif self.dataset_type == "ACE05_Lu":
            return "../data/dataset_layer/ACE/layer_train.data"
        elif self.dataset_type == "GENIA":
            return "../data/dataset_layer/GENIA/layer_train.data"
        elif self.dataset_type == "ACE04+05":
            return "../data/dataset_layer/ACE04+05/layer_train.data"
        elif self.dataset_type == "JNLPBA":
            return "../data/dataset_layer/JNLPBA/layer_train.data"

    def get_dev_path(self):
        if self.dataset_type == "ACE2004":
            return "../data/dataset_layer/ACE2004/layer_dev.data"
        elif self.dataset_type == "ACE05":
            return "../data/big_first/layer_dev.data"
            # return "../data/big_first/layer_dev_sample.data"
        elif self.dataset_type == "ACE05_Lu":
            return "../data/dataset_layer/ACE/layer_dev.data"
        elif self.dataset_type == "GENIA":
            return "../data/dataset_layer/GENIA/layer_dev.data"
        elif self.dataset_type == "ACE04+05":
            return "../data/dataset_layer/ACE04+05/layer_dev.data"
        elif self.dataset_type == "JNLPBA":
            return "../data/dataset_layer/JNLPBA/layer_dev.data"

    def get_test_path(self):
        if self.dataset_type == "ACE2004":
            return "../data/dataset_layer/ACE2004/layer_test.data"
        elif self.dataset_type == "ACE05":
            return "../data/big_first/layer_test.data"
            # return "../data/big_first/layer_test_sample.data"
        elif self.dataset_type == "ACE05_Lu":
            return "../data/dataset_layer/ACE/layer_test.data"
        elif self.dataset_type == "GENIA":
            return "../data/dataset_layer/GENIA/layer_test.data"
        elif self.dataset_type == "ACE04+05":
            return "../data/dataset_layer/ACE04+05/layer_test.data"
        elif self.dataset_type == "JNLPBA":
            return "../data/dataset_layer/JNLPBA/layer_test.data"


if __name__ == '__main__':
    config = Config()

    config.model_save_path(0)
