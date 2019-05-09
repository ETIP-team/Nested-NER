import os

vec_file = {"wiki": "./model/word2vec/wikipedia-pubmed-and-PMC-w2v.bin", "glove":"./model/word2vec/glove_word2vec.txt"}

class Config():
    def __init__(self):
        self.modify = "_allSample_regionRepresentation_noPosition"   # _base _regionRepresentation _freeze _allSample _neg_dropout _noPosition

        self.if_sent_padding = False
        self.if_chunk = False
        self.data_set = "GENIA"  # ACE ACE2004 ACE_us GENIA
        self.chunk_model_path = "./model/chunking/" + self.data_set + "/"
        self.data_path = "./dataset/" + self.data_set + "/"

        self.if_output = False
        self.if_detail = False

        self.vec_model = "wiki"  # glove wiki
        self.word2vec_path = vec_file[self.vec_model]

        self.nested_weight = [1, 1, 1, 1]
        self.nested_flag = ['NT', 'ST', 'DT', 'MT']

        self.C = 10  # 10 8 6
        self.word_embedding_size = 100 if self.vec_model == "glove" else 200
        self.if_char = True
        self.char_embedding_size = 25
        self.if_pos = True
        self.pos_embedding_size = 6
        self.in_channels = 3
        self.feature_maps_size = 36
        self.kernel_size = 5
        self.pooling_out_size = 2

        self.lambd = 0
        self.dropout = 0.5
        self.if_c_attention = False
        self.if_s_attention = False
        self.if_transformer = True
        self.N = 3
        self.h = 4
        self.if_bidirectional = True

        self.epoch = 100
        self.batch_size = 8
        self.opt = "Adam"
        self.lr = 3e-4  # 0.005 1e-4
        self.l2 = 0  # 1e-4 1e-3
        self.toi_num = 64
        self.pos_rate = 0.25
        self.train_pos_iou_th = 1
        self.train_neg_iou_th = 0.67
        self.if_shuffle = True

        self.if_filter = False
        self.score_th = 0.5
        self.if_nms = False
        self.if_nms_with_nested = False
        self.nms_th = 0.6

        self.if_gpu = True

        self.sample_modify = f""  # f""  # f"_batch_size{self.batch_size}"
        self.train_modify = f"_N{self.N}_h{self.h}"  # _w _w+c

    def __repr__(self):
        return str(vars(self))

    def get_pkl_path(self, mode):
        path = config.data_path
        if mode == "word2vec":
            path += f"word_vec"
        else:
            if mode == "config":
                path += f"config"
            else:
                path += mode + "/"
                if mode == "train":
                    path += f"train_iou_th_pos{self.train_pos_iou_th}_neg{self.train_neg_iou_th:.2f}"
                # elif config.if_crf:
                #     path += "with_crf"
            if self.if_chunk:
                path += "_chunking"
            else:
                path += "_enumerate"
            if not self.if_sent_padding:
                path += "_noPadding"

            path += f"_max_len{self.C}"
            path += self.sample_modify
            # path += "_v2"
        return path + f"_{self.vec_model}.pkl"

    def load_config(self, misc_dict):
        self.word_kinds = misc_dict["word_kinds"]
        self.char_kinds = misc_dict["char_kinds"]
        self.pos_tag_kinds = misc_dict["pos_tag_kinds"]
        self.label_kinds = misc_dict["label_kinds"]
        self.id2label = misc_dict["id2label"]

        print(self)
        self.id2word = misc_dict["id2word"]

    def get_model_path(self):
        path = f"./model/{self.data_set}/iou_th_pos{self.train_pos_iou_th}_neg{self.train_neg_iou_th:.2f}_lambd_{self.lambd}"
        # path += f"_in_channels_{self.in_channels}"
        path += f"_max_len{self.C}"
        path += f"_{self.vec_model}"
        path += self.modify
        path += self.sample_modify
        path += self.train_modify
        if self.if_chunk:
            path += "_chunking"
        else:
            path += "_enumerate"
        if not self.if_sent_padding:
            path += "_noPadding"
        if self.if_c_attention:
            path += "_c_attention"
        if self.if_s_attention:
            path += "_s_attention"
        if self.if_transformer:
            path += "_transformer"
            if self.if_bidirectional:
                path += "_bidirectional"
        # path += "_v2"
        if not os.path.exists(path):
            os.mkdir(path)
        return path + "/"

    def get_result_path(self):
        path = f"./result/{self.data_set}/iou_th_pos{self.train_pos_iou_th}_neg{self.train_neg_iou_th:.2f}_lambd_{self.lambd}"
        # path += f"_in_channels_{self.in_channels}"
        path += f"_max_len{self.C}"
        path += f"_{self.vec_model}"
        path += self.modify
        if self.if_chunk:
            path += "_chunking"
        else:
            path += "_enumerate"
        if not self.if_sent_padding:
            path += "_noPadding"
        if self.if_c_attention:
            path += "_c_attention"
        if self.if_s_attention:
            path += "_s_attention"
        if self.if_transformer:
            path += "_transformer"
            if self.if_bidirectional:
                path += "_bidirectional"
        return path + ".data"


config = Config()
