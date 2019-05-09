import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pickle
from collections import namedtuple, defaultdict
import numpy as np
from gensim.models import KeyedVectors
from until import check_contain, calc_iou, find_boundary

# Sent_info = namedtuple("Sent_info", "words, chars, pos_tags, entities, nested_flag, tois")
Sent_info = namedtuple("Sent_info", "words, chars, pos_tags, entities")


class Reader():
    def __init__(self, config):
        self.config = config
        self.UNK = "#UNK#"
        self.candidations = {}
        self.infos = {}
        self.max_C = defaultdict(int)
        self.candidation_hit = defaultdict(int)
        self.entity_num = defaultdict(int)
        self.tois_num = defaultdict(int)

    def chunking(self, mode):
        if self.config.if_chunk:  # and mode != "train"
            candidations = []
            with open(self.config.chunk_model_path + f"{mode}_parser.data", "r") as in_file:
                #   open(self.config.chunk_model_path + mode + "error.data", "w") as out_file:
                # all_data = in_file.read().strip().split("\n\n")
                # pre_position = []
                # for data in all_data:
                #     position = []
                #     word_positions = data.strip("\n").split("\n")[1:]
                #     for w_p in word_positions:
                #         position.append(w_p.strip().split(" ")[1])
                #     pre_position.append(position)
                # self.gt_position[mode] = pre_position
                all_data = in_file.read().strip().split("\n\n")
                for i, data in enumerate(all_data):
                    infos = data.split("\n")
                    sentence = infos[0]
                    # out_file.write(sentence + "\n")
                    entities = [] if infos[1] == " " else infos[1].strip().split("|")
                    words = sentence.split()
                    sent = "".join(words)
                    boundary = [len(w) for w in words]
                    for i in range(1, len(boundary)):
                        boundary[i] += boundary[i - 1]

                    p = []
                    for chunk in infos[2:]:
                        start = 0
                        chunk = chunk.replace('``', '"').replace("''", '"')
                        ws = chunk.split()
                        se = "".join(ws)
                        while True:
                            i = sent[start:].find(se)
                            if i == -1:
                                break
                            start += i
                            left = find_boundary(start, boundary)
                            start += len(se)
                            right = find_boundary(start - 1, boundary)
                            if right + 1 - left > self.config.C:
                                continue
                            if " ".join(words[left: right + 1]) != chunk:
                                # print(" ".join(words[left: right + 1]) + "\n" + chunk + "\n\n")
                                continue
                            p.append((left, right + 1))

                    # for entity in entities:
                    #     boxes = entity.split()[0].split(",")
                    #     boxes = (int(boxes[0]), int(boxes[1]))
                    #     if boxes not in p:
                    #         if mode == "train":
                    #             p.append(boxes)
                    #         out_file.write(" ".join(words[boxes[0]: boxes[1]]) + "\n")
                    # candidations.append(sorted(p))
                    candidations.append(p)
                    # out_file.write("\n")
            self.candidations[mode] = candidations
        else:
            self.candidations[mode] = None

    def get_nested_flag(self, entities):
        def flag2index(flag):
            index = 0
            if flag >= 10:
                index += 2
            if flag % 10 != 0:
                index += 1
            return index

        num = len(entities)
        nested_flags = []
        for i in range(num):
            flag = 0
            for j in range(num):
                if i == j:
                    continue
                if check_contain(entities[i][0:2], entities[j][0:2]):
                    flag += 1 if entities[i][2] == entities[j][2] else 10
            # nested_flag.append(flag)
            nested_flags.append(flag2index(flag))
        return nested_flags

    def get_toi(self, index, sent_len, entities, mode):
        gt_boxes = [(e[0], e[1]) for e in entities]
        if self.candidations[mode] is None:
            candidate_boxes = [(start, start + length + 1) for start in range(sent_len) for length in
                               range(min(self.config.C, sent_len - start))]
        else:
            candidate_boxes = self.candidations[mode][index]
        if len(candidate_boxes) == 0:
            return []
        # if mode == "train":
        #     for boxes in gt_boxes:
        #         if boxes not in candidate_boxes:
        #             candidate_boxes.append(boxes)
        candidate_boxes = np.array(sorted(candidate_boxes))

        ious = calc_iou(candidate_boxes, np.array(gt_boxes))
        max_ious = ious.max(axis=1)
        max_idx = ious.argmax(axis=1)
        nested_flag = self.get_nested_flag(entities)

        pos_boxes = []
        neg_boxes = []
        for i in range(candidate_boxes.shape[0]):
            if max_ious[i] == 1:
                self.candidation_hit[mode] += 1
                pos_boxes.append(
                    (candidate_boxes[i, 0], candidate_boxes[i, 1], nested_flag[max_idx[i]], entities[max_idx[i]][2]))
            elif max_ious[i] >= self.config.train_pos_iou_th and mode == "train":
                pos_boxes.append(
                    (candidate_boxes[i, 0], candidate_boxes[i, 1], nested_flag[max_idx[i]], entities[max_idx[i]][2]))
            elif (max_ious[i] < self.config.train_neg_iou_th) or mode != "train":  # and max_ious[i] > 0.1
                neg_boxes.append((candidate_boxes[i, 0], candidate_boxes[i, 1], 3, 0))

        self.tois_num[mode] += (len(pos_boxes) + len(neg_boxes))
        return [pos_boxes, neg_boxes] if mode == "train" else sorted(pos_boxes + neg_boxes)

    def read_file(self, file, mode):
        sent_infos = []
        # with open(file, "r", encoding="utf-8") as f:
        with open(file, "r") as f:
            all_infos = f.read().strip().split("\n\n")
            for i, infos in enumerate(all_infos):
                _infos = infos.strip().split("\n")
                words = _infos[0].split()
                chars = [list(t) for t in words]
                pos_tags = _infos[1].split()
                if len(_infos) == 2:
                    entity_list = []
                    # nested_flags = []
                else:
                    entities = _infos[2].split("|")
                    entity_list = []
                    for entity in entities:
                        positions, label = entity.split()
                        positions = positions.split(",")
                        new_entity = (int(positions[0]), int(positions[1]), label)
                        self.max_C[mode] = max(self.max_C[mode], new_entity[1] - new_entity[0])
                        if new_entity not in entity_list:
                            entity_list.append(new_entity)
                    # nested_flags = self.get_nested_flag(entities)
                # tois = self.get_toi(i, len(words), entity_list, mode)
                # sent_infos.append(Sent_info(words, chars, pos_tags, entity_list, nested_flags, tois))
                self.entity_num[mode] += len(entity_list)
                sent_infos.append(Sent_info(words, chars, pos_tags, entity_list))
        return sent_infos

    def create_dic(self):
        word_set = set()
        char_set = set()
        pos_tag_set = set()
        label_set = set()
        max_word_len = 0
        max_sent_len = 0

        for sent_infos in self.infos.values():
            for sent_info in sent_infos:
                for word in sent_info.chars:
                    max_word_len = max(max_word_len, len(word))
                    for char in word:
                        char_set.add(char)
                max_sent_len = max(max_sent_len, len(sent_info.words))
                for word in sent_info.words:
                    word_set.add(word)
                for pos_tag in sent_info.pos_tags:
                    pos_tag_set.add(pos_tag)
                for entity in sent_info.entities:
                    label_set.add(entity[2])

        self.id2word = sorted(list(word_set))  # [self.UNK] +
        # self.word2id = {v: i for i, v in enumerate(self.id2word)}
        self.word2id = {}
        self.load_vectors_model()
        self.id2char = sorted(list(char_set))  # [self.UNK] +
        self.char2id = {v: i for i, v in enumerate(self.id2char)}
        self.id2pos_tag = sorted(list(pos_tag_set))
        self.pos_tag2id = {v: i for i, v in enumerate(self.id2pos_tag)}
        self.id2label = ["BG"] + sorted((list(label_set)))
        self.label2id = {v: i for i, v in enumerate(self.id2label)}
        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len

    # def mapping(self):
    #     train_word_dic = defaultdict(int)
    #     train_char_dic = defaultdict(int)
    #     train_pos_tag_dic = defaultdict(int)
    #     label_set = set()
    #     max_word_len = 0
    #     max_sent_len = 0
    #
    #     for word in self.infos["train"].chars:
    #         max_word_len = max(max_word_len, len(word))
    #         for char in word:
    #             train_char_dic[char] += 1
    #     max_sent_len = max(max_sent_len, len(self.infos["train"].words))
    #     for word in self.infos["train"].words:
    #         train_word_dic[word] += 1
    #     for pos_tag in self.infos["train"].pos_tags:
    #         train_pos_tag_dic[pos_tag] += 1
    #     for entity in self.infos["train"].entities:
    #         label_set.add(entity[2])
    #
    #
    #     self.id2char, self.char2id = self.create_mapping(train_char_dic)
    #     self.id2word, self.word2id = self.create_mapping(train_word_dic)
    #     self.id2pos_tag, self.pos_tag2id = self.create_mapping(train_pos_tag_dic)
    #     self.id2label = ["BG"] + sorted((list(label_set)))
    #     self.label2id = {v: i for i, v in enumerate(self.id2label)}
    #     self.max_word_len = max_word_len
    #     self.max_sent_len = max_sent_len
    #
    # def create_mapping(self, train_dic):
    #     train_dic[self.UNK] = 100000
    #     sorted_items = sorted(train_dic.items(), key=lambda x: (-x[1], x[0]))
    #     id2item = {i: v[0] for i, v in enumerate(sorted_items)}
    #     item2id = {v: k for k, v in id2item.items()}
    #     return id2item, item2id

    def read_all_data(self):
        for mode in ["debug"]:
        # for mode in ["train", "test", "dev"]:
            self.chunking(mode)
            self.infos[mode] = self.read_file(self.config.data_path + f"/{mode}/{mode}.data", mode)
        self.create_dic()

    def to_batch(self, mode):
        sent_len_dic = defaultdict(list)
        word_dic = defaultdict(list)
        char_dic = defaultdict(list)
        pos_tag_dic = defaultdict(list)
        entity_dic = defaultdict(list)
        char_len_dic = defaultdict(list)
        toi_dic = defaultdict(list)

        sent_len_batches = []
        word_batches = []
        char_batches = []
        char_len_batches = []
        pos_tag_batches = []
        entity_batches = []
        toi_batches = []

        for i, sent_info in enumerate(self.infos[mode]):
            entity_vec = [(e[0], e[1], self.label2id[e[2]]) for e in sent_info.entities]
            # if len(entity_vec) == 0 and mode == "train":
            #     continue
            word_vec = [self.word2id[w] for w in sent_info.words]
            sent_len = len(word_vec)
            if self.config.if_sent_padding:
                pad_id = self.word2id["."]
                word_vec += [pad_id] * (self.max_sent_len - sent_len)

            char_mat = [[self.char2id[c] for c in w] for w in sent_info.chars]
            pad_id = self.char2id["."]
            if self.config.if_sent_padding:
                char_mat += [[pad_id]] * (self.max_sent_len - len(char_mat))
            char_len_vec = [len(w) for w in char_mat]
            char_mat = [w + [pad_id] * (self.max_word_len - len(w)) for w in char_mat]

            pos_tag_vec = [self.pos_tag2id[p] for p in sent_info.pos_tags]
            if self.config.if_sent_padding:
                pad_id = self.pos_tag2id["."]
                pos_tag_vec += [pad_id] * (self.max_sent_len - len(pos_tag_vec))

            word_len = len(word_vec)
            tois = self.get_toi(i, word_len, entity_vec, mode)
            if len(tois) == 0:
                if len(entity_vec) != 0:
                    print(i)
                continue

            sent_len_dic[word_len].append(sent_len)
            word_dic[word_len].append(word_vec)
            char_dic[word_len].append(char_mat)
            char_len_dic[word_len].append(char_len_vec)
            pos_tag_dic[word_len].append(pos_tag_vec)
            entity_dic[word_len].append(entity_vec)
            toi_dic[word_len].append(tois)

        for length in word_dic.keys():
            sent_len_batch = [sent_len_dic[length][i: i + self.config.batch_size] for i in
                              range(0, len(sent_len_dic[length]), self.config.batch_size)]
            word_batch = [word_dic[length][i: i + self.config.batch_size] for i in
                          range(0, len(word_dic[length]), self.config.batch_size)]
            char_batch = [char_dic[length][i: i + self.config.batch_size] for i in
                          range(0, len(char_dic[length]), self.config.batch_size)]
            char_len_batch = [char_len_dic[length][i: i + self.config.batch_size] for i in
                              range(0, len(char_len_dic[length]), self.config.batch_size)]
            pos_tag_batch = [pos_tag_dic[length][i: i + self.config.batch_size] for i in
                             range(0, len(pos_tag_dic[length]), self.config.batch_size)]
            entity_batch = [entity_dic[length][i: i + self.config.batch_size] for i in
                            range(0, len(entity_dic[length]), self.config.batch_size)]
            toi_batch = [toi_dic[length][i: i + self.config.batch_size] for i in
                         range(0, len(toi_dic[length]), self.config.batch_size)]

            sent_len_batches.extend(sent_len_batch)
            word_batches.extend(word_batch)
            char_batches.extend(char_batch)
            char_len_batches.extend(char_len_batch)
            pos_tag_batches.extend(pos_tag_batch)
            entity_batches.extend(entity_batch)
            toi_batches.extend(toi_batch)

        return (sent_len_batches, word_batches, char_batches, char_len_batches, pos_tag_batches, entity_batches, toi_batches)

    def load_vectors_model(self):
        try:
            vector_model = KeyedVectors.load_word2vec_format(self.config.word2vec_path, binary=True)
        except:
            vector_model = KeyedVectors.load_word2vec_format(self.config.word2vec_path, binary=False)

        unk = np.random.uniform(-0.01, 0.01, self.config.word_embedding_size).astype("float32")
        word2vec = [unk]
        id2word = self.id2word.copy()
        for word in id2word:
            try:
                self.word2id[word] = len(word2vec)
                word2vec.append(vector_model[word])
            except:
                try:
                    word2vec.append(vector_model[word.lower()])
                except:
                    self.id2word.remove(word)
                    self.word2id[word] = 0
        self.id2word = [self.UNK] + self.id2word

        # with open(self.config.get_pkl_path("word2vec"), "wb") as f:
        #     pickle.dump(np.array(word2vec), f)
        # print("load vector model form " + self.config.word2vec_path)
