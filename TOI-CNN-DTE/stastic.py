import pickle
from collections import defaultdict

import numpy as np
from gensim.models import KeyedVectors

import config as cfg
from until import check_contain, calc_iou


def stastic_len(data_set, mode):
    all_info = open(f'./dataset/{data_set}/{mode}/{mode}.data', 'r').read()
    all_info = all_info.strip('\n').split('\n\n')
    max_sentences_len = 0
    label_static = {}
    len_count = {}
    for item in all_info:
        infos = item.strip('\n').split('\n')

        sentences_len = len(infos[0].split(' '))
        max_sentences_len = max(sentences_len, max_sentences_len)

        entitys = infos[2].split('|') if len(infos) == 3 else []
        for entity in entitys:
            RL, cls = entity.split(' ')
            right, left = RL.split(',')[0: 2]
            label_len = int(left) - int(right)
            if label_len not in len_count:
                len_count[label_len] = 0
            len_count[label_len] += 1
            # if label_len > 20:
            #     print(entity)
            if cls not in label_static:
                label_static[cls] = {'max_len': label_len, 'num': 1, 'avg_len': label_len,
                                     'SLR': label_len / sentences_len}
            else:
                label_static[cls]['max_len'] = max(label_len, label_static[cls]['max_len'])
                label_static[cls]['num'] += 1
                label_static[cls]['avg_len'] += label_len
                label_static[cls]['SLR'] += label_len / sentences_len
    # print('max_sentences_len:', max_sentences_len)
    # for k, v in label_static.items():
    #     v['avg_len'] = v['avg_len'] / v['num']
    #     v['SLR'] = v['SLR'] / v['num']
    #     print(k)
    #     print(v)
    print(sorted(len_count.items(), key=lambda x: x[0]))


def correct():
    # writefile = open('./dataset/train/train.data', 'w')
    # all_info = open('./GENIA/train.data', 'r').read()
    writefile = open('./dataset/test/test.data', 'w')
    all_info = open('./GENIA/test.data', 'r').read()
    all_info = all_info.strip('\n').split('\n\n')
    for item in all_info:
        infos = item.strip('\n').split('\n')
        if len(infos) != 3:
            print(infos)
            continue
        writefile.write('\n'.join([infos[0], infos[2], '\n']))
    writefile.close()


def words_count():
    train_info = open('./dataset/ACE/extent/train.data', 'r').read().strip('\n').split('\n\n')
    test_info = open('./dataset/ACE/extent/test.data', 'r').read().strip('\n').split('\n\n')
    wordlist = []
    for info in train_info:
        words = info.split('\n')[0].split(' ')
        for word in words:
            if word not in wordlist:
                wordlist.append(word)
    for info in test_info:
        words = info.split('\n')[0].split(' ')
        for word in words:
            if word not in wordlist:
                wordlist.append(word)
    print(len(wordlist))


def count_neg(neg):
    datatype = ['train', 'test']
    writefile = open('./count/' + neg + '_count.data', 'w')
    for type in datatype:
        train_info = open(type.join(['./dataset/', '/', '.data']), 'r').read().strip('\n').split('\n\n')
        writefile.write('-----------------------------------------\n' + type + 'data:\n')
        count = {}
        num = 0
        for info in train_info:
            infos = info.split('\n')
            words = infos[0].split(' ')
            range = infos[1].split('|')
            for r in range:
                RL, Cls = r.split(' G#')
                left, right = RL.split(',')
                if neg in words[int(left): int(right)]:
                    num += 1
                    writefile.write(' '.join(words[int(left): int(right)]) + '\n')
                    length = int(right) - int(left)
                    if length not in count:
                        count[length] = 0
                    count[length] += 1
        sorted_count = sorted(count.items(), key=lambda item: item[0])
        writefile.write('num: ' + str(num) + '\n(len,count) ' + str(sorted_count) + '\n')
    writefile.close()


def nested_flag_count(pattern):
    def flag2index(flag):
        index = 0
        if flag >= 10:
            index += 2
        if flag % 10 != 0:
            index += 1
        return index

    def get_nested_flag(gt_boxes, gt_classes):
        num = len(gt_boxes)
        nested_flag = []
        for i in range(num):
            flag = 0
            for j in range(num):
                if i == j:
                    continue
                if check_contain(gt_boxes[i], gt_boxes[j]):
                    flag += 1 if gt_classes[i] == gt_classes[j] else 10
            # nested_flag.append(flag2index(flag))
            nested_flag.append(flag)
            # if flag == 31:
            #     wait = True
        return nested_flag

    pkl_file_path = pattern.join(['dataset/', '/ace_', '_word2vec_max_cls_len{}.pkl'.format(cfg.MAX_CLS_LEN)])
    all_data = pickle.load(open(pkl_file_path, "rb"))
    write_file = open('./nested_count.data', 'w')
    sample_num = len(all_data)
    flags = {}
    for sample_index in range(sample_num):
        if_write = False
        gt_boxes = all_data[sample_index]["ground_truth_bbox"]
        gt_boxes = np.array([list(item) for item in gt_boxes])
        gt_classes = all_data[sample_index]["ground_truth_cls"]
        nested_flag = get_nested_flag(gt_boxes, gt_classes)
        for flag in nested_flag:
            if flag not in flags:
                flags[flag] = 0
            flags[flag] += 1
            if flag not in [0, 1, 10]:
                if_write = True
        if if_write:
            write_file.write(all_data[sample_index]['str'] + '\n')
            write_file.write('|'.join(
                [f'{gt_boxes[i][0]},{gt_boxes[i][1]} G#{cfg.LABEL[gt_classes[i] - 1]}' + (f' :F{nested_flag[i]}' if
                 nested_flag[i] not in [0, 1, 10] else '') for i in range(len(gt_classes))]))
            write_file.write('\n\n')
    write_file.close()
    print(pattern + ':')
    # print(sorted(flags.items(), key=lambda item: item[0]))
    print(flags)


def nested_iou_count(data_set, mode):
    path = f"./dataset/{data_set}/{mode}/{mode}.data"
    iou_dic = {}
    with open(path, "r") as infile:
        all_sample = infile.read().strip().split("\n\n")
        for sample in all_sample:
            infos = sample.strip().split("\n")
            if len(infos) == 2:
                continue
            labels = infos[2].split("|")
            gt_bos = []
            for label in labels:
                lr = label.split(" ")[0].split(",")[:2]
                gt_bos.append([int(item) for item in lr])

            ious = calc_iou(np.array(gt_bos), np.array(gt_bos))
            gt_num = len(gt_bos)
            for i in range(gt_num):
                # if gt_bos[i][1] - gt_bos[i][0] > 6:
                #     continue
                for j in range(gt_num):
                    if i == j:
                        continue
                    if check_contain(gt_bos[i], gt_bos[j]):
                        iou = round(ious[i, j], 2)
                        if iou == 1:
                            wait = True
                        if iou not in iou_dic:
                            iou_dic[iou] = 0
                        iou_dic[iou] += 1
    print(sorted(iou_dic.items(), key=lambda x: x[0]))


def unk_count(word_vector_model):
    train_word_num = 0
    word_dic = {}
    vector_missing_train = 0
    with open("./dataset/train/train.data", "r") as train_file:
        train_data = train_file.read().strip().split('\n\n')
        for data in train_data:
            words = data.strip().split("\n")[0].split(" ")
            for word in words:
                train_word_num += 1
                if word not in word_dic:
                    word_dic[word] = 1
                else:
                    word_dic[word] += 1
                if word not in word_vector_model:  #  and word.lower() not in word_vector_model
                    vector_missing_train += 1
    frequent_count = [0] * 21
    for v in word_dic.values():
        if v <= 20:
            frequent_count[v] += 1

    print("train ")
    print(f"word num: {train_word_num}")
    print(f"word kinds: {len(word_dic)}")
    print(f"frequent words count: {frequent_count[1:]}")
    print(f"vector missing: {vector_missing_train}, rate: {vector_missing_train / train_word_num}")

    test_word_num = 0
    unk_num = 0
    vector_missing_test = 0
    both_missing_num = 0
    with open("./dataset/test/test.data", "r") as test_file:
        test_data = test_file.read().strip().split("\n\n")
        for data in test_data:
            words = data.strip().split("\n")[0].split(" ")
            for word in words:
                test_word_num += 1
                if word not in word_vector_model:  #  and word.lower() not in word_vector_model
                    vector_missing_test += 1
                    if word not in word_dic:
                        both_missing_num += 1
                if word not in word_dic:
                    unk_num += 1
    print("\ntest")
    print(f"word num: {test_word_num}")
    print(f"unk_num: {unk_num}, rate: {unk_num / test_word_num}")
    print(f"vector missing: {vector_missing_test}, rate: {vector_missing_test / test_word_num}")
    print(f"both missing: {both_missing_num}, rate: {both_missing_num / test_word_num}")


def label_count(mode):
    dict = defaultdict(int)
    with open(f"./dataset/{mode}/{mode}.data", "r") as file:
        _data = file.read().strip().split('\n\n')
        for data in _data:
            infos = data.strip().split("\n")
            if len(infos) < 3:
                continue
            entities = infos[2].split("|")
            for entity in entities:
                dict[entity.split()[1]] += 1
    print(dict)


def pos_chunk():
    def read(mode):
        _set = set()
        with open(f"./dataset/{mode}/{mode}.data", "r") as file:
            _data = file.read().strip().split("\n\n")
            for data in _data:
                infos = data.strip().split("\n")
                if len(infos) < 3:
                    continue
                pos_tag = infos[1].split(" ")
                entities = infos[2].split("|")
                for entity in entities:
                    left, right = entity.split()[0].split(",")[0: 2]
                    _set.add(" ".join(pos_tag[int(left): int(right)]))
        return _set
    train_set = read("train")
    test_set = read("test")
    dev_set = read("dev")
    FN, count = 0, 0
    for item in test_set:
        count += 1
        if item not in train_set:
            FN += 1
    print(f"test.data: FN = {FN}, count = {count}, recall = {1 - FN/count:.4f}")
    FN, count = 0, 0
    for item in dev_set:
        count += 1
        if item not in train_set:
            FN += 1
    print(f"dev.data: FN = {FN}, count = {count}, recall = {1 - FN/count:.4f}")


if __name__ == '__main__':
    for data_set in ["GENIA", "ACE2004", "ACE"]:
        print("data set: " + data_set)
        for mode in ["train", "test", "dev"]:
            print(mode + ":")
            # stastic_len(data_set, mode)
            nested_iou_count(data_set, mode)
        print("\n\n")
    # correct()
    # words_count()
    # count_neg('complex')
    # nested_flag_count('train')
    # nested_iou_count('test')

    # KeyedVectors.load_word2vec_format(cfg.BIO_NLP_VEC, binary=False)  glove
    # KeyedVectors.load_word2vec_format(cfg.BIO_NLP_VEC, binary=True)   wiki
    # unk_count(KeyedVectors.load_word2vec_format(cfg.BIO_NLP_VEC, binary=True))

    # label_count("test")
    # pos_chunk()
