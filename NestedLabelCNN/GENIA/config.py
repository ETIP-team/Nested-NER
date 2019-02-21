# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2018-10-03

K_FOLD = 1
LABEL = ['DNA', 'RNA', 'cell_type', 'protein', 'cell_line', 'bg']
# LABEL = ["PER", "ORG", "GPE", "LOC", "FAC", "VEH", "WEA"]

# NESTED_LABEL = ['Not Nested', 'SameTypeNested', 'DifferentTypeNested', 'MultipleTypeNested']
#
NESTED_LABEL = ['NotNested', 'SameTypeNested']

NESTED_CLASSES_NUM = len(NESTED_LABEL)

TRAIN_FLAG = False
SW_TECH_FLAG = True

# WORD_SEGMENTATION_METHOD = "genia"  # "nlpir"

# CLASS_WEIGHT = [1, 20, 20, 20]
CLASS_WEIGHT = [1, 100]
MAX_CLS_LEN = 32
SENTENCE_LENGTH = 200

# Model  Setting
WORD_EMBEDDING_DIM = 200
LEARNING_RATE = 1e-4
KERNEL_LENGTH = 2
L2_BETA = 0.001
FEATURE_MAPS_NUM = 100

POOLING_SIZE = 2  # to be declared.

# Training Setting
TH_IOU_TRAIN = 0.6
TH_IOU_NMS = 0.05
TH_IOU_TEST = 0.6
TH_SCORE = 0.6

BIO_NLP_VEC = "../../model/word_vector_model/bio_nlp_vec.tar/bio_nlp_vec/PubMed-shuffle-win-2.bin"
# BIO_NLP_VEC = "../../model/word_vector_model/wikipedia-pubmed-and-PMC-w2v.bin"

# TEST_PATH = "result/CNN_genia_sw/"

TRAIN_FILE_PATH = "../../dataset/train/train.data"
TEST_FILE_PATH = "../../dataset/test/test.data"
