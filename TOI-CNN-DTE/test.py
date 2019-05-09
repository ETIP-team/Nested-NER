import pickle
import torch

from Evaluate import Evaluate
from config import config
from model import TOICNN

with open(config.get_pkl_path("dev"), "rb") as f:
    sent_len_batches, word_batches, char_batches, char_len_batches, pos_tag_batches, entity_batches, toi_batches = pickle.load(f)

config.if_detail = False
config.if_output = False
config.if_filter = True
config.score_th = 0.5
config.if_nms = False
config.if_nms_with_nested = False
config.nms_th = 0.6
misc_config = pickle.load(open(config.get_pkl_path("config"), "rb"))
config.load_config(misc_config)

for e in range(50, 101):
    model_path = config.get_model_path() + f"epoch{e}.pth"
    print("load model from " + model_path)
    ner_model = TOICNN(config)
    ner_model.load_state_dict(torch.load(model_path))
    if config.if_gpu and torch.cuda.is_available():
        ner_model = ner_model.cuda()
    evaluate = Evaluate(ner_model, config)

    evaluate.get_f1(zip(sent_len_batches, word_batches, char_batches, char_len_batches, pos_tag_batches, entity_batches, toi_batches))
    print("\n\n")
