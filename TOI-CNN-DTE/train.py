import pickle
import time
from random import shuffle
import numpy as np
import torch
from torch.autograd import Variable

from Evaluate import Evaluate
from config import config
from model import TOICNN
from until import create_opt, select_toi, sequent_mask, adjust_learning_rate

with open(config.get_pkl_path("train"), "rb") as f:
    train_sent_len_batches, train_word_batches, train_char_batches, train_char_len_batches, train_pos_tag_batches, train_entity_batches, train_toi_batches = pickle.load(f)
with open(config.get_pkl_path("dev"), "rb") as f:
    dev_sent_len_batches, dev_word_batches, dev_char_batches, dev_char_len_batches, dev_pos_tag_batches, dev_entity_batches, dev_toi_batches = pickle.load(f)


misc_config = pickle.load(open(config.get_pkl_path("config"), "rb"))
config.load_config(misc_config)

ner_model = TOICNN(config)
ner_model.load_vector()
# config.lambd = 0
# ner_model.load_state_dict(torch.load(config.get_model_path() + "epoch61.pth"))
# config.lambd = 1

if config.if_gpu and torch.cuda.is_available(): ner_model = ner_model.cuda()
evaluate = Evaluate(ner_model, config)

parmeters = filter(lambda p: p.requires_grad, ner_model.parameters())
optimizer = create_opt(parmeters, config)
# optimizer = torch.optim.Adam(ner_model.parameters(), lr=config.lr)

best_model = None
best_per = 0
pre_loss = 100000
train_all_batches = list(zip(train_sent_len_batches, train_word_batches, train_char_batches, train_char_len_batches, train_pos_tag_batches, train_entity_batches, train_toi_batches))

for e_ in range(config.epoch):
    print("Epoch:", e_ + 1)
    cur_time = time.time()
    if config.if_shuffle:
        shuffle(train_all_batches)

    losses = []
    cls_losses = []
    nested_losses = []
    for sent_len_batch, word_batch, char_batch, char_len_batch, pos_tag_batch, entity_batch, toi_batch in train_all_batches:
        batch_num = len(word_batch)
        max_sent_len = len(word_batch[0])

        word_batch_var = Variable(torch.LongTensor(np.array(word_batch)))
        char_batch_var = Variable(torch.LongTensor(np.array(char_batch)))
        pos_tag_batch_var = Variable(torch.LongTensor(np.array(pos_tag_batch)))
        mask_batch_var = sequent_mask(sent_len_batch, max_sent_len)
        # toi_random_batch, nested_random_batch, label_random_batch = select_toi(toi_batch, config.toi_num, config.pos_rate, "train")
        toi_random_batch, nested_random_batch, label_random_batch = select_toi(toi_batch, config.toi_num, config.pos_rate, "test")
        gold_label_vec = Variable(torch.LongTensor(np.hstack(label_random_batch)))
        gold_nested_vec = Variable(torch.LongTensor(np.hstack(nested_random_batch)))
        if config.if_gpu:
            word_batch_var = word_batch_var.cuda()
            char_batch_var = char_batch_var.cuda()
            pos_tag_batch_var = pos_tag_batch_var.cuda()
            mask_batch_var = mask_batch_var.cuda()
            gold_label_vec = gold_label_vec.cuda()
            gold_nested_vec = gold_nested_vec.cuda()

        ner_model.train()
        optimizer.zero_grad()
        cls_s, nested_s, _ = ner_model(mask_batch_var, word_batch_var, char_batch_var, char_len_batch, pos_tag_batch_var, toi_random_batch)
        loss, cls_loss, nested_loss = ner_model.calc_loss(cls_s, nested_s, gold_label_vec, gold_nested_vec)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(ner_model.parameters(), 3, norm_type=2)
        optimizer.step()

        losses.append(loss.data.cpu().numpy())
        cls_losses.append(cls_loss.data.cpu().numpy())
        nested_losses.append(nested_loss.data.cpu().numpy())

    sub_loss = np.mean(losses)
    print(f'Avg loss = {sub_loss:.4f}; cls_loss = {np.mean(cls_losses):.4f}, nested_loss = {np.mean(nested_losses):.4f}')
    print(f"Training step took {time.time() - cur_time:.0f} seconds")
    if e_ >= 0:
        print("Dev:")
        cls_f1, nested_f1 = evaluate.get_f1(zip(dev_sent_len_batches, dev_word_batches, dev_char_batches, dev_char_len_batches, dev_pos_tag_batches, dev_entity_batches, dev_toi_batches))
        # if cls_f1 > best_per:
        if True:
            best_per = cls_f1
            model_path = config.get_model_path() + f"epoch{e_ + 1}.pth"
            torch.save(ner_model.state_dict(), model_path)
            print("model save in " + model_path + '\n\n')
    # if e_ > 4:
    #     ner_model.config.lambd = 0.5
    # if e_ % 30 == 0 and e_ > 0:
    #     adjust_learning_rate(optimizer)
    if sub_loss >= pre_loss:
        adjust_learning_rate(optimizer)

    pre_loss = sub_loss
