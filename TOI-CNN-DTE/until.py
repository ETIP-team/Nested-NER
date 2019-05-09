from random import sample

import numpy as np
import torch
from torch import optim


class opt:
    def __init__(self, config, parameters):
        if config.opt == "Adam":
            self.optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.l2)
        self._step = 0

        self.if_warm_up = config.if_warm_up
        self.warm_up = config.warm_up
        self.factor = config.factor
        self.model_size = config.model_size
        self._rate = 0

    def step(self):
        self._step += 1
        if self.if_warm_up:
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate
            self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warm_up ** (-1.5)))


def create_opt(parameters, config):
    if config.opt == "Adam":
        # return optim.Adam(parameters, lr=config.lr, weight_decay=config.l2)
        return optim.Adam(parameters, lr=config.lr)
    return None


def adjust_learning_rate(optimizer, decay_rate=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate
    print(f"lr change to {optimizer.param_groups[0]['lr']}")


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # vec is only 1d vec
    # return the argmax as a python int
    _, idx = torch.max(vec, 0)
    return to_scalar(idx)


def log_sum_exp(vec_list):
    """
    Compute log sum exp in a numerically stable way for the forward algorithm
    vec is n * m, norm in row
    return n * 1
    """
    if type(vec_list) == list:
        mat = torch.stack(vec_list, 1)
    else:
        mat = vec_list
    row, column = mat.size()
    ret_l = []
    for i in range(row):
        vec = mat[i]
        max_score = vec[argmax(vec)]
        max_score_broadcast = max_score.view(-1).expand(1, vec.size()[0])
        ret_l.append(max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast))))
    return torch.cat([item.expand(1) for item in ret_l], 0)


def check_contain(p1, p2):
    """
    if p1 contain p2
        return True
    """
    if type(p1) != "tuple":
        p1 = tuple(p1)
    if type(p2) != "tuple":
        p2 = tuple(p2)
    left = min(p1[0], p2[0])
    right = max(p1[1], p2[1])
    union = (left, right)
    # return p1 == union or p2 == union
    return p1 == union


def calc_iou(b1, b2):
    assert b1.size > 0
    if b2.size == 0:
        return np.zeros((b1.shape[0], 1))
    l_max = np.maximum(b1[:, 0].reshape((-1, 1)), b2[:, 0].reshape((1, -1)))
    l_min = np.minimum(b1[:, 0].reshape((-1, 1)), b2[:, 0].reshape((1, -1)))
    r_max = np.maximum(b1[:, 1].reshape((-1, 1)), b2[:, 1].reshape((1, -1)))
    r_min = np.minimum(b1[:, 1].reshape((-1, 1)), b2[:, 1].reshape((1, -1)))

    inter = np.maximum(0, r_min - l_max)
    union = r_max - l_min
    return inter / union


def select_toi(toi_batch, toi_num, pos_rate, mode):
    new_batch = []
    # if True:
    if type(toi_batch[0][0]) == list:
        for item in toi_batch:
            toi = []
            pos_num = len(item[0])
            neg_num = len(item[1])
            if pos_num > 0:
                if mode == "train":
                    # pos_num = len(item[0])
                    # pos_num = min(toi_num * pos_rate, len(item[0]))
                    # random_index = np.random.choice(len(item[0]), int(toi_num * pos_rate))
                    # toi.extend(list(np.array(item[0])[random_index]))
                    # toi.extend(sample(item[0], int(pos_num)))
                    toi.extend(item[0])
                else:
                    toi.extend(item[0])
            if neg_num > 0:
                if mode == "train":
                    # neg_sample_num = min(neg_num, max(4 * pos_num, (neg_num + 1) // 2))
                    neg_sample_num = (neg_num + 1) // 2
                    # neg_num = len(item[1])
                    # neg_num = min(toi_num * (1 - pos_rate), len(item[1]))
                    # toi.extend(sample(item[1], int(neg_num)))
                    toi.extend(sample(item[1], neg_sample_num))
                    # random_index = np.random.choice(len(item[1]), int(toi_num * (1 - pos_rate)))
                    # toi.extend(list(np.array(item[1])[random_index]))
                else:
                    toi.extend(item[1])
            new_batch.append(toi)
    else:
        new_batch = toi_batch

    tois = [np.array([(t[0], t[1]) for t in sent]) for sent in new_batch]
    nested_flags = [np.array([t[2] for t in sent]) for sent in new_batch]
    labels = [np.array([t[3] for t in sent]) for sent in new_batch]

    return tois, nested_flags, labels


def find_boundary(start, boundary):
    for i, b in enumerate(boundary):
        if b > start:
            return i
    return len(boundary)


def sequent_mask(sent_len, max_sent_len):
    mask = []
    for l in sent_len:
        m = [1] * l
        m.extend([0] * (max_sent_len - l))
        mask.append(m)
    return torch.from_numpy(np.array(mask)) == 1


def batch_split(score, toi_section):
    score = score.cpu()
    result = [torch.index_select(score, 0, torch.arange(0, toi_section[0]).long())]
    for i in range(len(toi_section) - 1):
        result.append(torch.index_select(score, 0, torch.arange(toi_section[i], toi_section[i + 1]).long()))
    return result


if __name__ == "__main__":
    print(check_contain((3, 7), (5, 7)))
    print(check_contain((0, 5), (0, 6)))
