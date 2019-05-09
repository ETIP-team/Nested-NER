#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@file: transfor.py 
@time: 2019/04/02
"""
from collections import defaultdict
import numpy as np

max_depth = 0

def if_overlap(a, b):
    left = min(a[0], b[0])
    right = max(a[1], b[1])
    return (right - left) < (a[1] + b[1] - a[0] - b[0])

def get_label(entities, sent_len, max_depth=6):
    ents  = []
    for entity in entities:
        position, label = entity.split(" ")
        left, right = position.split(",")[0: 2]
        ents.append((int(left), int(right), label))

    ents = sorted(ents, key=lambda x:(x[0], -x[1], x[2]))
    layers = {}
    depth = -1
    for ent in ents:
        d = 0
        while d <= depth:
            flag = False
            for e in layers[d]:
                if if_overlap(e, ent):
                    flag = True
                    break
            if flag:
                d += 1
            else:
                layers[d].append(ent)
                break
        if d > depth:
            layers[d] = [ent]
            depth += 1

    labels = [["O" for j in range(sent_len)] for i in range(max_depth)]
    for depth, layer in layers.items():
        for left, right, label in layer:
            labels[depth][left] = "B-" + label
            labels[depth][left + 1: right] = ["I-" + label] * (right - left - 1)
    return np.array(labels).transpose().tolist()

def transfor(mode):
    read_path = mode.join(['dataset/ACE/', '/', '.data'])
    write_path = mode.join(['dataset/ACE/', '/layer_', '.data'])
    with open(read_path, "r") as read_file, open(write_path, "w") as write_file:
        all_data = read_file.read().strip().split("\n\n")
        for sample in all_data:
            infos = sample.strip().split("\n")
            words = infos[0].split()
            sent_len = len(words)
            pos_tags = infos[1].split()
            entities = infos[2].split("|") if len(infos) == 3 else []
            labels = get_label(entities, sent_len)

            for i in range(sent_len):
                write_file.write(" ".join([words[i], " ".join(labels[i])]) + "\n")
            write_file.write("\n")

if __name__ == "__main__":
    for mode in ["train", "test", "dev"]:
        transfor(mode)