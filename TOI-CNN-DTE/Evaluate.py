import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable

from until import calc_iou, check_contain, select_toi, sequent_mask, batch_split


class Evaluate():
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.outfile = open(config.get_result_path(), "w") if config.if_output else None
        self.classes_num = len(self.model.config.id2label) - 1
        self.nested_falg_num = len(self.model.config.nested_flag)

    def get_f1(self, batch_zip):
        self.init()

        for sent_len_batch, word_batch, char_batch, char_len_batch, pos_tag_batch, entity_batch, toi_batch in batch_zip:
            word_batch_var = Variable(torch.LongTensor(np.array(word_batch)))
            char_batch_var = Variable(torch.LongTensor(np.array(char_batch)))
            pos_tag_batch_var = Variable(torch.LongTensor(np.array(pos_tag_batch)))
            mask_batch_var = sequent_mask(sent_len_batch, len(word_batch[0]))
            _toi_batch, nested_batch, label_batch = select_toi(toi_batch, None, None, "test")
            if self.model.config.if_gpu:
                word_batch_var = word_batch_var.cuda()
                char_batch_var = char_batch_var.cuda()
                pos_tag_batch_var = pos_tag_batch_var.cuda()
                mask_batch_var = mask_batch_var.cuda()

            self.model.eval()
            cls_s, nested_s, toi_section = self.model(mask_batch_var, word_batch_var, char_batch_var, char_len_batch,
                                                      pos_tag_batch_var, _toi_batch)
            cls_s = batch_split(cls_s, toi_section)
            nested_s = batch_split(nested_s, toi_section)
            for i in range(len(word_batch)):
                pred_entities = self.predict(_toi_batch[i], cls_s[i], nested_s[i], entity_batch[i],
                                             word_batch_var[i][:sent_len_batch[i]], nested_batch[i])
                if self.config.if_detail:
                    self.evaluate_detail(entity_batch[i], pred_entities)
                else:
                    self.evaluate(entity_batch[i], pred_entities)

        return self.calc_f1()

    def calc_f1(self):
        if self.config.if_detail:
            print("nested flag:")
            print(self.nested_confusion_matrix.astype(np.int32))
            df, nested_f1 = self.get_count_dataframe(self.nested_confusion_matrix, self.model.config.nested_flag)
            print(df)
            print("\n\n")
            print("toi\t", self.toi_F)
            print("after filter\t", self.filter_F)
            print("after nms\t", self.nms_F)
            print(self.confusion_matrix.astype(np.int32))
            df, cls_f1 = self.get_count_dataframe(self.confusion_matrix, list(self.model.config.id2label)[1:])
            print(df)
            print("contain:")
            print(str(self.contain_matrix.astype(np.int32)))
        else:
            print("nested flag:")
            # print(self.nested_confusion_matrix.astype(np.int32))
            df, nested_f1 = self.get_count_dataframe(self.nested_confusion_matrix, self.model.config.nested_flag)
            print(f"{nested_f1:.4f}")
            print("\n\n")
            print(self.pred_all, self.pred, self.recall_all, self.recall)
            precision = self.pred / self.pred_all if self.pred_all != 0 else 0
            recall = self.recall / self.recall_all if self.recall_all != 0 else 0
            cls_f1 = (2 * precision * recall) / (precision + recall) if precision * recall != 0 else 0
            print(f"Precision {precision:.4f}, Recall {recall:.4f}, F1 {cls_f1:.4f}")
        return cls_f1, nested_f1

    def predict(self, tois, cls_s, nested_s, entities, word_ids, gold_nested):
        words = [self.model.config.id2word[word] for word in word_ids]
        self.write_result(None, None, None, None, entities, words, 'begin')

        if cls_s.shape[0] == 0:
            self.write_result(None, None, None, None, None, None, 'end')
            return []

        cls_score = F.softmax(cls_s, dim=1).data.cpu().numpy()
        nested_score = F.softmax(nested_s, dim=1).data.cpu().numpy()
        cls = np.argmax(cls_score, axis=1)
        cls_score = np.max(cls_score, axis=1)
        nested_flag = np.argmax(nested_score, axis=1)

        entity_index = np.where(cls != 0)[0]
        tois_ = tois[entity_index]
        cls = cls[entity_index]
        cls_score = cls_score[entity_index]
        nested_flag = nested_flag[entity_index]
        gold_nested_ = gold_nested[entity_index]
        self.renewal_F_result(tois_, cls, entities, self.toi_F)
        self.write_result(tois_, cls, nested_flag, cls_score, entities, words, 'TOI')

        if self.config.if_filter:
            filter_index = np.where(cls_score >= self.config.score_th)[0]
            tois_ = tois_[filter_index]
            cls = cls[filter_index]
            cls_score = cls_score[filter_index]
            nested_flag = nested_flag[filter_index]
            gold_nested_ = gold_nested_[filter_index]
            self.renewal_F_result(tois_, cls, entities, self.filter_F)
            self.write_result(tois_, cls, nested_flag, cls_score, entities, words, 'After filter')

        self.evaluate_nested(nested_flag, gold_nested_)

        if self.config.if_nms:
            nms_index = self.nms(tois_, cls, cls_score, nested_flag)
            tois_ = tois_[nms_index]
            cls = cls[nms_index]
            cls_score = cls_score[nms_index]
            nested_flag = nested_flag[nms_index]
            self.renewal_F_result(tois_, cls, entities, self.nms_F)
            self.write_result(tois_, cls, nested_flag, cls_score, entities, words, 'After NMS')

        self.write_result(None, None, None, None, None, None, 'end')
        return [(tois_[i][0], tois_[i][1], cls[i]) for i in range(len(tois_))]

    def write_result(self, tois, cls, nested_flag, cls_score, gold_entities, words, type):
        if not self.config.if_output:
            return
        if type == 'begin':
            self.outfile.write(' '.join(words) + '\ngt: ')
            for gt in gold_entities:
                self.outfile.write(self.model.config.id2label[gt[2]] + ' [' + ' '.join(words[gt[0]: gt[1]]) + ']  ')
            self.outfile.write('\n')
            return
        if type == 'end':
            self.outfile.write('--------------------------------------------------------\n\n')
            return
        pre_is_right = np.zeros(len(tois))
        for i in range(len(tois)):
            pre = (tois[i][0], tois[i][1], cls[i])
            if pre in gold_entities:
                pre_is_right[i] = 1
        pre_is_right = pre_is_right.astype(np.int32)
        r_e = ["F", "T"]
        if type == "1":
            self.outfile.write('--------------------------------------------------------\n' + type + '\n')
            for i in range(len(tois)):
                if pre_is_right[i] == 1:
                    self.outfile.write('\t'.join([r_e[pre_is_right[i]], self.model.config.id2label[cls[i]],
                                                  self.model.config.nested_flag[nested_flag[i]], str(tois[i]),
                                                  str(np.round(cls_score[i], 3)),
                                                  '[' + ' '.join(words[tois[i][0]: tois[i][1]]) + ']']) + '\n')
        else:
            self.outfile.write('--------------------------------------------------------\n' + type + '\n')
            for i in range(len(tois)):
                self.outfile.write('\t'.join([r_e[pre_is_right[i]], self.model.config.id2label[cls[i]],
                                              self.model.config.nested_flag[nested_flag[i]], str(tois[i]),
                                              str(np.round(cls_score[i], 3)),
                                              '[' + ' '.join(words[tois[i][0]: tois[i][1]]) + ']']) + '\n')
            self.outfile.write('\n')

    def renewal_F_result(self, tois, pred_cls, gold_entities, dic):
        pred_entities = [(tois[i][0], tois[i][1], pred_cls[i]) for i in range(len(tois))]

        for pre in pred_entities:
            if pre not in gold_entities:
                dic["FP"][pre[2] - 1] += 1
        for gt in gold_entities:
            if gt not in pred_entities:
                dic["FN"][gt[2] - 1] += 1

    def evaluate_detail(self, gold_entities, pred_entities):
        for gt in gold_entities:
            if gt not in pred_entities:
                self.confusion_matrix[gt[2] - 1, self.classes_num] += 1  # FN
        right_pres = []
        for pre in pred_entities:
            if pre in gold_entities:
                right_pres.append(pre)
                self.confusion_matrix[pre[2] - 1, pre[2] - 1] += 1  # TP
            else:
                self.confusion_matrix[self.classes_num, pre[2] - 1] += 1  # FP

        # 更新right_pre_contain关系矩阵
        for i in range(len(right_pres)):
            for j in range(i + 1, len(right_pres)):
                if check_contain(right_pres[i][0:2], right_pres[j][0:2]):
                    self.contain_matrix[right_pres[i][2] - 1, right_pres[j][2] - 1] += 1
                elif check_contain(right_pres[j][0:2], right_pres[i][0:2]):
                    self.contain_matrix[right_pres[j][2] - 1, right_pres[i][2] - 1] += 1

    def evaluate(self, gold_entities, pred_entities):
        self.recall_all += len(gold_entities)
        self.pred_all += len(pred_entities)

        for gt in gold_entities:
            if gt in pred_entities:
                self.recall += 1

        for pre in pred_entities:
            if pre in gold_entities:
                self.pred += 1

    def evaluate_nested(self, pred_nested, gold_nested):
        for i in range(pred_nested.size):
            self.nested_confusion_matrix[gold_nested[i], pred_nested[i]] += 1

    def nms(self, bboxs, cls, scores, nested_flag):
        def renewal_candidates(candidates, index, bboxs, contain_flag):
            if not self.config.if_nms_with_nested:
                contain_flag = 3
            for i in range(len(candidates)):
                if candidates[i]:
                    iou = calc_iou(np.array([bboxs[index]]), np.array([bboxs[i]]))[0][0]
                    if check_contain(bboxs[index], bboxs[i]):
                        if contain_flag == 0 or (iou >= 0.68 and cls[index] == cls[i]):
                            candidates[i] = False
                        elif contain_flag == 2 and cls[index] == cls[i]:
                            candidates[i] = False
                    elif iou >= self.config.nms_th and cls[index] == cls[i]:
                        candidates[i] = False
            return candidates

        roi_num = scores.shape[0]
        # sorted_score_indexes = np.argsort(bboxs[:, 1] - bboxs[:, 0])[::-1]  # sort by length.
        # sentence_length = len(info['gt_str'])
        # bboxs_length = bboxs[:, 1] - bboxs[:, 0] + 1
        # SLR = bboxs_length / sentence_length
        # sorted_length_indexes = np.argsort(bboxs_length)[::-1]
        # sorted_score_indexes = np.argsort(scores)[::-1]
        #
        # lenth_rank = np.zeros(roi_num)
        # score_rank = np.zeros(roi_num)
        # for i in range(roi_num):
        #     lenth_rank[sorted_length_indexes[i]] = i
        #     score_rank[sorted_score_indexes[i]] = i
        # mix_rank = [lenth_rank[i] * SLR[i] + score_rank[i] * (1 - SLR[i]) for i in range(roi_num)]
        # sorted_mix_indexes = np.argsort(mix_rank)
        #
        # bboxs = bboxs[sorted_mix_indexes, :]
        sorted_score_indexes = np.argsort(scores)[::-1]
        bboxs_sort = bboxs[sorted_score_indexes, :]
        nested_flag_sort = nested_flag[sorted_score_indexes]

        result_index = []
        candidates = [True] * roi_num

        for i in range(roi_num):
            if candidates[i]:
                result_index.append(sorted_score_indexes[i])
                candidates = renewal_candidates(candidates, i, bboxs_sort, nested_flag_sort[i])
        result_index = np.array(result_index, dtype=np.int32)
        return result_index

    def init(self):
        self.confusion_matrix = np.zeros((self.classes_num + 1, self.classes_num + 1))
        self.contain_matrix = np.zeros((self.classes_num, self.classes_num))
        self.toi_F = {"FN": [0] * self.classes_num, "FP": [0] * self.classes_num}
        self.filter_F = {"FN": [0] * self.classes_num, "FP": [0] * self.classes_num}
        self.nms_F = {"FN": [0] * self.classes_num, "FP": [0] * self.classes_num}

        self.pred_all, self.pred, self.recall_all, self.recall = 0, 0, 0, 0

        self.nested_confusion_matrix = np.zeros((self.nested_falg_num, self.nested_falg_num))

    def get_count_dataframe(self, confusion_matrix, labels):
        num = len(labels)
        experiment_metrics = []
        sum_result_dir = {"TP": 0, "FP": 0, "FN": 0}
        for one_label in range(num):
            TP = confusion_matrix[one_label, one_label]
            sum_result_dir["TP"] += TP
            FP = sum(confusion_matrix[:, one_label]) - TP
            sum_result_dir["FP"] += FP
            FN = sum(confusion_matrix[one_label, :]) - TP
            sum_result_dir["FN"] += FN

            precision = TP / (TP + FP) if TP + FP != 0 else 0
            recall = TP / (TP + FN) if TP + FN != 0 else 0
            F1_score = (2 * precision * recall) / (precision + recall) if precision * recall != 0 else 0
            experiment_metrics.append([precision, recall, F1_score])

        TP = sum_result_dir["TP"]
        FP = sum_result_dir["FP"]
        FN = sum_result_dir["FN"]
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        F1_score = (2 * precision * recall) / (precision + recall) if precision * recall != 0 else 0

        experiment_metrics.append([precision, recall, F1_score])
        return pd.DataFrame(experiment_metrics, columns=["precision", "recall", "F1_score"],
                            index=labels + ["overall"]), F1_score
