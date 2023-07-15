#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import json

from src.bert_classifier.predict import Predictor
from src.bert_classifier.metrics import get_multi_metrics, mean
import io


def Test(FLAGS):
    with io.open(FLAGS.config_path, "r") as fr:
        config = json.load(fr)
    print(config)
    predictor = Predictor(config)

    label_to_index_1 = predictor.label_to_index_1
    label_to_index_2 = predictor.label_to_index_2
    label_to_index_3 = predictor.label_to_index_3

    label_list_1 = [value for key, value in label_to_index_1.items()]
    label_list_2 = [value for key, value in label_to_index_2.items()]
    label_list_3 = [value for key, value in label_to_index_3.items()]

    text, labels_1, labels_2, labels_3 = predictor.read_data(config['test_data'])

    # text = ["歼20座舱盖上的两条“花纹”是什么？", "歼20座舱盖上的两条“花纹”是什么？"]
    # labels_1 = [0, 1]
    # labels_2 = [0, 1]
    # labels_3 = [0, 1]

    # 3，标签转索引
    labels_ids_1 = predictor.trans_label_to_index(labels_1, label_to_index_1)
    # print(labels_ids_1)
    # print(type(labels_ids_1))
    labels_ids_2 = predictor.trans_label_to_index(labels_2, label_to_index_2)
    labels_ids_3 = predictor.trans_label_to_index(labels_3, label_to_index_3)

    input_ids, input_masks, segment_ids = predictor.sentence_to_idx(text)

    eval_accs_1, eval_accs_2, eval_accs_3 = [], [], []
    for test_batch in predictor.next_batch(input_ids, input_masks, segment_ids, labels_ids_1, labels_ids_2, labels_ids_3):
        predictions_1, predictions_2, predictions_3 = predictor.predict(test_batch)

        acc_1, recall_1, prec_1, f_beta_1 = get_multi_metrics(pred_y=predictions_1, true_y=test_batch["label_ids_1"],
                                                              labels=label_list_1)
        acc_2, recall_2, prec_2, f_beta_2 = get_multi_metrics(pred_y=predictions_2, true_y=test_batch["label_ids_2"],
                                                              labels=label_list_2)
        acc_3, recall_3, prec_3, f_beta_3 = get_multi_metrics(pred_y=predictions_3, true_y=test_batch["label_ids_3"],
                                                              labels=label_list_3)

        # predictions_1, predictions_2, predictions_3 = predictor.predict(eval_batch)
        eval_accs_1.append(acc_1)
        eval_accs_2.append(acc_2)
        eval_accs_3.append(acc_3)
        # eval_recalls_1.append(recall_1)
        # eval_precs_1.append(prec_1)
        # eval_f_betas_1.append(f_beta_1)

    print("level1 acc {:g}, level2 acc {:g}, level3 acc {:g}".format(mean(eval_accs_1), mean(eval_accs_2), mean(eval_accs_3)))