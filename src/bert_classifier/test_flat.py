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

    # 3，标签转索引
    labels_ids_1 = predictor.trans_label_to_index(labels_1, label_to_index_1)
    labels_ids_2 = predictor.trans_label_to_index(labels_2, label_to_index_2)
    labels_ids_3 = predictor.trans_label_to_index(labels_3, label_to_index_3)

    input_ids, input_masks, segment_ids = predictor.sentence_to_idx(text)

    eval_accs_3 = []
    for test_batch in predictor.next_batch(input_ids, input_masks, segment_ids, labels_ids_1, labels_ids_2, labels_ids_3):
        predictions_3 = predictor.predict_flat(test_batch)
        acc_3, recall_3, prec_3, f_beta_3 = get_multi_metrics(pred_y=predictions_3, true_y=test_batch["label_ids_3"],
                                                              labels=label_list_3)
        eval_accs_3.append(acc_3)
    print("level3 acc")
    print(mean(eval_accs_3))