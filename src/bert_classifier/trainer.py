#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof

import json
import os
import time

import tensorflow as tf
from src.bert_classifier.model_HATC import BertClassifier
# from src.bert_classifier.model import BertClassifier
# from src.bert_classifier.model_base import BertClassifier
from src.bert import modeling
from src.bert_classifier.data_helper import TrainData
from src.bert_classifier.metrics import mean, get_multi_metrics


class Trainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)
        self.__bert_checkpoint_path = os.path.join(self.config["bert_model_path"], "bert_model.ckpt")

        # 加载数据集
        self.data_obj = self.load_data()
        self.t_in_ids, self.t_in_masks, self.t_seg_ids, self.t_lab_ids_1, lab_to_idx_1, \
        self.t_lab_ids_2, lab_to_idx_2, self.t_lab_ids_3, lab_to_idx_3 = self.data_obj.gen_data(
            self.config["train_data"])

        self.e_in_ids, self.e_in_masks, self.e_seg_ids, self.e_lab_ids_1, lab_to_idx_1, \
        self.e_lab_ids_2, lab_to_idx_2, self.e_lab_ids_3, lab_to_idx_3 = self.data_obj.gen_data(
            self.config["eval_data"], is_training=False)

        print("train data size: {}".format(len(self.t_lab_ids_1)))
        print("eval data size: {}".format(len(self.e_lab_ids_1)))

        self.label_list_1 = [value for key, value in lab_to_idx_1.items()]
        print("label_1 numbers: ", len(self.label_list_1))
        self.label_list_2 = [value for key, value in lab_to_idx_2.items()]
        print("label_2 numbers: ", len(self.label_list_2))
        self.label_list_3 = [value for key, value in lab_to_idx_3.items()]
        print("label_3 numbers: ", len(self.label_list_3))

        num_train_steps = int(
            len(self.t_lab_ids_1) / self.config["batch_size"] * self.config["epochs"])
        num_warmup_steps = int(num_train_steps * self.config["warmup_rate"])
        # 初始化模型对象
        self.model = self.create_model(num_train_steps, num_warmup_steps)

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        data_obj = TrainData(self.config)
        return data_obj

    def create_model(self, num_train_step, num_warmup_step):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        model = BertClassifier(config=self.config, num_train_step=num_train_step, num_warmup_step=num_warmup_step)
        return model

    def train(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, self.__bert_checkpoint_path)
            print("init bert model params")
            tf.train.init_from_checkpoint(self.__bert_checkpoint_path, assignment_map)
            print("init bert model params done")
            sess.run(tf.variables_initializer(tf.global_variables()))

            current_step = 0
            start = time.time()
            saver = tf.train.Saver(max_to_keep=1)
            max_acc_1, max_acc_2, max_acc_3 = 0, 0, 0
            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.data_obj.next_batch(self.t_in_ids, self.t_in_masks, self.t_seg_ids,
                                                      self.t_lab_ids_1, self.t_lab_ids_2, self.t_lab_ids_3):
                    loss, predictions_1, predictions_2, predictions_3 = self.model.train(sess, batch)

                    acc_1, recall_1, prec_1, f_beta_1 = get_multi_metrics(pred_y=predictions_1,
                                                                          true_y=batch["label_ids_1"],
                                                                          labels=self.label_list_1)

                    acc_2, recall_2, prec_2, f_beta_2 = get_multi_metrics(pred_y=predictions_2,
                                                                          true_y=batch["label_ids_2"],
                                                                          labels=self.label_list_2)

                    acc_3, recall_3, prec_3, f_beta_3 = get_multi_metrics(pred_y=predictions_3,
                                                                          true_y=batch["label_ids_3"],
                                                                          labels=self.label_list_3)

                    print(
                        "level 1 train: step: {}, loss: {:.4f}, acc: {:.4f}, recall: {:.4f}, precision: {:.4f}, f_beta: {:.4f}".format(
                            current_step, loss, acc_1, recall_1, prec_1, f_beta_1))
                    print(
                        "level 2 train: step: {}, loss: {:.4f}, acc: {:.4f}, recall: {:.4f}, precision: {:.4f}, f_beta: {:.4f}".format(
                            current_step, loss, acc_2, recall_2, prec_2, f_beta_2))
                    print(
                        "level 3 train: step: {}, loss: {:.4f}, acc: {:.4f}, recall: {:.4f}, precision: {:.4f}, f_beta: {:.4f}".format(
                            current_step, loss, acc_3, recall_3, prec_3, f_beta_3))

                    current_step += 1
                    if self.data_obj and current_step % self.config["checkpoint_every"] == 0:

                        eval_losses = []
                        eval_accs_1, eval_accs_2, eval_accs_3 = [], [], []
                        # eval_aucs = []
                        eval_recalls_1, eval_recalls_2, eval_recalls_3 = [], [], []
                        eval_precs_1, eval_precs_2, eval_precs_3 = [], [], []
                        eval_f_betas_1, eval_f_betas_2, eval_f_betas_3 = [], [], []

                        for eval_batch in self.data_obj.next_batch(self.e_in_ids, self.e_in_masks,
                                                                   self.e_seg_ids, self.e_lab_ids_1,
                                                                   self.e_lab_ids_2, self.e_lab_ids_3):
                            eval_loss, eval_predictions_1, eval_predictions_2, eval_predictions_3 = self.model.eval(
                                sess, eval_batch)

                            eval_losses.append(eval_loss)

                            acc_1, recall_1, prec_1, f_beta_1 = get_multi_metrics(pred_y=eval_predictions_1,
                                                                                  true_y=eval_batch["label_ids_1"],
                                                                                  labels=self.label_list_1)
                            eval_accs_1.append(acc_1)
                            eval_recalls_1.append(recall_1)
                            eval_precs_1.append(prec_1)
                            eval_f_betas_1.append(f_beta_1)

                            acc_2, recall_2, prec_2, f_beta_2 = get_multi_metrics(pred_y=eval_predictions_2,
                                                                                  true_y=eval_batch["label_ids_2"],
                                                                                  labels=self.label_list_2)
                            eval_accs_2.append(acc_2)
                            eval_recalls_2.append(recall_2)
                            eval_precs_2.append(prec_2)
                            eval_f_betas_2.append(f_beta_2)

                            acc_3, recall_3, prec_3, f_beta_3 = get_multi_metrics(pred_y=eval_predictions_3,
                                                                                  true_y=eval_batch["label_ids_3"],
                                                                                  labels=self.label_list_3)
                            eval_accs_3.append(acc_3)
                            eval_recalls_3.append(recall_3)
                            eval_precs_3.append(prec_3)
                            eval_f_betas_3.append(f_beta_3)

                        print("\n")
                        print(
                            "level 1 eval: loss: {:.4f}, acc: {:.4f}, recall: {:.4f}, precision: {:.4f}, f_beta: {:.4f}".format(
                                mean(eval_losses), mean(eval_accs_1), mean(eval_recalls_1),
                                mean(eval_precs_1), mean(eval_f_betas_1)))
                        print(
                            "level 2 eval: loss: {:.4f}, acc: {:.4f}, recall: {:.4f}, precision: {:.4f}, f_beta: {:.4f}".format(
                                mean(eval_losses), mean(eval_accs_2), mean(eval_recalls_2),
                                mean(eval_precs_2), mean(eval_f_betas_2)))
                        print(
                            "level 3 eval: loss: {:.4f}, acc: {:.4f}, recall: {:.4f}, precision: {:.4f}, f_beta: {:.4f}".format(
                                mean(eval_losses), mean(eval_accs_3), mean(eval_recalls_3),
                                mean(eval_precs_3), mean(eval_f_betas_3)))
                        print("\n")

                        if mean(eval_accs_3) > max_acc_3:
                            max_acc_1 = mean(eval_accs_1)
                            max_acc_2 = mean(eval_accs_2)
                            max_acc_3 = mean(eval_accs_3)

                            save_path = self.config["ckpt_model_path"]
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            path = saver.save(sess, model_save_path, global_step=current_step)

                            # path = self.model.saver.save(sess, model_save_path, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))

                            # if self.config["ckpt_model_path"]:
                            #     save_path = self.config["ckpt_model_path"]
                            #     if not os.path.exists(save_path):
                            #         os.makedirs(save_path)
                            #     model_save_path = os.path.join(save_path, self.config["model_name"])
                            #     self.model.saver.save(sess, model_save_path, global_step=current_step)
                        print("level1 topacc {:g}, level2 topacc {:g}, level3 topacc {:g}".
                              format(max_acc_1, max_acc_2, max_acc_3))
                        print("\n")
            end = time.time()
            print("level1 topacc {:g}, level2 topacc {:g}, level3 topacc {:g}".format(max_acc_1, max_acc_2, max_acc_3))
            print("total train time: ", end - start)

# if __name__ == "__main__":
#     # 读取用户在命令行输入的信息
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config_path", help="config path of model")
#     args = parser.parse_args()
#     trainer = Trainer(args)
#     trainer.train()
