from src.config.config import *


if FLAGS.phase == 'hierarchical_preprocess':
    from src.utils.data_helper import load_inputs_data_preprocess
    load_inputs_data_preprocess(FLAGS.dataset, FLAGS.max_sentence_len, FLAGS.embedding_file_path, FLAGS.embedding_dim)
elif FLAGS.phase == 'train':
    if FLAGS.dataset == "DBpedia":
        from src.train.train import train
        train()
elif FLAGS.phase == 'test':
    from src.train.test import test
    test()
elif FLAGS.phase == 'bert':
    from src.bert_classifier.trainer import Trainer
    trainer = Trainer(FLAGS)
    trainer.train()
elif FLAGS.phase == 'bert_test':
    from src.bert_classifier.test import Test
    Test(FLAGS)
else:
    raise ValueError(FLAGS.phase)