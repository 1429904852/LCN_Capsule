### Label-correction Capsule Network for Hierarchical Text Classification

### Dataset

||WOS|DBPedia|
|:-:|:-:|:-:|
|Number of documents|       46,985       |     381,025 |
|Mean document length|       200.7       |      106.9  |
|Classes in level 1|       7       |      9      |
|Classes in level 2|       143       |      70     |
|Classes in level 3|   NA   |      219    |

------

### Requirements

- tensorflow>=1.9
- python>=2.7

------

### Hypermeters

- phase
	- hierarchical_preprocess（数据预处理）
	- train, test（当encoder为BiLSTM使用）
	- bert, bert_test(当encoder为BERT对DBPedia数据集进行训练和测试)
- dataset
- model_type(集成了很多模型)
- batch_size
- embedding_file_path（DB_embedding.txt或者WOS_embedding.txt）

------

### Usage

#### Data process

- This is optional, because I have provided the pre-processed data under the folder named "data/data_name/hierarchical_processed/"(mainly generate .npy file)
```bash
python main.py --phase="hierarchical_preprocess" --dataset="DBpedia" --embedding_file_path="DB_embedding.txt"
```

------

we first use BiLSTM encode to verify the motivation of modal.
#### Training for BiLSTM

- This is the training code of tuning parameters on the dev set, you can change hyperparameter to select different dataset or model_type.
```bash
sh run_dbpedia.sh
or
python main.py --phase="train" --dataset="DBpedia" --model_type="HAN_capsule_overrall" --batch_size=64 --n_class_1=9 --n_class_2=70 --n_class_3=219 --embedding_file_path='DB_embedding.txt' --saver_checkpoint='HAN_capsule_overrall'
```

####  Testing for BiLSTM

- After training the model, the following code is used for directly loading the trained model and testing it on the test set:

```bash
python main.py --phase="test" --dataset="DBpedia" --model_type="HAN_capsule_overrall" --batch_size=64 --n_class_1=9 --n_class_2=70 --n_class_3=219 --embedding_file_path='DB_embedding.txt' --saver_checkpoint='HAN_capsule_overrall'
```

------

after that, we also use BERT encode to verify the scalability of our model.
#### Training for BERT

- For example, you can use the following command to fine-tune Bert on the HTC task:
```bash
sh run_bert_dbpedia.sh
or
python main.py --phase="bert" --config_path="src/bert_classifier/config/dbpedia_config.json"
```

#### Testing for BERT

- After training the model, you can use the following command to test Bert on the HTC task:

```bash
python main.py --phase="bert_test" --config_path="src/bert_classifier/config/dbpedia_config.json"
```

------

### Reference

- **[ICMLA-17]**: HDLTex: Hierarchical Deep Learning for Text Classification. [[paper]](https://arxiv.org/pdf/1709.08267.pdf) [[code]](https://github.com/kk7nc/HDLTex)

- **[EMNLP-18]**: A Hierarchical Neural Attention-based Text Classifier. [[paper]](https://www.aclweb.org/anthology/D18-1094.pdf) [[code]](https://github.com/koustuvsinha/hier-class)

