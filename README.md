### Label-correction Capsule Network for Hierarchical Text Classification

Code and data for "[Label-correction Capsule Network for Hierarchical Text Classification](https://ieeexplore.ieee.org/document/10149184?source=authoralert)" (TASLP 2023)

------

### Requirements

- tensorflow>=1.9
- python>=2.7

------

### Hypermeters

- phase
	- hierarchical_preprocess（data process）
	- train, test（encoder=BiLSTM）
	- bert, bert_test(encoder=BERT)
- dataset
- model_type
- batch_size
- embedding_file_path

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

If the code and data are used in your research, please cite the paper:

```bash
@ARTICLE{10149184,
  author={Zhao, Fei and Wu, Zhen and He, Liang and Dai, Xin-Yu},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Label-Correction Capsule Network for Hierarchical Text Classification}, 
  year={2023},
  volume={31},
  number={},
  pages={2158-2168},
  doi={10.1109/TASLP.2023.3282099}}
```


