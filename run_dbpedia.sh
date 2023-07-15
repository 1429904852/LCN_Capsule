#!/usr/bin/env bash
python main.py --phase="train" --dataset="DBpedia" --model_type="HAN_capsule_overrall" --batch_size=64 --n_class_1=9 --n_class_2=70 --n_class_3=219 --embedding_file_path='DB_embedding.txt' --saver_checkpoint='HAN_capsule_overrall'