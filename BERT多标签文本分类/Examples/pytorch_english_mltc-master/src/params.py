# -*- coding: utf-8 -*-
# @Time : 2022/3/12 18:24
# @Author : Jclian91
# @File : params.py
# @Place : Yangpu, Shanghai

# 模型参数
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-05
LABELS = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
# 模型
MODEL_NAME_OR_PATH = '/Users/kevin/Desktop/program files/TCM-BERT/问诊病历/BertModelWWW'
HIDDEN_LAYER_SIZE = 768
CHECKPOINT_PATH = '/Users/kevin/Desktop/program files/DeepLearning/BERT多标签文本分类/Examples/pytorch_english_mltc-master/Data/current_checkpoint.pkl'
BEST_MODEL = '/Users/kevin/Desktop/program files/DeepLearning/BERT多标签文本分类/Examples/pytorch_english_mltc-master/Data/best_model.pkl'
INPUTPATH = '/Users/kevin/Desktop/program files/DeepLearning/BERT多标签文本分类/Examples/pytorch_english_mltc-master/src/data/train.csv'