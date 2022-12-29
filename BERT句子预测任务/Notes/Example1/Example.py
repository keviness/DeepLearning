import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW, BertForNextSentencePrediction,BertForMaskedLM
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import logging
logging.set_verbosity_error()
import torch.nn.functional as nnf
from sklearn.metrics import accuracy_score

# ---一，参数设置---
#SEED = 123
device = 'cuda' if torch.cuda.is_available() else 'cpu'
UNCASED = '/Users/kevin/Desktop/program files/TCM-BERT/问诊病历/BertModelWWW'
Model_cache_dir = '/Users/kevin/Desktop/program files/研究生论文/药性-BERT/2020中国药典/MultiLabelBert/BestModel/'
bert_config = BertConfig.from_pretrained(UNCASED)
BATCH_SIZE = 30
learning_rate = 2e-4
weight_decay = 1e-2
epsilon = 1e-8
maxLength = 70
epochs = 25
hidden_size = 768
#class_num = 22

# ----二，读取文件及文件预处理----
path = "/Users/kevin/Desktop/program files/研究生论文/药性-BERT/2020中国药典/MultiLabelBert/Data/2020HerbsInfoVectorize.xlsx"
sheetName = "Sheet1"
outputPath = '/Users/kevin/Desktop/program files/研究生论文/药性-BERT/2020中国药典/MultiLabelBert/Result/'

# ---load Data and prepare handle---
def getData(path, labelList):
    sourceDataFrame = pd.read_excel(path, sheet_name=sheetName)
    #print("sourceDataFrame:\n", sourceDataFrame)
    herbsArray = sourceDataFrame['Herbs'].values
    contentArray = sourceDataFrame["功能与主治"].values
    #print("sentences:\n", contentArray)
    labelIds = sourceDataFrame.loc[:,labelList].values 
    #print("targets:\n", labelIds.shape)
    labelIds = torch.tensor(labelIds, dtype=torch.float)
    return herbsArray, labelIds, contentArray 

def writeToExcelFile(trainLossArray,trainAccuracyArray):
    dataFrame = pd.DataFrame({'trainLossArray':trainLossArray,
                              'trainAccuracyArray':trainAccuracyArray})
    dataFrame.to_excel(outputPath+'TrainLossAcc.xlsx')
    print('write to excel successfully!')

def convertTextToToken(contentArray, maxLength):
    tokenizer = BertTokenizer.from_pretrained(UNCASED)
    inputIds = []
    attentionMask = []
    token_type_ids = []
    for elementText in contentArray:
        token = tokenizer(elementText, add_special_tokens=True, padding='max_length', truncation=True, max_length=maxLength)
        inputIds.append(token['input_ids'])
        attentionMask.append(token['attention_mask'])
        token_type_ids.append(token['token_type_ids'])
    inputIds = torch.tensor(inputIds, dtype=torch.long)
    attentionMask = torch.tensor(attentionMask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    #print('inputIds:\n', inputIds)
    #print('attentionMask:\n', attentionMask)
    return inputIds, attentionMask, token_type_ids



samples = ["[CLS]今天天气怎么样？[SEP]今天天气很好。[SEP]", "[CLS]小明今年几岁了？[SEP]小明爱吃西瓜。[SEP]"]
tokenizer = BertTokenizer.from_pretrained(UNCASED)
tokenized_text = [tokenizer.tokenize(i) for i in samples]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
input_ids = torch.LongTensor(input_ids)
print('input_ids:\n', input_ids)

segments_ids = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]

segments_tensors = torch.tensor(segments_ids)
print('segments_tensors:\n', segments_tensors)

model = BertForNextSentencePrediction.from_pretrained(UNCASED)
model.eval()
#最后将样本输入模型进行预测，输出模型的预测结果。

outputs = model(input_ids)
print('outputs:\n',outputs)
seq_relationship_scores = outputs[0]
print('seq_relationship_scores:\n',seq_relationship_scores)
sample = seq_relationship_scores.detach().numpy()
pred = np.argmax(sample, axis=1)
print('pred:\n', pred)