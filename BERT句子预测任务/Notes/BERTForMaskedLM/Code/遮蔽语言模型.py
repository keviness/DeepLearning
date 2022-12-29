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
UNCASED = '/Users/kevin/Desktop/program files/DeepLearning/BERT句子预测任务/Notes/BERTForMaskedLM/Result/FinalModel'
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


samples = ['[CLS] 祛风除湿；活血能络。主风湿痹痛；跌打损伤；外伤出血｜ [SEP] [MASK] [MASK] [MASK] [MASK] [MASK]  [SEP]']  # 准备输入模型的语句

tokenizer = BertTokenizer.from_pretrained(UNCASED)
tokenized_text = [tokenizer.tokenize(i) for i in samples]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
input_ids = torch.LongTensor(input_ids)
print('input_ids:\n', input_ids)

# 读取预训练模型
model = BertForMaskedLM.from_pretrained(UNCASED)
model.eval()
#此时，我们已经准备好了待输入的语句和预训练模型，接下来需要做的就是让模型去预测的覆盖的词的序号。

outputs = model(input_ids)
prediction_scores = outputs[0]
print('prediction_scores:\n',prediction_scores)
print('prediction_scores.shape:\n',prediction_scores.shape)
#最后找到预测值中最大值对应的序号，然后通过 tokenizer.convert_ids_to_tokens() 在词表中查找，转换成对应的字。

sample = prediction_scores[0].detach().numpy()
pred = np.argmax(sample, axis=1)

result = tokenizer.convert_ids_to_tokens(pred)
print('result:\n', result)