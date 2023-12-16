import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd
# 参数设置
batch_size = 30
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10
learning_rate = 0.01    #Learning Rate不宜太大
maxLength = 160
UNCASED = '/Users/kevin/Desktop/program files/TCM-BERT/问诊病历/BertModelWWW'
Model_cache_dir = '/Users/kevin/Desktop/program files/DeepLearning/BERT文本多分类/Examples/BERT句子分类/BestModel/'
bert_config = BertConfig.from_pretrained(UNCASED)

path = "/Users/kevin/Desktop/program files/研究生论文/药性-BERT/2020中国药典/HotColdPropertyBert/Data/2015HerbsProcessResult.xlsx"
#sheetName = "ExperimentDataNoWei"
sheetName = "2ClassExperimentData"
outputPath = '/Users/kevin/Desktop/program files/DeepLearning/BERT文本多分类/Examples/BERT句子分类/Result/'

def getData(path):
    sourceDataFrame = pd.read_excel(path, sheet_name=sheetName)
    #print("sourceDataFrame:\n", sourceDataFrame)
    herbsArray = sourceDataFrame['Herbs'].values
    contentArray = sourceDataFrame["Functions and Indications"].values
    #print("sentences:\n", contentArray)
    labelIds = sourceDataFrame.loc[:,'PropertyCode'].values 
    #print("targets:\n", labelIds)
    labelIds = torch.tensor(labelIds, dtype=torch.long)
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
    #print('token_type_ids:\n', token_type_ids)
    return inputIds, attentionMask, token_type_ids

# -----------划分数据集-----------
def trainTestSplit(labelIds, inputIds, attentionMask, token_type_ids):
    train_input_ids, test_input_ids, train_labels, test_labels, train_masks, test_masks, train_token_type_ids, test_token_type_ids = train_test_split(inputIds,labelIds, attentionMask, token_type_ids, random_state=666, test_size=0.2)
    
    #print("train_labels:\n", test_labels)
    train_data = TensorDataset(train_input_ids, train_token_type_ids,train_masks, train_labels)
    #train_dataloader = DataLoader(train_data,  batch_size=BATCH_SIZE)
    test_data = TensorDataset(test_input_ids, test_token_type_ids,test_masks, test_labels)
    #print("train_data_test_labels:\n", train_data.train_labels)
    return train_data, test_data


def main(train_dataset, valid_dataset, num_labels):
    # 生成Batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained(UNCASED)

    # 初始化模型
    model = BertClassifier(bert_config, num_labels).to(device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, epochs+1):
        losses = 0      # 损失
        accuracy = 0    # 准确率
        
        model.train()
        train_bar = tqdm(train_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            # 梯度清零
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)

            # 传入数据，调用model.forward()
            output = model(
                input_ids=input_ids.to(device), 
                attention_mask=attention_mask.to(device), 
                token_type_ids=token_type_ids.to(device), 
            )

            # 计算loss
            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)   # 预测出的label
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels) #acc
            accuracy += acc

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(), acc=acc)


        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)

        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)

        # 验证
        model.eval()
        losses = 0      # 损失
        accuracy = 0    # 准确率
        valid_bar = tqdm(valid_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id  in valid_bar:
            valid_bar.set_description('Epoch %i valid' % epoch)

            output = model(
                input_ids=input_ids.to(device), 
                attention_mask=attention_mask.to(device), 
                token_type_ids=token_type_ids.to(device), 
            )
            
            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)   # 预测出的label
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels) #acc
            accuracy += acc
            valid_bar.set_postfix(loss=loss.item(), acc=acc)

        average_loss = losses / len(valid_dataloader)
        average_acc = accuracy / len(valid_dataloader)

        print('\tValid ACC:', average_acc, '\tLoss:', average_loss)

        
        # 判断并保存验证集上表现最好的模型
        if average_acc > best_acc:
            best_acc = average_acc
            torch.save(model.state_dict(), Model_cache_dir+'best_model.pkl')
        
if __name__ == '__main__':
    #labels = ['寒','热','温','凉','平']
    labels = ['寒凉','温热']
    num_labels = len(labels)
    
    herbsArray, labelIds, contentArray = getData(path)
    inputIds, attentionMask, token_type_ids = convertTextToToken(contentArray, maxLength)
    
    train_data, test_data = trainTestSplit(labelIds, inputIds, attentionMask, token_type_ids)
    main(train_data, test_data, num_labels)