# 对模型进行MLM预训练
from transformers import AutoModelForMaskedLM,AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
import os
import pandas as pd
import math
import torch

UNCASED = '/Users/kevin/Desktop/program files/TCM-BERT/问诊病历/BertModelWWW'
train_file = "/Users/kevin/Desktop/program files/DeepLearning/BERT句子预测任务/Notes/BERTForMaskedLM/Result/data.txt"
eval_file = "/Users/kevin/Desktop/program files/DeepLearning/BERT句子预测任务/Notes/BERTForMaskedLM/Result/data.txt"
max_seq_length = 512
out_model_path = '/Users/kevin/Desktop/program files/DeepLearning/BERT句子预测任务/Notes/BERTForMaskedLM/Result/'
train_epoches = 10
batch_size = 10

# ----二，读取文件及文件预处理----
path = "/Users/kevin/Desktop/program files/DeepLearning/BERT句子预测任务/Notes/BERTForMaskedLM/Data/ZHBCColdHotPropertyResult.xlsx"
sheetName = "3ClassLabelsData"
outputPath = '/Users/kevin/Desktop/program files/DeepLearning/BERT句子预测任务/Notes/BERTForMaskedLM/Result/'

# ---load Data and prepare handle---
def getData(path):
    sourceDataFrame = pd.read_excel(path, sheet_name=sheetName)
    #print("sourceDataFrame:\n", sourceDataFrame)
    herbsArray = sourceDataFrame['Herbs'].values
    contentArray = sourceDataFrame["FunctionText"].values
    #print("sentences:\n", contentArray)
    labelIds = sourceDataFrame['Property&Flavor'].values 
    #print("targets:\n", labelIds.shape)
    MeridianArray = sourceDataFrame['Meridian'].values
    comb = contentArray+'｜'+labelIds+'｜'+MeridianArray+'\n'
    #comb = str(comb)
    #print('comb:\n', comb)
    f = open(outputPath+'data.txt', 'w')
    for comEle in comb: f.writelines(comEle)
    f.close()
    return herbsArray, labelIds, contentArray 

getData(path)

# 这里不是从零训练，而是在原有预训练的基础上增加数据进行预训练，因此不会从 config 导入模型
tokenizer = AutoTokenizer.from_pretrained(UNCASED, use_fast=True)

model = AutoModelForMaskedLM.from_pretrained(UNCASED)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=160,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=eval_file,
    block_size=160,
)

training_args = TrainingArguments(
        output_dir=out_model_path,
        overwrite_output_dir=True,
        num_train_epochs=train_epoches,
        per_device_train_batch_size=batch_size,
        save_steps=2000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(out_model_path)
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")