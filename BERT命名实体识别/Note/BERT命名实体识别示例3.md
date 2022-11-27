# 命名实体识别（NER）实战

> 🔗 原文链接： [https://zhuanlan.zhihu.com/p/911946...](https://zhuanlan.zhihu.com/p/91194691)

## 大纲

* NER简单介绍
* 统计的方法及代码
* 统计和机器学习的方法及代码
* CRF方法及代码
* Bert方法及代码

## NER简单介绍

命名实体识别（NER）（也称为实体识别、实体分块和实体提取）是信息提取的一个子任务，旨在将文本中的命名实体定位并分类为预先定义的类别，如人员、组织、位置、时间表达式、数量、货币值、百分比等。

实现NER的方法有以下几种

* 统计的方法
* 机器学习
* 统计和机器学习
* LSTM
* CRF HMM
* LSTM + CRF
* Bert

文章的数据来自于 [https://www. kaggle.com/abhinavwalia 95/entity-annotated-corpus/download ](https://link.zhihu.com/?target=https%3A//www.kaggle.com/abhinavwalia95/entity-annotated-corpus/download)

## 统计的方法及代码

```Python
import pandas as pd
import numpy as np

data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")
print(data.tail(10))

words = list(set(data["Word"].values))
n_words = len(words)

print(n_words,"\n")

class SentenceGetter(object):
  def __init__(self,data):
    self.data = data
    self.n_sent = 1
    self.empty = False

  def get_next(self):
    try:
      s = self.data[self.data["Sentence #"] == "Sentence: {}".format(self.n_sent)]
      self.n_sent += 1
      return s["Word"].values.tolist(),s["POS"].values.tolist(), s["Tag"].values.tolist()
    except:
      self.empty = True
      return None,None,None

getter = SentenceGetter(data)
sent,pos,tag = getter.get_next()
print(sent,pos,tag)

from sklearn.base import BaseEstimator,TransformerMixin

class MemoryTagger(BaseEstimator,TransformerMixin):
  def fit(self,X,y):
    voc = {}
    self.tags = []
    for w, t in zip(X,y):
      if t not in self.tags:
        self.tags.append(t)
      if w not in voc:
        voc[w] = {}
      if t not in voc[w]:
        voc[w][t] = 0
      voc[w][t] += 1
    self.memory = {}
    for k, d in voc.items():
      self.memory[k] = max(d,key=d.get)
  def predict(self,X,y=None):
    return[self.memory.get(x,"O") for x in X]
# 存储形式： voc = {w1: {tag1:f1, tag2:f2...},w2: {tag1:f1, tag2:f2...},w3: {tag1:f1, tag2:f2...}...}

# 测试一下
tagger = MemoryTagger()
tagger.fit(sent,tag)
print(tagger.predict(sent))

# 用所有数据进行训练
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
words = data["Word"].values.tolist()
tags = data["Tag"].values.tolist()
pred = cross_val_predict(estimator=MemoryTagger(), X=words, y=tags, cv=5)
report = classification_report(y_pred=pred, y_true=tags)
print(report, "\n")
```

f1_score为0.96，效果还不错

## 统计和机器学习的方法及代码

```Python
from sklearn.ensemble import RandomForestClassifier
def feature_map(word):
  return np.array([word.istitle(),word.islower(),word.isupper(),len(word),word.isdigit(),word.isalpha()])

words = [feature_map(w) for w in data["Word"].values.tolist()]
# print(words)
pred = cross_val_predict(RandomForestClassifier(n_estimators=20),X=words, y=tags, cv=5)
report = classification_report(y_pred=pred,y_true=tags)
print(report)
```

发现特征太少，效果不是很好

加入前后单词的pos和tag的值

```Python
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class FeatureTransformer(BaseEstimator, TransformerMixin):

  def __init__(self):
    self.memory_tagger = MemoryTagger()
    self.tag_encoder, self.pos_encoder = LabelEncoder(), LabelEncoder()

    # s = data[data["Sentence #"] == "Sentence: {}".format(1)]
    # print(s)

  def fit(self, X, y):
    words = X["Word"].values.tolist()
    self.pos = X["POS"].values.tolist()
    tags = X["Tag"].values.tolist()
    self.memory_tagger.fit(words, tags)
    self.tag_encoder.fit(tags)
    self.pos_encoder.fit(self.pos)
    return self  # fit函数返回的结果就是self, 允许链式调用

  def transform(self, X, y=None):
    def pos_default(p):
      if p in self.pos:
        return self.pos_encoder.transform([p])[0]
      else:
        return -1

    pos = X["POS"].values.tolist()
    words = X["Word"].values.tolist()
    out = []

    print(len(words))

    for i in range(len(words)):
      print(i)

      w = words[i]
      p = pos[i]
      if i < len(words) - 1:

        # test_1 = words[i + 1]
        # test_2 = self.memory_tagger.predict([words[i + 1]])
        # test_3 = self.tag_encoder.transform(self.memory_tagger.predict([words[i + 1]]))

        wp = self.tag_encoder.transform(self.memory_tagger.predict([words[i + 1]]))[0]
        posp = pos_default(pos[i + 1])
      else:
        wp = self.tag_encoder.transform(['O'])[0]
        posp = pos_default(".")
      if i > 0:
        if words[i - 1] != ".":
          wm = self.tag_encoder.transform(self.memory_tagger.predict([words[i - 1]]))[0]
          posm = pos_default(pos[i - 1])
        else:
          wm = self.tag_encoder.transform(['O'])[0]
          posm = pos_default(".")
      else:
        posm = pos_default(".")
        wm = self.tag_encoder.transform(['O'])[0]

      test_array = np.array([w.istitle(), w.islower(), w.isupper(), len(w), w.isdigit(), w.isalpha(),
                           self.tag_encoder.transform(self.memory_tagger.predict([w]))[0],
                           pos_default(p), wp, wm, posp, posm])

      out.append(np.array([w.istitle(), w.islower(), w.isupper(), len(w), w.isdigit(), w.isalpha(),
                           self.tag_encoder.transform(self.memory_tagger.predict([w]))[0],
                           pos_default(p), wp, wm, posp, posm]))
    return out

# from sklearn.pipeline import Pipeline
# pred = cross_val_predict(Pipeline([("feature_map", FeatureTransformer()), ("clf", RandomForestClassifier(n_estimators=20, n_jobs=3))]),X=data, y=tags, cv=5)
# report = classification_report(y_pred=pred, y_true=tags)
# print(report)

featureTrans = FeatureTransformer()
featureTrans.fit(data, tags)
X_data = featureTrans.transform(data)
# print(X_data)
pred = cross_val_predict(RandomForestClassifier(n_estimators=20),X=X_data, y=tags, cv=5)
report = classification_report(y_pred=pred,y_true=tags)
print(report)
```

## CRF方法及代码

```Python
# -*- coding: utf-8 -*-
"""NER with CRF.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1VIpv0MXOf_Qq0Gs21GGq12dwzES9a471
**如何应用CRF解决NER，并且训练出的模型进行可视化**
载入数据
"""

import pandas as pd
import numpy as np


data = pd.read_csv("ner_dataset.csv",encoding = "latin1")
data = data.fillna(method = "ffill")
# data.tail(10)

words = list(set(data["Word"].values))
n_words = len(words)
# n_words

"""我们的数据中共有47959个句子，其中包含了35178个单词。 
现在来构建一个输出句子的构造器。
"""

class SentenceGetter(object):
  def __init__(self,data):
    self.n_sent = 0
    self.data = data
    self.empty = False
    agg_func = lambda s : [(w,p,t) for w,p,t in zip(s["Word"].values.tolist(),s["POS"].values.tolist(),s["Tag"].values.tolist())]
    self.grouped = self.data.groupby("Sentence #").apply(agg_func)
    self.sentences = [s for s in self.grouped]
  def get_next(self):
    try:
      s = self.sentences[self.n_sent]
      self.n_sent += 1
      return s
    except:
      self.empty = True
      return None

getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)

"""获取所有句子"""

sentences = getter.sentences

"""为CRF添加特征，包括词汇本身特征以及前后文的特征"""

def word2features(sent,i):
  word = sent[i][0]
  postag = sent[i][1]

  features = {
      "bias": 1.0,
      "word.lower()": word.lower(),
      "word[-3:]": word[-3:],
      "word[2:]": word[2:],
      "word.isupper()": word.isupper(),
      "word.istitle()": word.istitle(),
      "word.isdigit()": word.isdigit(),
      "postag": postag,
      "postag[:2]": postag[:2]
  }
  if i > 0:
    word1 = sent[i-1][0]
    postag1 = sent[i-1][1]
    features.update({
      "-1:word.isupper()": word1.isupper(),
      "-1:word.istitle()": word1.istitle(),
      "-1:word.isdigit()": word1.isdigit(),
      "-1:postag": postag1,
      "-1:postag[:2]": postag1[:2]
    })
  else:
    features["BOS"] = True
  if i < len(sent) - 1:
    word1 = sent[i+1][0]
    postag1 = sent[i+1][1]
    features.update({
      "+1:word.isupper()": word1.isupper(),
      "+1:word.istitle()": word1.istitle(),
      "+1:word.isdigit()": word1.isdigit(),
      "+1:postag": postag1,
      "+1:postag[:2]": postag1[:2]
    })
  else:
    features["EOS"] = True

  # test_1 = features

  return features

def sent2features(sent):
  return [word2features(sent,i) for i in range(len(sent))]

def sent2labels(sent):
  return [label for _, _, label in sent]

def sent2tokens(sent):
  return [token for token, _, _ in sent]

print(sent2features(sentences[0]))

print(sent2labels(sentences[0]))

print(sent2tokens(sentences[0]))

"""获取所有输入（features）与输出(labels)"""

X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

"""应用sklearn-crfsuite构建CRF"""


from sklearn_crfsuite import CRF

crf = CRF(
    algorithm = "lbfgs",
    c1 = 0.1,
    c2 = 0.1,
    max_iterations = 100,
    all_possible_transitions = False
)

from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report

pred = cross_val_predict(estimator=crf,X=X,y=y,cv=5)

report = flat_classification_report(y_pred=pred, y_true=y)
print(report)

crf.fit(X,y)

"""可视化tags间的转移分数以及各features的重要性"""

# !pip install eli5
import eli5

eli5.show_weights(crf,top=30)

"""通过增加c1增加L1正则化的力度，使features变稀疏"""

crf = CRF(algorithm='lbfgs',
          c1=10,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
report = flat_classification_report(y_pred=pred, y_true=y)
print(report)

crf.fit(X, y)


eli5.show_weights(crf, top=30)

"""增加L1正则后，features减少，但是模型效果仍旧不错"""
```

## Bert方法及代码

```Python
# -*- coding: utf-8 -*-
"""NER_Bert.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/13rfbXaTbKpzF4VEiN2kkRPVnVoshoSuy
利用Bert解决NER问题
dataset: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/download
"""

# !pip install pytorch-pretrained-bert==0.4.0

"""载入数据"""

import pandas as pd
import numpy as np
# from tqdm import tqdm, trange

# from google.colab import files
# data = files.upload()

"""展示数据最后几行"""

data = pd.read_csv("ner_dataset.csv", encoding="latin1").fillna(method="ffill")
data.tail(10)

"""构建SentenceGetter"""

class SentenceGetter(object):
  
    def __init__(self, data):
      self.n_sent = 1
      self.data = data
      self.empty = False
      agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),s["POS"].values.tolist(),s["Tag"].values.tolist())]
      self.grouped = self.data.groupby("Sentence #").apply(agg_func)
      self.sentences = [s for s in self.grouped]
  
    def get_next(self):
      try:
        s = self.grouped["Sentence: {}".format(self.n_sent)]
        self.n_sent += 1
        return s
      except:
        self.empty = True
        return None

getter = SentenceGetter(data)

sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
# sentences[0]

labels = [[s[2] for s in sent] for sent in getter.sentences]
print(labels[0])

"""构建tag词典"""

tags_vals = list(set(data["Tag"].values))
tag2idx = {t: i for i, t in enumerate(tags_vals)}

"""导入相关库"""

# !pip install pytorch_pretrained_bert

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam

"""设置基本参数"""

max_len = 60
batch_size = 32

"""设置device"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

"""tokenize处理"""

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print(tokenized_texts[0])

"""将输入转化为id 并且 截长补短"""

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_len, dtype="long", truncating="post", padding="post")
print(input_ids[0])

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=max_len, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")
print(tags[0])

"""准备mask_attention"""

attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
print(attention_masks[0])

"""将数据进行划分"""

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, random_state=2019, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2019, test_size=0.1)

"""将数据转化为tensor的形式"""

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

"""定义dataloader,在训练阶段shuffle数据，预测阶段不需要shuffle"""

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data) #预测阶段需要shuffle
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)  #测试阶段不需要shuffle
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

"""**开始训练过程**"""

model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))

model.cuda()

"""定义optimizer(分为是否调整全部参数两种情况)"""

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta'] # 不需要正则化的参数
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = BertAdam(optimizer_grouped_parameters, lr=3e-5)

"""定义评估accuracy的函数
f1: https://blog.csdn.net/qq_37466121/article/details/87719044
"""

# !pip install seqeval
from seqeval.metrics import f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

"""开始微调过程，建议4个左右epochs"""

epochs = 5
max_grad_norm = 1.0

for _ in range(epochs, desc="Epoch"): # trange有可视化功能
    # 训练过程
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(train_dataloader):
        # 将batch设置为gpu模式
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # 前向过程
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # 后向过程
        loss.backward()
        # 损失
        tr_loss += loss.item()
        nb_tr_steps += 1
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # 更新参数
        optimizer.step()
        model.zero_grad()
    #打印每个epoch的损失
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # 验证过程
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()#detach的方法，将variable参数从网络中隔离开，不参与参数更新
        label_ids = b_labels.to('cpu').numpy()

        # print("label_ids", label_ids)
        # print("np.argmax(logits, axis=2)", np.argmax(logits, axis=2))

        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        # 计算accuracy 和 loss
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    # 打印信息
    print("Validation loss: {}".format(eval_loss/nb_eval_steps))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))#传入的是具体的tag

"""附：pytorch中各类数据的转换
*   Numpy to tensor: tensor_data_cpu = torch.from_numpy(np_data)
*   Cpu tensor to cuda: tensor_data_cuda= tensor_data_cpu.cuda()
*   Cuda data to Variable: tensor_data_cuda_var=Variable(tensor_data_cuda)
*   cuda Tensor to numpy: np_data=tensor_data_cuda.cpu().numpy()
*   cuda Variable to numpy: np_data=tensor_data_cuda_var.detach().cpu().numpy()
"""
```

完整代码github地址

[kaimenluo/ailearning](https://link.zhihu.com/?target=https%3A//github.com/kaimenluo/ailearning/tree/master/ner)

编辑于 2019-11-12 14:47
