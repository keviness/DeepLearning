BERT实战：实现多分类

前面以及介绍过bert的理论知识，以及它相应的实现方法，那么让我们通过实战加深对bert的了解。

我们将通过bert实现一个文本多分类任务，具体是kaggle上的一个真假新闻的任务。具体如下：

文件地址：https://www.kaggle.com/c/fake-news-pair-classification-
challenge/data 模型形式：BERT + Linear Classifier
参考链接：LeeMeng - 進擊的 BERT：NLP 界的巨人之力與遷移學習
参考博客：Simple to Bert | Ripshun Blog
github地址：nlp-code/Bert_真假新闻分类.ipynb at main · cshmzin/nlp-code (github.com)
加载数据
通过pandas将数据从csv中取出来

```python
import pandas as pd
df_train = pd.read_csv("train.csv")
df_train = df_train.reset_index()
df_train = df_train.loc[:, ['title1_zh', 'title2_zh', 'label']]
df_train.columns = ['text_a', 'text_b', 'label']df_test = pd.read_csv("test.csv")
df_test = df_test.loc[:, ["title1_zh", "title2_zh", "id"]]
df_test.columns = ["text_a", "text_b", "Id"]print("训练样本数量：", len(df_train))
print("预测样本数：", len(df_test))
df_train.head()
df_test.head()
```


构建Dataset
在pytorch中通常将数据给放入dataset中，以方便我们下一步操作,在dataset中我们需要将构建bert指定格式的tokens和segments张量。

```
from torch.utils.data import Dataset
from transformers import BertTokenizertokenizer = BertTokenizer.from_pretrained('bert-base-chinese')class NewsDataset(Dataset):
def init(self, mode, tokenizer):
self.mode = mode
self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
self.len = len(self.df)
self.label_map = {'agreed': 0, 'disagreed': 1, 'unrelated': 2}
self.tokenizer = tokenizer  #使用 BERT tokenizer
```


```python
#@pysnooper.snoop()  # 加入以了解所有转换过程
def __getitem__(self, idx):
    if self.mode == "test":
        text_a, text_b = self.df.iloc[idx, :2].values
        label_tensor = None
    else:
        text_a, text_b, label = self.df.iloc[idx, :].values
        label_id = self.label_map[label]
        label_tensor = torch.tensor(label_id)
      
    word_pieces = ["[CLS]"]
    tokens_a = self.tokenizer.tokenize(text_a)
    word_pieces += tokens_a + ["[SEP]"]
    len_a = len(word_pieces)
  
    tokens_b = self.tokenizer.tokenize(text_b)
    word_pieces += tokens_b + ["[SEP]"]
    len_b = len(word_pieces) - len_a
  
    ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
    tokens_tensor = torch.tensor(ids)
  
    segments_tensor = torch.tensor([0] * len_a + [1] * len_b,dtype=torch.long)
  
    return (tokens_tensor, segments_tensor, label_tensor)

def __len__(self):
    return self.len
```

trainset = NewsDataset("train", tokenizer=tokenizer)

构建DateLoader
构建了Dataset后，我们需要将数据以成批的方式输入到模型中，所以使用pytorch中的dataloader类。

```
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequencedef create_mini_batch(samples):
tokens_tensors = [s[0] for s in samples]
segments_tensors = [s[1] for s in samples]
```


```
if samples[0][2] is not None:
    label_ids = torch.stack([s[2] for s in samples])
else:
    label_ids = None

# zero pad 到同一序列长度
tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)
segments_tensors = pad_sequence(segments_tensors,batch_first=True)

# attention masks，将 tokens_tensors 不为 zero padding 的位置设为1
masks_tensors = torch.zeros(tokens_tensors.shape,dtype=torch.long)
masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

return tokens_tensors, segments_tensors, masks_tensors, label_ids
```

BATCH_SIZE = 64
trainloader = DataLoader(trainset,batch_size=BATCH_SIZE,collate_fn=create_mini_batch)

构建模型
由于我们需要的是三分类模型，所以只需要将原始的bert模型做微调即可，简单的方式就是在模型的输出位置加上一层Linear，将维度降为3.

```
class BertForSequenceClassification(BertPreTrainedModel):
def init(self, config, num_labels=3):
super(BertForSequenceClassification, self).init(config)
self.num_labels = num_labels
self.bert = BertModel(config)
self.dropout = nn.Dropout(config.hidden_dropout_prob)
self.classifier = nn.Linear(config.hidden_size, num_labels)
```

```
def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
    outputs = self.bert(input_ids, token_type_ids, attention_mask)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    if labels is not None:

        # 如果Lables不为空返回返回损失值，即训练模式。
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

        #如果没有输入Labels则返回预测值，即测试模式
    elif self.output_attentions:
        return all_attentions, logits
    return logit
```

```python
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
EPOCHS = 10
for epoch in range(EPOCHS):
```

```python
losses = 0.0
for data in trainloader:
  
    tokens_tensors, segments_tensors, \
    masks_tensors, labels = [t.to(device) for t in data]
    optimizer.zero_grad()
    outputs = model(input_ids=tokens_tensors, 
                    token_type_ids=segments_tensors, 
                    attention_mask=masks_tensors, 
                    labels=labels)

    loss = outputs[0]
    loss.backward()
    optimizer.step()
    losses += loss.item()
 print(losses)
```


模型预测

```python
text_a = "。。。。"
text_b = "。。。。。。。"
word_pieces = ["[CLS]"]
tokens_a = tokenizer.tokenize(text_a)
word_pieces += tokens_a + ["[SEP]"]
len_a = len(word_pieces)tokens_b = tokenizer.tokenize(text_b)
word_pieces += tokens_b + ["[SEP]"]
len_b = len(word_pieces) - len_aids = tokenizer.convert_tokens_to_ids(word_pieces)
tokens_tensor = torch.tensor(ids).unsqueeze(0)segments_tensor = torch.tensor([0] * len_a + [1] * len_b,dtype=torch.long).unsqueeze(0)masks_tensors = torch.zeros(tokens_tensor.shape,dtype=torch.long)
masks_tensors = masks_tensors.masked_fill(tokens_tensor != 0, 1).unsqueeze(0)outputs = model(input_ids=tokens_tensor.to(device),token_type_ids=segments_tensor.to(device),attention_mask=masks_tensors.to(device))
logits = outputs[0]
_, pred = torch.max(logits.data, 1)
label_map = {0:'agreed', 1: 'disagreed', 2: 'unrelated'}print(outputs)
print(label_map[pred.cpu().tolist()[0]])
```
