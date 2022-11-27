# BERT命名实体识别(NER)实战

> 🔗 原文链接： [https://blog.csdn.net/abc1352622216...](https://blog.csdn.net/abc13526222160/article/details/122052004)

### 文章目录

* 一. 数据集介绍
* 二. 数据集读取&预处理
* 三. 数据分词tokenizer
* 四. 定义数据读取(继承Dataset)
* 五. 定义模型&优化器&学习率
* 六. 训练测试以及准确率
* 七. 模型预测
* 八. 整个代码
* 九. BILSTM+Pytorch
* 十. 参考
* **BERT技术详细介绍： **[https://zhangkaifang.blog.csdn.net/article/details/120507302](https://zhangkaifang.blog.csdn.net/article/details/120507302)
* **本项目代码github链接： **[https://github.com/zhangkaifang/NLP-Learning](https://github.com/zhangkaifang/NLP-Learning)
* **BERT命名实体识别模型如下：**

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=ZmFhMjRiYWJjZjYyZjBkNmJkYzRkMThjNjE3OTYyNDJfdkxrOWV1TnAzRXZNZG9USnVndTE2QUZwOE5Va2t5cVlfVG9rZW46Ym94Y25HNmRWRWxRNkdING04Y1B0UUdoT2NkXzE2Njk1NDAyNjU6MTY2OTU0Mzg2NV9WNA)

# 一. 数据集介绍

* **实验使用的数据集是微软亚洲研究院提供的词性标注数据集，其目标是识别文本中具有特定意义的实体,包括人名、地名、机构名。链接： **[https://mirror.coggle.club/dataset/ner/msra.zip](https://mirror.coggle.club/dataset/ner/msra.zip)
* **百度云链接: **[https://pan.baidu.com/s/17MRMTrQKWJ6-HUI-rWL80A ](https://pan.baidu.com/s/17MRMTrQKWJ6-HUI-rWL80A)提取码: oq7w

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NGMyY2QxMTI1YWM3M2VhZDNiOTQxOWJkZGFhZjlhZGJfb21ob2RUY0htcHZYOFpkc09tMEhvU1ZUUjVzMWdDRmhfVG9rZW46Ym94Y25sWnVCbjBOUTlKYWhDVDNtV1FXa05oXzE2Njk1NDAyNjU6MTY2OTU0Mzg2NV9WNA)

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=OTQzZDNhYzY0YjlkYmQyODgwZWMyNWQ1Njg5MjU4ZDBfQ2VSUEJpaEtDQmtHYnFzendMNHo2eUlPcUs1a0Jwd3dfVG9rZW46Ym94Y25NcU91QmpVa3VFVEoyR2RSME1taW1lXzE2Njk1NDAyNjU6MTY2OTU0Mzg2NV9WNA)

# 二. 数据集读取&预处理

```Python
import codecs

################## 1. 读取数据
# 训练数据和标签
train_lines = codecs.open('msra/train/sentences.txt').readlines()
train_lines = [x.replace(' ', '').strip() for x in train_lines]  # 用于移除字符串开头和结尾指定的字符（默认为空格或换行符）或字符序列。
train_tags = codecs.open('msra/train/tags.txt').readlines()
train_tags = [x.strip().split(' ') for x in train_tags]
train_tags = [[tag_type.index(x) for x in tag] for tag in train_tags]
train_lines, train_tags = train_lines[:20000], train_tags[:20000]  # 只取两万数据
print(train_lines[0], "\n", train_tags[0])
# 如何解决足球界长期存在的诸多矛盾，重振昔日津门足球的雄风，成为天津足坛上下内外到处议论的话题。 
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 验证数据和标签
val_lines = codecs.open('msra/val/sentences.txt').readlines()
val_lines = [x.replace(' ', '').strip() for x in val_lines]
val_tags = codecs.open('msra/val/tags.txt').readlines()
val_tags = [x.strip().split(' ') for x in val_tags]
val_tags = [[tag_type.index(x) for x in tag] for tag in val_tags]
```

# 三. 数据分词tokenizer

* **注意：中文注意加 ****`list(train_lines)`**** ,因为不加因为单词作为整体了。**

```Python
################## 2. 对数据进行分词
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# 中文注意加list(train_lines),因为不加因为单词作为整体了。
max_length = 64
train_encoding = tokenizer.batch_encode_plus(list(train_lines), truncation=True, padding=True, max_length=max_length)
val_encoding = tokenizer.batch_encode_plus(list(val_lines), truncation=True, padding=True, max_length=max_length)
```

# 四. 定义数据读取(继承Dataset)

* **注意：下面labels需要填充开头cls，结尾部分不够maxlen也要填0。**

```Python
################## 3. 定义Dataset类对象
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx][:maxlen]) for key, value in self.encodings.items()}
        # 字级别的标注，注意填充cls，这里[0]代表cls。后面不够长的这里也是补充0，样本tokenizer的时候已经填充了
        # item['labels'] = torch.tensor([0] + self.labels[idx] + [0] * (63-len(self.labels[idx])))[:64]
        item['labels'] = torch.tensor([0] + self.labels[idx] + [0] * (maxlen - 1 - len(self.labels[idx])))[:maxlen]
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TextDataset(train_encoding, train_tags)
test_dataset = TextDataset(val_encoding, val_tags)
print(train_dataset[0])

# Dataset转换成Dataloader
batchsz = 32
train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsz, shuffle=True)
```

# 五. 定义模型&优化器&学习率

```Python
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

################## 4. 定义模型
model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=7)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)

# 优化器和学习率
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=total_steps)  # Default value in run_glue.py
```

# 六. 训练测试以及准确率

* **注意：outputs输出结果中的logits，NER其实就是对每个token进行分类。**

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NjgyYmUwZTAxMTFlNGIwMjg2OWNjNzAzNjIwODE5MWFfTGxyS29NWm1rRERGMFE3MWxrdnJTZTdnTWVxVEdLVE5fVG9rZW46Ym94Y25OVFYyZzQzbGlSUzlLaGhhVTZONGliXzE2Njk1NDAyNjU6MTY2OTU0Mzg2NV9WNA)

* **然后对dim=2维度上取argmax，找出每个位置所属的类别下标。**

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=OWY0NzhhZWQwMjcxMzJmZDk3YzIzMTAxZTk2YTZiMzVfaU5pN2RnREFsY2FqN2JDOGd0RTdwOWtKdHNSSFd6RlNfVG9rZW46Ym94Y25yemdDbFpLcXB6VzkybEJuOEl3VHpmXzE2Njk1NDAyNjU6MTY2OTU0Mzg2NV9WNA)

```Shell
# 这里测试计算准确率中的：
a = torch.tensor([1, 2, 3, 4, 2])
b = torch.tensor([1, 2, 4, 3, 2])
print((a==b).float().mean())
print((a==b).float().mean().item())
```

* **代码如下：**

```Python
from tqdm import tqdm

def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for idx, batch in enumerate(train_loader):
        optim.zero_grad()
      
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # loss = outputs[0]

        loss = outputs.loss
      
        if idx % 20 == 0:
            with torch.no_grad():
                # 64 * 7
                print((outputs[1].argmax(2).data == labels.data).float().mean().item(), loss.item())
      
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        iter_num += 1
        if(iter_num % 100==0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))
      
    print("Epoch: %d, Average training loss: %.4f"%(epoch, total_train_loss/len(train_loader)))
  
def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += (outputs[1].argmax(2).data == labels.data).float().mean().item()
      
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f"%(total_eval_loss/len(test_dataloader)))
    print("-------------------------------")
  

for epoch in range(4):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    validation()
```

# 七. 模型预测

```Python
model = torch.load('bert-ner.pt')

tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']

def predcit(s):
    item = tokenizer([s], truncation=True, padding='longest', max_length=64) # 加一个list
    with torch.no_grad():
        input_ids = torch.tensor(item['input_ids']).to(device).reshape(1, -1)
        attention_mask = torch.tensor(item['attention_mask']).to(device).reshape(1, -1)
        labels = torch.tensor([0] * attention_mask.shape[1]).to(device).reshape(1, -1)
      
        outputs = model(input_ids, attention_mask, labels)
        outputs = outputs[0].data.cpu().numpy()
      
    outputs = outputs[0].argmax(1)[1:-1]
    ner_result = ''
    ner_flag = ''
  
    for o, c in zip(outputs,s):
        # 0 就是 O，没有含义
        if o == 0 and ner_result == '':
            continue
      
        # 
        elif o == 0 and ner_result != '':
            if ner_flag == 'O':
                print('机构：', ner_result)
            if ner_flag == 'P':
                print('人名：', ner_result)
            if ner_flag == 'L':
                print('位置：', ner_result)
              
            ner_result = ''
      
        elif o != 0:
            ner_flag = tag_type[o][2]
            ner_result += c
    return outputs

s = '整个华盛顿已笼罩在一片夜色之中，一个电话从美国总统府白宫打到了菲律宾总统府马拉卡南宫。'
# 识别出句子里面的实体识别（NER）
data = predcit(s)
s = '人工智能是未来的希望，也是中国和美国的冲突点。'
data = predcit(s)
s = '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。'
data = predcit(s)
s = '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难'
data = predcit(s)
s = '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报'
data = predcit(s)
s = '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
data = predcit(s)
```

```Shell
位置： 华盛顿
位置： 美国总统府白宫
位置： 菲律宾总统府马拉卡南宫
位置： 华盛顿
位置： 美国总统府白宫
位置： 菲律宾总统府马拉卡南宫
位置： 中国
位置： 美国
位置： 海淀
人名： 刘涛
人名： 王华
机构： 同煤集团同生安平煤业公司
机构： 山东省政府办公厅
机构： 平邑县玉荣商贸有限公司
位置： 黑龙江
机构： 龙煤集团
```

# 八. 整个代码

* **此外提供了notebook版本代码，百度云: **[https://pan.baidu.com/s/1tiLqvsdzuBgWFb6defNyWg ](https://pan.baidu.com/s/1tiLqvsdzuBgWFb6defNyWg)提取码: gu8t

```Python
# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""=====================================
@author : kaifang zhang
@time   : 2021/12/19 1:33 PM
@contact: kaifang.zkf@dtwave-inc.com
====================================="""
import codecs
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
# B-ORG I-ORG 机构的开始位置和中间位置
# B-PER I-PER 人物名字的开始位置和中间位置
# B-LOC I-LOC 位置的开始位置和中间位置

################## 1. 读取数据
# 训练数据和标签
train_lines = codecs.open('msra/train/sentences.txt').readlines()
train_lines = [x.replace(' ', '').strip() for x in train_lines]  # 用于移除字符串开头和结尾指定的字符（默认为空格或换行符）或字符序列。
train_tags = codecs.open('msra/train/tags.txt').readlines()
train_tags = [x.strip().split(' ') for x in train_tags]
train_tags = [[tag_type.index(x) for x in tag] for tag in train_tags]
train_lines, train_tags = train_lines[:20000], train_tags[:20000]  # 只取两万数据
print(f"样例数据：{train_lines[0]} \n样例标签：{train_tags[0]}")

# 验证数据和标签
val_lines = codecs.open('msra/val/sentences.txt').readlines()
val_lines = [x.replace(' ', '').strip() for x in val_lines]
val_tags = codecs.open('msra/val/tags.txt').readlines()
val_tags = [x.strip().split(' ') for x in val_tags]
val_tags = [[tag_type.index(x) for x in tag] for tag in val_tags]  # 标签转换为数值

################## 2. 对数据进行分词
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# 中文注意加list(train_lines),因为不加因为单词作为整体了。
maxlen = 64
train_encoding = tokenizer.batch_encode_plus(list(train_lines), truncation=True, padding=True, max_length=maxlen)
val_encoding = tokenizer.batch_encode_plus(list(val_lines), truncation=True, padding=True, max_length=maxlen)

################## 3. 定义Dataset类对象
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx][:maxlen]) for key, value in self.encodings.items()}
        # 字级别的标注，注意填充cls，这里[0]代表cls。后面不够长的这里也是补充0，样本tokenizer的时候已经填充了
        # item['labels'] = torch.tensor([0] + self.labels[idx] + [0] * (63-len(self.labels[idx])))[:64]
        item['labels'] = torch.tensor([0] + self.labels[idx] + [0] * (maxlen - 1 - len(self.labels[idx])))[:maxlen]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encoding, train_tags)
test_dataset = TextDataset(val_encoding, val_tags)
batchsz = 32
train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsz, shuffle=True)

# print(train_dataset[0])

# 测试样本是否满足最大长度
for idx in range(len(train_dataset)):
    item = train_dataset[idx]
    for key in item:
        if item[key].shape[0] != 64:
            print(key, item[key].shape)
for idx in range(len(test_dataset)):
    item = test_dataset[idx]
    for key in item:
        if item[key].shape[0] != 64:
            print(key, item[key].shape)

################## 4. 定义模型
model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=7)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# 优化器和学习率
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=total_steps)  # Default value in run_glue.py

################## 4. 训练测试以及字符的分类准确率
def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # shape: [32, 64]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        # loss = outputs.loss
        logits1 = outputs[1]  # shape: [32, 64, 7]
        out = logits1.argmax(dim=2)
        out1 = out.data
        # logits2 = outputs.logits

        if idx % 20 == 0:  # 看模型的准确率
            with torch.no_grad():
                # 假如输入的是64个字符，64 * 7
                print((outputs[1].argmax(2).data == labels.data).float().mean().item(), loss.item())

        total_train_loss += loss.item()

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 参数更新
        optimizer.step()
        scheduler.step()

        iter_num += 1
        if (iter_num % 100 == 0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))

def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += (outputs[1].argmax(2).data == labels.data).float().mean().item()

    avg_val_accuracy = total_eval_accuracy / len(test_loader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_loader)))
    print("-------------------------------")

# tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
def predcit(s):
    item = tokenizer([s], truncation=True, padding='longest', max_length=64)  # 加一个list
    with torch.no_grad():
        input_ids = torch.tensor(item['input_ids']).to(device).reshape(1, -1)
        attention_mask = torch.tensor(item['attention_mask']).to(device).reshape(1, -1)
        labels = torch.tensor([0] * attention_mask.shape[1]).to(device).reshape(1, -1)

        outputs = model(input_ids, attention_mask, labels)
        outputs = outputs[0].data.cpu().numpy()

    outputs = outputs[0].argmax(1)[1:-1]
    ner_result = ''
    ner_flag = ''

    for o, c in zip(outputs, s):
        # 0 就是 O，没有含义
        if o == 0 and ner_result == '':
            continue
        #
        elif o == 0 and ner_result != '':
            if ner_flag == 'O':
                print('机构：', ner_result)
            if ner_flag == 'P':
                print('人名：', ner_result)
            if ner_flag == 'L':
                print('位置：', ner_result)

            ner_result = ''

        elif o != 0:
            ner_flag = tag_type[o][2]
            ner_result += c
    return outputs

# for epoch in range(4):
#     print("------------Epoch: %d ----------------" % epoch)
#     train()
#     validation()
# torch.save(model, 'bert-ner.pt')

model = torch.load('/data/aibox/kaifang/NLP学习资料/bert-ner.pt')
s = '整个华盛顿已笼罩在一片夜色之中，一个电话从美国总统府白宫打到了菲律宾总统府马拉卡南宫。'
# 识别出句子里面的实体识别（NER）
data = predcit(s)
s = '整个华盛顿已笼罩在一片夜色之中，一个电话从美国总统府白宫打到了菲律宾总统府马拉卡南宫。'
# 识别出句子里面的实体识别（NER）
data = predcit(s)
s = '人工智能是未来的希望，也是中国和美国的冲突点。'
data = predcit(s)
s = '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。'
data = predcit(s)
s = '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难'
data = predcit(s)
s = '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报'
data = predcit(s)
s = '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
data = predcit(s)
```

# 九. BILSTM+Pytorch

* **代码数据下载链接 ** ： [https://www.aliyundrive.com/s/oQJFwaSt17p](https://www.aliyundrive.com/s/oQJFwaSt17p)

```PowerShell
# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""=====================================
@author : kaifang zhang
@time   : 2022/2/1 23:45
@contact: kaifang.zkf@dtwave-inc.com
====================================="""
import os
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


def build_corpus(split, make_vocab=True, data_dir="data"):
    """ 读取数据 """
    assert split in ["train", "dev", "test"]
    word_lists, tag_lists = [], []
    with open(os.path.join(data_dir, split + ".char.bmes"), mode="r", encoding="utf-8") as f:
        word_list, tag_list = [], []
        for line in f:
            if line != "\n":
                word, tag = line.strip("\n").split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list, tag_list = [], []
    word_lists = sorted(word_lists, key=lambda x: len(x), reverse=False)
    tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=False)

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        word2id['<UNK>'] = len(word2id)
        word2id['<PAD>'] = len(word2id)
        tag2id['<PAD>'] = len(tag2id)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps


class MyDataset(nn.Module):
    """ 自定义Dataset类 """

    def __init__(self, datas, tags, word2index, tag2index):
        self.datas = datas
        self.tags = tags
        self.word2index = word2index
        self.tag2index = tag2index

    def __getitem__(self, index):
        data = self.datas[index]
        tag = self.tags[index]

        data_index = [self.word2index.get(i, self.word2index['<UNK>']) for i in data]
        tag_index = [self.tag2index[i] for i in tag]

        return data_index, tag_index

    def __len__(self):
        assert len(self.datas) == len(self.tags)
        return len(self.tags)

    def pro_batch_data(self, batch_datas):
        """ 每个batch如何自动填充 """
        global device
        datas, tags, batch_lens = [], [], []
        for data, tag in batch_datas:
            datas.append(data)
            tags.append(tag)
            batch_lens.append(len(data))
        batch_max_len = max(batch_lens)
        datas = [i + [self.word2index['<PAD>']] * (batch_max_len - len(i)) for i in datas]
        tags = [i + [self.tag2index['<PAD>']] * (batch_max_len - len(i)) for i in tags]

        return torch.tensor(datas, dtype=torch.int64, device=device), torch.tensor(tags, dtype=torch.long,
                                                                                   device=device)  # long也是int64


class MyModel(nn.Module):
    def __init__(self, corpus_num, embedding_num, hidden_num, class_num, bi=True):
        super().__init__()
        self.embedding = nn.Embedding(corpus_num, embedding_num)
        self.lstm = nn.LSTM(embedding_num, hidden_num, batch_first=True, bidirectional=bi)

        if bi:
            self.classifer = nn.Linear(hidden_num * 2, class_num)
        else:
            self.classifer = nn.Linear(hidden_num, class_num)
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, batch_data, batch_tag=None):
        embedding = self.embedding(batch_data)
        out, _ = self.lstm(embedding)
        pred = self.classifer(out)
        self.pred = torch.argmax(pred, dim=-1).reshape(-1)
        if batch_tag is not None:
            loss = self.cross_loss(pred.reshape(-1, pred.shape[-1]), batch_tag.reshape(-1))
            return loss


def test():
    global word2index, model, index2tag, device  # 全局变量声明，只是读取
    while True:
        text = input("请输入：")
        text_index = [[word2index.get(i, word2index['<UNK>']) for i in text]]
        text_index = torch.tensor(text_index, dtype=torch.int64, device=device)
        model.forward(text_index)
        pred = [index2tag[i] for i in model.pred]

        print([f'{w}_{s}' for w, s in zip(text, pred)])


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_word_lists, train_tag_lists, word2index, tag2index = build_corpus("train", make_vocab=True)
    dev_data, dev_tag = build_corpus("dev", make_vocab=False)
    index2tag = [i for i in tag2index]

    # 定义一些变量
    corpus_num = len(word2index)
    class_num = len(tag2index)  # 命名实体识别就是为每个字进行分类
    epoch = 50
    lr = 0.001
    embedding = 101
    hidden_num = 107
    bi = True
    batchsz = 64

    train_dataset = MyDataset(train_word_lists, train_tag_lists, word2index, tag2index)
    # 自己处理：collate_fn=train_dataset.pro_batch_data
    train_dataloader = DataLoader(train_dataset, batch_size=batchsz, shuffle=False,
                                  collate_fn=train_dataset.pro_batch_data)

    dev_dataset = MyDataset(dev_data, dev_tag, word2index, tag2index)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batchsz, shuffle=False,
                                collate_fn=dev_dataset.pro_batch_data)

    model = MyModel(corpus_num, embedding, hidden_num, class_num, bi)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    for e in tqdm.trange(epoch):
        model.train()
        for batch_data, batch_tag in train_dataloader:
            train_loss = model(batch_data, batch_tag)
            train_loss.backward()
            opt.step()
            opt.zero_grad()
        print(f"train loss: {train_loss:.3f}")

        model.eval()
        all_pred, all_tag = [], []
        for dev_batch_data, dev_batch_tag in dev_dataloader:
            dev_loss = model(dev_batch_data, dev_batch_tag)
            all_pred.extend(model.pred.detach().cpu().numpy().tolist())
            all_tag.extend(dev_batch_tag.detach().cpu().numpy().reshape(-1).tolist())
        # print(f"dev loss: {dev_loss:.3f}")
        score = f1_score(all_tag, all_pred, average="macro")
        print(f"{e},f1_score:{score:.3f},dev_loss:{dev_loss:.3f}")
    # test()
```

# 十. 参考

* **主要参考dasou博主的视频 ** ： [https://www.bilibili.com/video/BV1Ey4y1874y?p=6&amp;spm_id_from=pageDriver](https://www.bilibili.com/video/BV1Ey4y1874y?p=6&spm_id_from=pageDriver)
* **腾讯Bugly的专栏 ** ： [图解BERT模型：从零开始构建BERT](https://cloud.tencent.com/developer/article/1389555)
* Bert源代码解读-以BERT文本分类代码为例子： [https://github.com/DA-southampton/Read_Bert_Code](https://github.com/DA-southampton/Read_Bert_Code)
* BERT大火却不懂Transformer？读这一篇就够了： [https://zhuanlan.zhihu.com/p/54356280](https://zhuanlan.zhihu.com/p/54356280)
* pytorch 中加载BERT模型, 获取词向量： [https://blog.csdn.net/znsoft/article/details/107725285](https://blog.csdn.net/znsoft/article/details/107725285)
* [Bert生成句向量(pytorch)](https://blog.csdn.net/weixin_30034903/article/details/113399809?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-1.no_search_link&spm=1001.2101.3001.4242.1)
* [https://blog.csdn.net/weixin_41519463/article/details/100863313](https://blog.csdn.net/weixin_41519463/article/details/100863313)
* 学习率预热(transformers.get_linear_schedule_with_warmup)： [https://blog.csdn.net/orangerfun/article/details/120400247](https://blog.csdn.net/orangerfun/article/details/120400247)
