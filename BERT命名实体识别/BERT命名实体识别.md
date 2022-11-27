# 手把手教你用BERT做命名实体识别（NER）

> 🔗 原文链接： [https://zhuanlan.zhihu.com/p/358376...](https://zhuanlan.zhihu.com/p/358376510)

本教程使用 [CLUENER（中文语言理解测评基准）2020数据集 ](https://link.zhihu.com/?target=https%3A//github.com/CLUEbenchmark/CLUENER2020)作为用来fine-tune的数据集，同时使用该repo下提供的 [base-line model ](https://link.zhihu.com/?target=https%3A//github.com/CLUEbenchmark/CLUENER2020/tree/master/tf_version)来fine-tune和预测。

## 数据

使用 [CLUENER（中文语言理解测评基准）2020数据集 ](https://link.zhihu.com/?target=https%3A//github.com/CLUEbenchmark/CLUENER2020)作为用来fine-tune模型的数据集。数据分为10个标签类别，分别为: 地址（address），书名（book），公司（company），游戏（game），政府（government），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）。

标签定义如下：

```Plaintext
· 地址（address）: **省**市**区**街**号，**路，**街道，**村等（如单独出现也标记）。地址是标记尽量完全的, 标记到最细。
· 书名（book）: 小说，杂志，习题集，教科书，教辅，地图册，食谱，书店里能买到的一类书籍，包含电子书。
· 公司（company）: **公司，**集团，**银行（央行，中国人民银行除外，二者属于政府机构）, 如：新东方，包含新华网/中国军网等。
· 游戏（game）: 常见的游戏，注意有一些从小说，电视剧改编的游戏，要分析具体场景到底是不是游戏。
· 政府（government）: 包括中央行政机关和地方行政机关两级。 中央行政机关有国务院、国务院组成部门（包括各部、委员会、中国人民银行和审计署）、国务院直属机构（如海关、税务、工商、环保总局等），军队等。
· 电影（movie）: 电影，也包括拍的一些在电影院上映的纪录片，如果是根据书名改编成电影，要根据场景上下文着重区分下是电影名字还是书名。
· 姓名（name）: 一般指人名，也包括小说里面的人物，宋江，武松，郭靖，小说里面的人物绰号：及时雨，花和尚，著名人物的别称，通过这个别称能对应到某个具体人物。
· 组织机构（organization）: 篮球队，足球队，乐团，社团等，另外包含小说里面的帮派如：少林寺，丐帮，铁掌帮，武当，峨眉等。
· 职位（position）: 古时候的职称：巡抚，知州，国师等。现代的总经理，记者，总裁，艺术家，收藏家等。
· 景点（scene）: 常见旅游景点如：长沙公园，深圳动物园，海洋馆，植物园，黄河，长江等。
```

## 预训练模型

如下图所示，该repo提供了3个base-line model。其中谷歌官方提供的BERT-base比双向LSTM+CRF模型高了7分，而由哈工大讯飞联合实验室基于全词遮罩（Whole Word Masking）技术发布的中文预训练模型 [BERT-wwm ](https://link.zhihu.com/?target=https%3A//github.com/ymcui/Chinese-BERT-wwm)又把模型效果提升了一些。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDhhMThkMTU0ZDgwYTMzNTM2MzU3OGQxMzU1MzQ4NThfSHFXaFpDc0k2TzJaMWw5WUhqNkFmSWhBMXVHWUVaRlRfVG9rZW46Ym94Y25hMHRCeVg5NDFVWEl1QmN1eWJzWm1mXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

三个Baseline model的效果

本文选择了效果最好的RoBERTa-wwm-large模型，其预测结果的f1 score为80.42，表现相当不错。作为对比，截至2021年3月19日，在 [CLUE命名实体任务排行榜 ](https://link.zhihu.com/?target=https%3A//www.cluebenchmarks.com/ner.html)上的最高得分也就是82.545分。如下图所示：

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=MDgzOTAyNzViN2E1OWQ2OWQ3MTM2MzAzM2IxZDg4NTVfSEF3Ykp0UnJheXhUNTVIV0tIV25HV3dkYklaSURlNktfVG9rZW46Ym94Y25kZ05WcmdYRWZ6MUdUSzY1Z0x4MEtnXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

CLUE命名实体任务排行榜

RoBERTa-wwm-large模型的分实体类型的f1 score如下图所示。可以发现效果最好的是人名、政府、电影名；效果最差的是地址、场景、书名。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=N2E2MDgzYTAzZGU1Y2RjZDYzMjI4MTI0NTMyMjc2NDZfZlVBOVFjSUVXYnBYQlJidXJtQmhxd1JGUTZ5c1k4QUhfVG9rZW46Ym94Y25VYXVueHhPRzZRdVlJMTR1dnVIdWdmXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

## 运行环境

我选用了Colab作为运行环境，具体原因有三。首先，因为Colab是一个交互式的运行环境，方便调试。其次，在Colab上预装了大量的包，其中就包括TensorFlow1和TensorFlow2，节省了配置环境的时间。最后，也是最重要的一点，Colab提供了非常强大的计算资源，如下图所示，显卡是Tesla K80，显存是12GB。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=YmI1Nzk2M2JmNzgyZWMxYTgzN2U1MjdkYzM1OWQ5MDlfMFpnNDVETzBBODV3T0Y4a21rRm52QktEUWhxSVQxTkhfVG9rZW46Ym94Y25wMEcxVmlDQklCS3FnaE5ZNGVoVTlnXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

Colab提供的GPU的硬件参数

另外说一下python的版本是Python 3.7.10。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NDA0MTU1MjJmMDZmOTBhNWRmMmQ3ZmVkYjk4MDA4YmVfR1R6ZmtXTDZSdkp6YmROY3hYVjNHSGRnOUFxS2VlZXJfVG9rZW46Ym94Y25KcEJEQm04dkZ2ODI4VG10SHVCcnhuXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

Colab中的python版本

## 数据集的格式

所有的数据集都是json格式。

首先看一下训练集。如下图中圆圈所示：

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDczZTI4OTg4MzQwODQ0YTBkYWQzZmU3ZDc1NTQ3MGRfRHJVSTFKcDhhTGVBOVNnVURIdkd5UGdmVGtpUU9yeERfVG9rZW46Ym94Y25zeUVUVk9HYnI4bWs2SWlydHNpajhmXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

* 红圈：最外层字典里面有两个键，第一个键是文本text，第二个键是标签label。
* 蓝圈：label里面又是一个字典，text里面有几类实体（例如人名+机构名是两类），这个字典里面就有几个键；键名是实体的类别，键值也就是实体的名字，放在第三层字典里面。
* 绿圈：第三层字典的键值是实体的名字（例如“英雄联盟”），键值是一个列表：这个实体在这个句子中出现了几次，这个列表里面就有几个子列表，每一个子列表的第一个元素表示这个实体的第一个字在这个text的第几个位置，第二个元素表示这个实体的最后一个字在这个text的第几个位置出现。
* 另外，  **要特别注意两个点 ** ：一是上图右下角，文档的编码必须是UTF-8；二是所有的引号必须是英文双引号，不能是单引号。

验证集的格式和训练集一样：

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=MGNkOGFhMzU1OWQwNjQ3YmUxZTdhYzUwYzVmODFkNjdfalJoUWpiQUo5N3Q5VmIwcXRMYjMxS2lzWUlMWmhhWndfVG9rZW46Ym94Y241cGpNdzZwbGdhRGwyOEl2MnVRdXhkXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

测试集也是字典格式，每一行是一个字典，第一个键是text的ID编号，第二个键是text文本。同样，也要是UTF-8编码。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=MjM5NzYyZDdlYzFiNWRkMjM2YWZhZmE2ZDliYmM4MWRfWmZGdzBZUmlGemlOQ2dmQ3hNRUg4OUZLM0dxU216REdfVG9rZW46Ym94Y25wRmtPTWlTYVZIZkFiTmF6SjVVdHBlXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

## Fine-tune模型

首先，我们赋予Colab访问Google Drive的权限，因为我们将把预训练模型、数据集、checkpoints、预测结果都保存在Google Drive里：

```Plaintext
from google.colab import drive
drive.mount('/content/drive')
```

把我们的工作目录改到我们为NER创建的文件夹：

```Plaintext
import os
path = "/content/drive/My Drive/NER-classifier_roberta_wwm_large"
os.chdir(path)
```

从GitHub上下载base-line Model到本地：

```Plaintext
! git clone https://github.com/CLUEbenchmark/CLUENER2020.git
```

把工作路径改到base-line model所在的路径，并且定位到TensorFlow模型路径：

```Plaintext
path = "/content/drive/My Drive/NER-classifier_roberta_wwm_large/CLUENER2020/tf_version/"
os.chdir(path)
```

**下面一步要特别注意。 **这个base-line model是基于TensorFlow1写的，所以我们需要在运行之前告诉colab我们要用的是TensorFlow1。

```Plaintext
%tensorflow_version 1.x
import tensorflow as tf
tf.__version__
```

然后直接运行run_classifier_roberta_wwm_large.sh文件即可。

```Plaintext
! bash run_classifier_roberta_wwm_large.sh
```

这个sh文件做了3件事情：

1. 下载用来fine-tune模型的数据集cluener_public.zip
2. 下载预训练模型chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip
3. 运行run_classifier_roberta_wwm_large.py文件，并传入我们设定好的模型训练的参数。

由于这个sh文件使用Linux命令自动获取当前路径，因此我们的路径里面如果含有空格，会导致它在创建文件夹以及在文件夹之间跳转的时候出现一些问题。我就遇到了这个问题，因此我手动下载了数据集和预训练模型，然后手动运行了第三步中的py文件。

在这之前，我们要将下载好的数据集zip文件加压到./CLUEdataset/ner文件夹下（这个路径以及下一句中的路径可以改成别的文件夹，只不过下文命令中传的参数也要改），下载好的预训练模型zip文件解压到./prev_trained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16文件夹下。

然后在cell里面运行run_classifier_roberta_wwm_large.py文件，并传入我们设定好的参数，来fine-tune和预测：

```Plaintext
! python run_classifier_roberta_wwm_large.py \
  --task_name='ner' \
  --do_train=true \
  --do_predict=true \
  --data_dir=./CLUEdataset/ner \
  --vocab_file=./prev_trained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt \
  --bert_config_file=./prev_trained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json \
  --init_checkpoint=./prev_trained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=4.0 \
  --output_dir=./ner_output
```

解释一下上面的命令：

* ! python run_classifier_roberta_wwm_large.py：英文感叹号“!”会把我们输入的命令转换成Bash，这样我们就可以在Colab里面运行Shell命令。这里我们就是用python解释器运行了这个py文件
* --task_name='ner'：从这里开始，都是我们传入到py文件里面运行的参数。这里表示我们运行的任务是ner，程序在看到这个参数以后，会实例化NerProcessor这个类，用来执行我们的NER任务。如果这个py文件里面有别的processor，我们也可以输入别的任务名。但在我们的py文件里面，只有这一个processor，因此如果改成别的会报错。
* --do_train=true \ 和--do_predict=true \：表示我们要模型在数据集上训练并输出预测结果。这里的True和False会在运行时传入到py文件中，并储存在tf.flags里面（也就是一个tensorflow对象的flasg属性里面），这些tf.flags是True还是False控制着main函数里面各个if逻辑语句下面的代码块（例如做train的代码块、做predict的代码块）是否执行，最终决定我们的程序实现的是什么功能。
* 下面三行包含路径的分别是BERT模型词汇表的路径、模型配置文件的路径、预训练模型的路径。
* 再下面4行指定了模型的训练参数。
* --output_dir=./ner_output：指定了模型的输出路径，包括checkpoints、预测结果、以及其他的文件都会输出到这里。

**我用Colab的GPU训练并预测，总共用了快2个小时。 **这时候在./ner_output文件夹下，我们可以查看模型预测的结果：

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTBhODk0NjE1YmIwN2Y4ZGQyOWQ4NzQ5OTVhM2ZmZjVfOXhHRnNCdTRYcGtRcTVVV1RITE4xRHFqS0pDeXFLRGVfVG9rZW46Ym94Y243MllKTWVVTm5qRHZRRkM2a0ViVWpzXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

## 输出

模型输出的直接结果是对每一个字打上包含两个维度的标签。一个是实体的类型（例如人名、地名等）另一个是这个字属于实体的开头、中间还是实体外。

标签示例：实体的类型

| LOC | 地名   |
| --- | ------ |
| PER | 人民   |
| ORG | 机构名 |

标签示例：一个是这个字属于实体的开头、中间还是实体外。

其中，短横线前面标注的是这个字在实体中的位置，短横线后面的“X”是上面表格中所说的实体类型。

| 标注 | 含义    | 含义                 |
| ---- | ------- | -------------------- |
| B-X  | Begin   | 代表实体X的开头      |
| I-X  | Inside  | 代表实体的内部       |
| O    | outside | 代表不属于任何类型的 |

一句话在打完标签后的结果如下表所示：

| 《 | 北    | 京    | 文 | 物 | 保 | 存    | 保    | 管 | 状    | 态    | 之 | 调    | 查    | 报    | 告        | 》        |
| -- | ----- | ----- | -- | -- | -- | ----- | ----- | -- | ----- | ----- | -- | ----- | ----- | ----- | --------- | --------- |
| O  | B-LOC | I-LOC | O  | O  | O  | O     | O     | O  | O     | O     | O  | O     | O     | O     | O         | O         |
| 调 | 查    | 范    | 围 | 涉 | 及 | 故    | 宫    | 、 | 历    | 博    | 、 | 古    | 研    | 所    | `` | `` |
| O  | O     | O     | O  | O  | O  | B-LOC | I-LOC | O  | B-LOC | I-LOC | O  | B-ORG | I-ORG | I-ORG | `` | `` |

为了让结果更便于使用，我们根据标签将实体名称提取出来，并根据实体类型分类：

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NTQxOGY0ZGU1NGNiZGQ0ZmM0ODliYjIyYTYwZmFmYTdfUDBUWHkxd29iN2hQbExTbncyMnV0MHFEd01JWnl6Z0tfVG9rZW46Ym94Y255VW13Y2R4eUJONnA3RFFpbFU1V1lnXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

如上图所示，返回的结果是一个列表。输入了几个句子，列表中就有几个元素。在样例中我们输入了一个句子，因此该列表中是有一个元素。

列表中的元素是字典的形式。最外层的键是实体的类型，值是一个内层的字典。内层的字典的键是实体的名称，值是这个名称出现在原句中的第几个字到第几个字。

## 如何输入自己的数据来预测？

首先预处理我们要预测的文本。也就是把我们需要预测的文本处理成上文介绍的格式，然后保存成test.json（必须是这个名字），并放在./text_data文件夹下（可以改成别的文件夹，只不过下文命令中传的参数也要改）。下图是我用来预测的文本，总共94个句子。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NDc0ZmU0MjEwZjM5MGNiNzE0NzZhY2FmMmNlZGRkNTNfZGVBYWJvRnRrRHFCdkdDOUJEcndFV3NjZm1ldnNjQ2lfVG9rZW46Ym94Y25qY0FiTHF3cjlnQlhyQWRNbFdpemFkXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

今天刚出的热乎新闻

在和之前一样的运行环境下，运行如下代码即可。

```Plaintext
! python run_classifier_roberta_wwm_large.py \
  --task_name='ner' \
  --do_train=False \
  --do_eval=False \
  --do_predict=true \
  --data_dir=./text_data \
  --vocab_file=./prev_trained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt \
  --bert_config_file=./prev_trained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json \
  --init_checkpoint=./ner_output/model.ckpt-5374 \
  --output_dir=./text_output
```

这里的参数相比之前做了一点改动：

* do_train和do_eval全部改成了False，do_predict仍然是True。因为我们不需要去训练和评估模型，只需要预测。
* data_dir：改成了我自己的数据的路径。
* init_checkpoint：改成了fine-tune后生成的checkpoints文件的路径，注意写到我在上面展示的文件名长度就行了（即“model.ckpt-5374”），不需要在最后加“.data-00000-of-00001”。
* output_dir：输出路径修改了，避免把前一轮预测的数据覆盖了。

如果仅仅用fine-tune以后的模型来预测，不需要模型来学习并修改权值，那么运算量是非常小的，用CPU就可以解决。

例如，我在i7-9750H上运行的两个NER文本时，CPU占用率保持在60%左右。94个句子的文本，运行过程为1分钟03秒（包括了20秒读取包和预训练模型的时间）；376个句子的任务运行时间为3分30秒。运行过程的CPU占用情况如下：

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=YmNjY2M4ZTA3MGI3NWMzNzgxOWQxY2M3NGNjY2JlNjBfQVdwR1pJMmJ5dFBPVXdaOHFUOG5mYmFVSU9IWWFNcEVfVG9rZW46Ym94Y24wRnlGSUsxNG5iUTVjMVhFNjY2UEFkXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

预测结果如下图所示：

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=MTQ4Mjg5YzNkZDFiYzA1OTViOGYxMzI4NzU1MTU5MmZfSjdSZ1JiWXZES3c3ejROY0xJRDR4TkFlbGo5VnoxM2tfVG9rZW46Ym94Y25Lc1hMOU0wZHBQVkZqWHNCTTg2U2RkXzE2Njk1MjM4NDg6MTY2OTUyNzQ0OF9WNA)

可以看出来效果还是非常好的。

编辑于 2021-04-16 14:23
