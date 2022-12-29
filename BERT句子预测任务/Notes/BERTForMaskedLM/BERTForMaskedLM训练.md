今天在读 NLP 比赛经验分享的帖子，许多人都提到一种刷分的方法：用特定领域的数据对 Bert 模型进行增量预训练。这样做的原因也比较简单：模型在预训练的过程中能够接触到更加贴近下游任务的语料。这个想法不难想到，实现起来也相对简单，这篇文章主要介绍一下笔者基于Pytorch的实现思路以及用到的工具。

### 以增量预训练 Bert 为例

我们知道，Bert 在预训练的过程中主要有两个任务：MLM 以及 NSP。MLM 任务训练模型还原被破坏的文本，NSP 任务则是训练模型判断输入的两个句子是否为上下文关系。

那么我们增量预训练的过程实际上就是通过增加一些 task-specific 数据使得模型在预训练的过程更加贴近下游任务。

增量预训练的实现思路大体可以分为：准备数据、数据处理、模型导入与训练。

本文的实现运用到了 huggingface 的一些库文件。huggingface 在实现 Bert 模型的过程中为其增加了许多下游任务的头，诸如BertForSequenceClassification、BertForQuestionAnswering等等，至于预训练，transformer库中同样有对应的头。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=YTU5ZjYwYTBlMDVhZjg5ZjM3YWNiYzMwYjNlMWVjZDNfcEpkMVBxZEg0MExOaVlHdGlSYnV1VTZjV0lLbG9lRThfVG9rZW46Ym94Y25sbnpVMk5WTjNHTHluVUJuZkdYWWZlXzE2NzIwNjg5MjI6MTY3MjA3MjUyMl9WNA)

其中，预训练对应的头包含有BertForPreTraining、BertForMaskedLM、BertForNextSentencePrediction等等。本文的增量预训练主要是对文本进行MLM任务的训练，因此使用 BertForMaskedLM。我们来看官方给出的一个demo

```Python
>>> from transformers import BertTokenizer, BertForMaskedLM
>>> import torch

>>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
>>> model = BertForMaskedLM.from_pretrained('bert-base-uncased')

>>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
>>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

>>> outputs = model(**inputs, labels=labels)
>>> loss = outputs.loss
>>> logits = outputs.logits
```

可以看到和我们在下游任务微调bert的代码思路是类似的，只要准备好 inputs、labels 传入 BertForMaskedLM 就可以开始训练了。如果需要使用两个任务进行增量训练的话可以使用 BertForPreTraining 这个头，实际上思路是一样的，只不过数据处理的方式就稍微繁琐一些，再加上很多实验都表明NSP任务实际上对模型精度提升作用不大，因此本文干脆只用 MLM 进行训练了。

### 数据处理

MLM 任务最复杂的是文本数据的处理方式，我们需要按照原文的方式对句子进行Mask。幸运的是这一过程 transformer 库中也有对应的实现。 [DataCollatorForLanguageModeling ](https://link.zhihu.com/?target=https%3A//huggingface.co/transformers/main_classes/data_collator.html%3Fhighlight%3Ddatacollatorforlanguagemodeling%23transformers.data.data_collator.DataCollatorForLanguageModeling)实现了一个对文本数据进行随机 mask 的data collator。我们只需要在制作迭代器的过程中将其作为 data_collator 传入就可以实现对文本数据的 mask。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDhiOTY4Yjk4M2QzODhkMzUyYzcwN2VkNzBmOTBhN2VfemtnQ1NxUlBsQ3dOaWk1UnpTa3RjcTR2V3ViZG5oellfVG9rZW46Ym94Y25Zc2h0TFlFaDNpSGtGcWx1NjdkZ2RjXzE2NzIwNjg5MjI6MTY3MjA3MjUyMl9WNA)

### 数据准备

那么剩下最后一步工作就是准备训练数据了。这里就没什么好多说的了，笔者在实现的过程中用到的数据是以 txt 格式存储，每行保存一个句子。因此在数据读入阶段使用了 transformers 提供的 LineByLineTextDataset 函数。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=OWM0MTk5NWZmMDZhNTlhNDcwZTNkOGRlYzI1MzYyNTJfSnFsc0FkdmNnRmNQd2xleEdpYkhUSEdQR2JjTTlZM3JfVG9rZW46Ym94Y25jYmVVOU9CYlFIbU9GV0xhR1h3NWdnXzE2NzIwNjg5MjI6MTY3MjA3MjUyMl9WNA)

完整版实现代码如下：

```Python
# 对模型进行MLM预训练
from transformers import AutoModelForMaskedLM,AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
import os
import math

model_dir = "../model"
model_name = "roberta-large"
train_file = "../data/wikitext-2/data.txt"
eval_file = "../data/wikitext-2/data.txt"
max_seq_length = 512
out_model_path = os.path.join(model_dir,'pretain')
train_epoches = 10
batch_size = 2

# 这里不是从零训练，而是在原有预训练的基础上增加数据进行预训练，因此不会从 config 导入模型
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir,model_name), use_fast=True)

model = AutoModelForMaskedLM.from_pretrained(os.path.join(model_dir,model_name))

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=eval_file,
    block_size=128,
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
```

### 小结

本来以为蛮繁琐的一件事，没想到翻文档的过程中意外发现 huggingface 把梯子都造好了，实在是 NLPer 的福音，再次顶礼膜拜。

编辑于 2021-11-26 16:59
