import numpy as np
import torch 
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction, BertForQuestionAnswering
from transformers import  BertModel
from model import BertClassifier

model_name = 'bert-base-chinese'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# a. 通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(model_name)
# b. 导入配置文件
#model_config = BertConfig.from_pretrained(model_name)
bert_config = BertConfig.from_pretrained(model_name)
# 修改配置
bert_config.output_hidden_states = True
bert_config.output_attentions = True
# 通过配置和路径导入模型
#bert_model = BertModel.from_pretrained(config=bert_config).to(device)
bert_model = BertClassifier(bert_config).to(device)

# ---1.分词器测试---
'''
testString = "伤寒表不解，心下有水气，干呕发热而咳，或渴，或利，或噎，或小便不利，少腹满，或喘者，小青龙汤主之。"
print(tokenizer.encode(testString))   
sen_code = tokenizer.encode_plus(testString)
print("sentenceCode:\n", sen_code)
print("sentenceCode_input_ids:\n", sen_code['input_ids'])
print("sentenceCode_token_type_ids:\n", sen_code['token_type_ids'])
print("sentenceCode_attention_mask:\n", sen_code['attention_mask'])
print(tokenizer.convert_ids_to_tokens(sen_code['input_ids']))
'''
# ---1. 遮蔽语言模型 Masked Language Model---
#model_name = 'bert-base-chinese' # 指定需下载的预训练模型参数
# BERT在预训练中引入[CLS]和[SEP]标记句子的开头和结尾
samples = ['[CLS] 中国的首都是哪里？ [SEP] 北京是 [MASK] 国的首都。 [SEP]'] 
tokenizer = BertTokenizer.from_pretrained(model_name)   
'''
tokenizer_text = [tokenizer.tokenize(i) for i in samples] # 将句子分割成一个个token，即一个个汉字和分隔符
# [['[CLS]', '中', '国', '的', '首', '都', '是', '哪', '里', '？', '[SEP]', '北', '京', '是', '[MASK]', '国', '的', '首', '都', '。', '[SEP]']]
print(tokenizer_text) 
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenizer_text]
input_ids = torch.LongTensor(input_ids)
print("input_ids1:\n", input_ids) 
# tensor([[ 101,  704, 1744, 4638, 7674, 6963, 3221, 1525, 7027, 8043,  102, 1266,776, 3221,  103, 1744, 4638, 7674, 6963,  511,  102]])
'''
input_ids1 = tokenizer.encode(samples[0])
print("input_ids1:\n", input_ids1)
input_ids2 = torch.LongTensor(input_ids1)
print("input_ids2:\n", input_ids2) 


# 读取预训练模型
model = BertForMaskedLM.from_pretrained(model_name)
model.eval()

'''
outputs = model(input_ids)
prediction_scores = outputs[0]  
# prediction_scores.shape=torch.Size([1, 21, 21128])
sample = prediction_scores[0].detach().numpy() # (21, 21128)  

# 21为序列长度，pred代表每个位置最大概率的字符索引
pred = np.argmax(sample, axis=1)  # (21,)
print(tokenizer.convert_ids_to_tokens(pred))
print(tokenizer.convert_ids_to_tokens(pred)[14])  
# 被标记的[MASK]是第14个位置, 中
'''

# ---3. 句子预测任务---
'''
predictModel = BertForQuestionAnswering(config=bert_config)
predictModel.eval()
question, text = '里昂是谁？', '里昂是一个杀手。'

sen_code = tokenizer.encode_plus(question, text)
tokens_tensor = torch.tensor([sen_code['input_ids']])
segments_tensors = torch.tensor([sen_code['token_type_ids']]) 
# 区分两个句子的编码（上句全为0，下句全为1）

start_pos, end_pos = predictModel(tokens_tensor, token_type_ids = segments_tensors)
#start_pos = start_pos.detach().numpy()
#end_pos = end_pos.detach().numpy()
print("start_pos:\n", type(start_pos), "\nend_pos", type(end_pos))
#进行逆编码，得到原始的token
all_tokens = tokenizer.convert_ids_to_tokens(sen_code['input_ids'])
#print(all_tokens)  
# ['[CLS]', '里', '昂', '是', '谁', '[SEP]', '里', '昂', '是', '一', '个', '杀', '手', '[SEP]']

# 对输出的答案进行解码的过程
answer = ''.join(all_tokens[torch.argmax(start_pos) : torch.argmax(end_pos) + 1])
print(answer)   # 一 个 杀 手
'''