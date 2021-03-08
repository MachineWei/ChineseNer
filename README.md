# bert crf for ner
这个项目支持多种预训练模型，如bert,albert,electra,robert等，只需修改其中的config、tokenizer、pretrain_model即可，之所以取名bert-crf，是因为转了一圈还是bert效果最好。
## 环境:
python=3.6  
transformer=4.2.2  
torch=1.6.0  
seqeval=1.2.2  
cuda=10.2  

## 数据
98人民日报1-6月，BIO标注，实体：时间、人、地点、机构   
训练集：145982条  
测试集：30827条  
验证集：31501条  
## 测试集结果
实体级别统计结果：  
p: 0.9725811534825086  
r: 0.9649969707037739  
f1: 0.968774218864963  

classification report: 
|        | precision | recall |  f1-score |   support| 
|  ----  | ----  |  ----  | ----  |  ----  |  
|         LOC    |   0.96  |    0.96  |    0.96  |   16781|
|         ORG    |   0.95   |   0.94   |   0.94   |   8852|
|         PER    |   0.99   |   0.99   |   0.99  |   12663|
|           T   |    0.99   |   0.98   |   0.98 |    12675|


