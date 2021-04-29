import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense,Dropout
from tqdm import tqdm
from keras.utils import plot_model
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--load_model',type=int,help='1 means loading，other means not loading ')
parser.add_argument('--epoch',type=int)
parser.add_argument('--sample_num',type=int,help='the number of the train samples')
parser.add_argument('--train_layer',type=int,help='the number of the train layers')
parser.add_argument('--en',type=str,help='true or other')

args = parser.parse_args()

config_path = '/home/hpe/project/zfy/offensive/bert/bert_config.json'
checkpoint_path = '/home/hpe/project/zfy/bert_model/multilingual_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/hpe/project/zfy/bert_model/multilingual_L-12_H-768_A-12/vocab.txt'
import os
import pandas as pd
root_path = '../code/examples/english/data'
file  = os.path.join(root_path,'olid-training-v1.0.tsv')
df = pd.read_csv(file,sep='\t')
test = df.rename(columns={'tweet': 'text', 'subtask_a': 'labels'})
data = test[['text', 'labels']]
len_num = data.shape[0]
texts = data['text'].tolist()
labels_ = data['labels'].tolist()
labels = list(set(data['labels'].tolist()))
num_classes = len(labels)

maxlen = 120
batch_size = 64

#建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        

#创建一个LossHistory的实例
history = LossHistory()

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if len(self.data[0]) == 2:
            for is_end, (text, label) in self.sample(random):
                token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if len(self.data[0]) == 3:
            for is_end, (texta, textb,label) in self.sample(random):
                token_ids, segment_ids = tokenizer.encode(texta, textb, maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total

from tqdm import tqdm

#土耳其
file = '../code/examples/turkish/data/offenseval-tr-training-v1.tsv'
df = pd.read_csv(file,sep='\t')
df.head(0)
test = df.rename(columns={'tweet': 'text', 'subtask_a': 'labels'})
data = test[['text', 'labels']]
len_num = data.shape[0]
texts = data['text'].tolist()
labels_ = data['labels'].tolist()
all_data = []
for i in range(0,len_num):
    a_data = (texts[i],labels.index(labels_[i]))

    all_data.append(a_data)
sample = 'turkish'
print(sample)
print('共有{}条数据'.format(len_num))
train_data = all_data[:int(len_num*0.9)] 
test_data = all_data[-1000:] 

def load_en():
    all_data = []
    root_path = '../code/examples/english/data'
    file  = os.path.join(root_path,os.listdir(root_path)[0])
    files = ['olid-training-v1.0.tsv','5993_solid.csv','3887_solid.csv']
    dfs = []
    df1 = pd.read_csv(os.path.join(root_path,files[0]),sep='\t').rename(columns={'tweet': 'text', 'subtask_a': 'labels'})[['text', 'labels']]
    df2 = pd.read_csv(os.path.join(root_path,files[1]),sep=',').rename(columns={'tweet': 'text', 'subtask_a': 'labels'})[['text', 'labels']]
    df3 = pd.read_csv(os.path.join(root_path,files[2]),sep=',').rename(columns={'tweet': 'text', 'subtask_a': 'labels'})[['text', 'labels']]
    dfs.append(df1)
    dfs.append(df2)
    df = pd.concat(dfs)
    len_num = df.shape[0]
    texts = df['text'].tolist()
    labels_ = df['labels'].tolist()
    labels = list(set(df['labels'].tolist()))
    num_classes = len(labels)
    for i in range(0,len_num):
        a_data = (texts[i],labels.index(labels_[i]))
        all_data.append(a_data)
    return all_data

from function import skl_eval

def eval_epoch():
    y_true = []
    y_pred = []
    for i,D in tqdm(enumerate(test_data)):
        y_true.append(test_data[i][1])
        token_ids, segment_ids = tokenizer.encode(test_data[i][0], maxlen=maxlen)
        y_pred.append(model.predict([[token_ids], [segment_ids]])[0].argmax())
    skl_eval(y_true,y_pred,labels)
def eval_epoch1():
    y_true = []
    y_pred = []
    for i,D in tqdm(enumerate(test_data)):
        y_true.append(test_data[i][1])
        token_ids, segment_ids = tokenizer.encode(test_data[i][0], maxlen=maxlen)
        y_pred.append(model.predict([[token_ids], [segment_ids]])[0].argmax())
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    a = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    f = f1_score(y_true, y_pred, average='macro')
    print([a,p,r,f])
    if args.en:
        en_ = args.en
    else:
        en_ = 'false'
    if args.sample_num:
        nums_ = args.sample_num
    else:
        nums_ = len_num
    result = pd.DataFrame(data=[[sample,args.load_model,en_,nums_,a,p,r,f]])
    result.to_csv('results.csv',mode='a',sep=',',header=None)
        
class Evaluator_2(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
#         val_acc = evaluate(valid_generator)
#         if val_acc > self.best_val_acc:
#             self.best_val_acc = val_acc
#         print(
#             u'val_acc: %.5f, best_val_acc: %.5f\n' %
#             (val_acc, self.best_val_acc)
#         )
#         eval_epoch()
        eval_epoch1()


if args.sample_num:
    import random
    train_num = args.sample_num
    off_data = [data for data in train_data if data[1] == 1]
    not_data = [data for data in train_data if data[1] == 0]
    train_data = off_data[:train_num//2]+not_data[:train_num//2]
    random.shuffle(train_data)   
#     print(len(off_data),len(off_data))
    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(off_data[train_num//2:]+not_data[train_num//2:],batch_size)
    print('train num ：{}\neval num：{}\ndev num：{}'.format(len(train_data),len(off_data[train_num//2:]+not_data[train_num//2:]),len(test_data)))
else:
    train_num = int(len(train_data)*0.9)  
    import random
    random.shuffle(train_data) 
#     print('train num ：{}\neval num：{}\ndev num：{}'.format(5000,1000,len(test_data)))
#     train_generator = data_generator(train_data[:5000], batch_size)
#     valid_generator = data_generator(train_data[1000:],batch_size)
    print('train num ：{}\neval num：{}\ndev num：{}'.format(train_num,len(train_data)-train_num,len(test_data)))
    train_generator = data_generator(train_data[:train_num], batch_size)
    valid_generator = data_generator(train_data[train_num:],batch_size)
    
if args.en == 'true':
    
    print('加入无监督机器翻译')
#     train_data = train_data+load_en()[:13000]
    train_data = train_data+load_en()
    
    train_generator = data_generator(train_data, batch_size)
    
else:
    print('直接学习')
# 转换数据集


evaluator = Evaluator_2()
from model import build_model
model = build_model(num_classes)
if args.load_model == 1:
    print('加载模型')
    model.load_weights('model/best_model.weights')
print('开始训练')
if args.train_layer:
    layers_num = args.train_layer
    for i,layer in enumerate(model.layers):
    #     print(dir(layer))
        if i<len(model.layers)-6:
            layer.trainable = False
        else:
            layer.trainable = True
        print(layer.trainable)
else:
    pass
model.fit(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=args.epoch,
    callbacks=[evaluator,history]
)
y_true = []
y_pred = []
print('turkish')
# from function import skl_eval
# for i,D in tqdm(enumerate(test_data)):
#     y_true.append(test_data[i][1])
#     token_ids, segment_ids = tokenizer.encode(test_data[i][0], maxlen=maxlen)
#     y_pred.append(model.predict([[token_ids], [segment_ids]])[0].argmax())
# skl_eval(y_true,y_pred,labels)