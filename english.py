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
parser.add_argument('--base',type=str,help='en,tr,el')
parser.add_argument('--train_layer',type=int,help='the number of the train layers')


args = parser.parse_args()
#
config_path = '/home/hpe/project/zfy/offensive/bert/bert_config.json'
checkpoint_path = '/home/hpe/project/zfy/bert_model/multilingual_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/hpe/project/zfy/bert_model/multilingual_L-12_H-768_A-12/vocab.txt'

import os
import pandas as pd
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



if args.base == 'en':
    all_data = []
    for i in range(0,len_num):
        a_data = (texts[i],labels.index(labels_[i]))
        all_data.append(a_data)
    train_num = int(len_num*0.85)   
    train_data = all_data[:train_num]
    test_data = all_data[train_num:]
    print('共有{}条数据\n训练集有{}条数据\n测试集有{}条数据'.format(len_num,len(train_data),len(test_data)))
    print(train_data[:2])
    print(test_data[:2])
elif args.base == 'tr':
    #土耳其
    file = '../code/examples/turkish/data/offenseval-tr-training-v1.tsv'
    df = pd.read_csv(file,sep='\t')
    df.head(0)
    test = df.rename(columns={'tweet': 'text', 'subtask_a': 'labels'})
    data = test[['text', 'labels']]
    len_num = data.shape[0]
    texts = data['text'].tolist()
    labels_ = data['labels'].tolist()
    test_data = []
    off_num =0
    not_num = 0
    for i in range(0,len_num):
        a_data = (texts[i],labels.index(labels_[i]))
        test_data.append(a_data)
        if labels.index(labels_[i]) == 0:
            not_num += 1
        else:
            off_num += 1
    sample_name = 'turkish'
    all_num = len_num
    sample = (sample_name,all_num,off_num,not_num)
    print(sample)
    print('共有{}条数据'.format(len_num))
    all_data = test_data
    train_num = int(len(all_data)*0.85)   
    train_data = all_data[:train_num]
    test_data = all_data[train_num:]
    print('共有{}条数据\n训练集有{}条数据\n测试集有{}条数据'.format(len_num,len(train_data),len(test_data)))
    print(train_data[:2])
    print(test_data[:2])
elif args.base == 'el':
    #希腊语
    file = '../code/examples/greek/data/offenseval-greek-training-v1.tsv'
    df = pd.read_csv(file,sep='\t')
    test = df.rename(columns={'tweet': 'text', 'subtask_a': 'labels'})
    data = test[['text', 'labels']]
    len_num = data.shape[0]
    texts = data['text'].tolist()
    labels_ = data['labels'].tolist()
    test_data = []
    off_num =0
    not_num = 0
    for i in range(0,len_num):
        a_data = (texts[i],labels.index(labels_[i]))
        test_data.append(a_data)
        if labels.index(labels_[i]) == 0:
            not_num += 1
        else:
            off_num += 1
    sample_name = 'greek'
    all_num = len_num
    sample = (sample_name,all_num,off_num,not_num)
    print(sample)
    print('共有{}条数据'.format(len_num))
    all_data = test_data
    train_num = int(len(all_data)*0.85)   
    train_data = all_data[:train_num]
    test_data = all_data[train_num:]
    print('共有{}条数据\n训练集有{}条数据\n测试集有{}条数据'.format(len_num,len(train_data),len(test_data)))
    print(train_data[:2])
    print(test_data[:2])
else:
    raise Exception('Your base is {},the base must be:"en"、"tr"or"el"'.format({args.base}))
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
# 转换数据集
val_rate = 0.1
split_num = int(len(train_data)*(1-val_rate))
print(len(train_data),split_num)

train_generator = data_generator(train_data[:split_num], batch_size)
# print(next(train_generator.forfit()))
# valid_generator = data_generator(valid_data, batch_size)
valid_generator = data_generator(train_data[split_num:],batch_size)

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            
            model.save_weights('model/best_model.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )

        
from model import build_model
model = build_model(num_classes)
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
do_train = True
if do_train:
    
    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[evaluator,history]
    )

else:

    model.load_weights('model/best_model.weights')

def skl_eval(y_true, y_pred):


    from sklearn.metrics import accuracy_score,confusion_matrix

    accuracy_score(y_true, y_pred)#accuracy
    print(confusion_matrix(y_true, y_pred))#混淆矩阵
    from sklearn.metrics import classification_report
    target_names = [labels[0], labels[1]]
    print(classification_report(y_true, y_pred, target_names=target_names))    


from tqdm import tqdm
#英语

y_true = []
y_pred = []
print('english')
for i,D in tqdm(enumerate(test_data)):
    y_true.append(test_data[i][1])
    token_ids, segment_ids = tokenizer.encode(test_data[i][0], maxlen=maxlen)
    y_pred.append(model.predict([[token_ids], [segment_ids]])[0].argmax())
skl_eval(y_true,y_pred)