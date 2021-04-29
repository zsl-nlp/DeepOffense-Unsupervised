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

#multilingual_L-12_H-768_A-12
config_path = '/home/hpe/project/zfy/offensive/bert/bert_config.json'
checkpoint_path = '/home/hpe/project/zfy/bert_model/multilingual_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/hpe/project/zfy/bert_model/multilingual_L-12_H-768_A-12/vocab.txt'

def bert_model():
    # Load the pre-training model
    return build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
    )
def build_model(num_classes):
    bert = bert_model()
    dropout = 0.1
    output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = Dense(
        units=num_classes,
        activation='softmax',
        kernel_initializer=bert.initializer
    )(output)
    model = keras.models.Model(bert.model.input, output)
    model.summary()

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(2e-5),
        metrics=['sparse_categorical_accuracy'],
    )
    return model