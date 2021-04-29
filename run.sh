#! /bin/bash

python english.py --base=en train_layer=1 #第一次选择高资源英语数据训练攻击性语言检测器，train_layer表示选择训练多少层


#以下为迁移到低资源语种上的训练
'''
如：
训练丹麦语言的攻击性语言检测器：
python danish.py --load_model=1 --epoch=20 --en=true

'''