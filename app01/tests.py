from django.test import TestCase
from MLfiles import CNN
from MLfiles import LSTM
from MLfiles import BiLSTM
from MLfiles import Transfomer

state = 0
# dataclass = 'CWRU'
dataclass = 'JNU'
# dataclass = 'MFPT'
result1, acc1 = CNN.CNN(state, dataclass)
result2, acc2 = LSTM.Lstm(state, dataclass)
result3, acc3 = BiLSTM.BiLstm(state, dataclass)
result4, acc4 = Transfomer.trans(state, dataclass)
print(result1, acc1)
print(result2, acc2)
print(result3, acc3)
print(result4, acc4)

# epoch 20
# CWRU      69.94   95.83   95.24   62.50
# JNU       75.76   81.06   80.91   80.91
# MPFT      39.24   99.58   97.48   77.64
