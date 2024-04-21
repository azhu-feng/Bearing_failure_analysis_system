from django.test import TestCase
from MLfiles import CNN
from MLfiles import LSTM
from MLfiles import BiLSTM

state = 0
# dataclass = 'CRWU'
dataclass = 'JNU'
# dataclass = 'MFPT'
result1, acc1 = CNN.CNN(state, dataclass)
result2, acc2 = LSTM.Lstm(state, dataclass)
result3, acc3 = BiLSTM.BiLstm(state, dataclass)
print(result1, acc1)
print(result2, acc2)
print(result3, acc3)

# epochs:8
# 77.98 88.69 82.74
# 68.88 98.94 99.70
# 82.28 98.31 98.73
