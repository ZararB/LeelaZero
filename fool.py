from lcztools import load_network, LeelaBoard
from keras_net import KerasNet
net = KerasNet()
board = LeelaBoard()

policy, value = net.evaluate(board)

print(policy)

