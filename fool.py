from lcztools import load_network, LeelaBoard

net = load_network('weights.txt.gz', 'pytorch_cuda')

board = LeelaBoard()

policy, value = net.evaluate(board)

print(policy)

