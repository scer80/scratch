import mxnet
import logging

import argparse

class Net():

	def __init__(self):
		self.data = mxnet.sym.var('data')
		# first conv layer
		self.conv1 = mxnet.sym.Convolution(data=self.data, kernel=(3,3), num_filter=16)
		self.relu1 = mxnet.sym.Activation(data=self.conv1, act_type="relu")
		self.pool1 = mxnet.sym.Pooling(data=self.relu1, pool_type="max", kernel=(2,2), stride=(2,2))
		# second conv layer
		self.conv2 = mxnet.sym.Convolution(data=self.pool1, kernel=(3,3), num_filter=32)
		self.relu2 = mxnet.sym.Activation(data=self.conv2, act_type="relu")
		self.pool2 = mxnet.sym.Pooling(data=self.relu2, pool_type="max", kernel=(2,2), stride=(2,2))
		# third conv layer
		self.conv3 = mxnet.sym.Convolution(data=self.pool2, kernel=(3,3), num_filter=64)
		self.relu3 = mxnet.sym.Activation(data=self.conv3, act_type="relu")
		self.pool3 = mxnet.sym.Pooling(data=self.relu3, pool_type="max", kernel=(2,2), stride=(2,2))		
		# first fullc layer
		self.flatten = mxnet.sym.flatten(data=self.pool2)
		self.fc1 = mxnet.symbol.FullyConnected(data=self.flatten, num_hidden=500)
		self.relu4 = mxnet.sym.Activation(data=self.fc1, act_type="relu")
		# second fullc
		self.fc2 = mxnet.sym.FullyConnected(data=self.relu4, num_hidden=10)
		# softmax loss
		self.net = mxnet.sym.SoftmaxOutput(data=self.fc2, name='softmax')


def get_mnist(folder='./data', batch_size=128):
	mnist = mxnet.test_utils.get_mnist()	
	train_iter = mxnet.io.NDArrayIter(mnist['train_data']/1.0, mnist['train_label'], batch_size, shuffle=True)
	test_iter = mxnet.io.NDArrayIter(mnist['test_data']/1.0, mnist['test_label'], batch_size)

	return train_iter, test_iter

def mxnet_info():
	print('MxNet {}'.format(mxnet.__version__))

def arguments():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
	                    help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
	                    help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
	                    help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
	                    help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
	                    help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
	                    help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
	                    help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
	                    help='how many batches to wait before logging training status')
	args = parser.parse_args()


	return args

def main():
	mxnet_info()
	args = arguments()
	train_iter, test_iter = get_mnist(args.batch_size)


	logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

	model = Net()

	# create a trainable module on GPU 0
	lenet_model = mxnet.mod.Module(symbol=model.net, context=mxnet.gpu())
	# train with the same
	lenet_model.fit(train_iter,
					eval_data=test_iter,
					optimizer='sgd',
					optimizer_params={'learning_rate':0.1},
					eval_metric='acc',
					batch_end_callback = mxnet.callback.Speedometer(args.batch_size, 100),
					num_epoch=args.epochs)	

if __name__ == "__main__":
	main()

