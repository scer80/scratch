import argparse
from tqdm import tqdm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class Net():
	
	def __init__(self):
		self.x = tf.placeholder(tf.float32, shape=[None, 784])
		self.labels = tf.placeholder(tf.float32, shape=[None, 10])

		self.x_reshaped = tf.reshape(self.x, [-1, 28, 28, 1])

		self.layer = tf.contrib.layers.conv2d(self.x_reshaped, 16, 3, padding='same', activation_fn=tf.nn.relu)
		self.layer = tf.contrib.layers.max_pool2d(self.layer, kernel_size=2, stride=2)

		self.layer = tf.contrib.layers.conv2d(self.layer, 32, 3, padding='same', activation_fn=tf.nn.relu)
		self.layer = tf.contrib.layers.max_pool2d(self.layer, kernel_size=2, stride=2)

		self.layer = tf.contrib.layers.conv2d(self.layer, 64, 3, padding='same', activation_fn=tf.nn.relu)
		self.layer = tf.contrib.layers.max_pool2d(self.layer, kernel_size=2, stride=2, padding='same')

		self.layer = tf.contrib.layers.flatten(self.layer)

		self.layer = tf.contrib.layers.fully_connected(self.layer, 500, activation_fn=tf.nn.relu)
		self.layer = tf.contrib.layers.fully_connected(self.layer, 10, activation_fn=None)

		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.layer))
		self.correct_prediction = tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.layer, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

def tf_info():
	print('Tensorflow {}'.format(tf.__version__))

def get_mnist(folder='./data'):

	mnist = input_data.read_data_sets(folder, one_hot=True)

	return mnist.train, mnist.validation, mnist.test

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
	parser.add_argument('--log-interval', type=int, default=100, metavar='N',
	                    help='how many batches to wait before logging training status')
	args = parser.parse_args()

	return args	

def main():
	tf_info()
	args = arguments()
	train, validation, test = get_mnist()
	model = Net()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(args.epochs):
			with tqdm(desc='Epoch', total=train.num_examples, unit=' examples') as pbar:
				for batch_idx in range(int(train.num_examples//args.batch_size)):
					batch = train.next_batch(args.batch_size)
					_, train_loss, train_accuracy = sess.run([model.train_step, model.cross_entropy, model.accuracy], 
							feed_dict={model.x: batch[0], model.labels: batch[1]})

					pbar.set_description('Epoch {:2}'.format(epoch))
					pbar.update(args.batch_size)
					pbar.set_postfix(Loss='{:.2f}'.format(train_loss), Accuracy='{:.2f}'.format(train_accuracy))

			valid_loss, valid_accuracy = sess.run([model.cross_entropy, model.accuracy], 
				feed_dict={model.x: validation.images, model.labels: validation.labels})
			print('Validation loss {:.2f} accuracy {:.2f}'.format(valid_loss,valid_accuracy))

		num_test_batches = int(test.num_examples//args.test_batch_size)
		test_accuracy = 0.0
		for batch_idx in range(num_test_batches):
			batch = test.next_batch(args.batch_size)
			test_accuracy += model.accuracy.eval(feed_dict={model.x: batch[0], model.labels: batch[1]})
		test_accuracy /= num_test_batches
		print('Test accuracy {:.2f}'.format(test_accuracy))

if __name__ == "__main__":
	main()
