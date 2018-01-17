import argparse
from tqdm import tqdm

import tensorflow as tf

class Net():
	
	def __init__(self):
		self.model = tf.contrib.keras.models.Sequential()
		self.model.add(tf.contrib.keras.layers.Conv2D(16, 3, padding='same', activation=tf.contrib.keras.activations.relu, input_shape=[28, 28, 1]))
		self.model.add(tf.contrib.keras.layers.MaxPool2D(pool_size=2, strides=2))
		self.model.add(tf.contrib.keras.layers.Conv2D(32, 3, padding='same', activation=tf.contrib.keras.activations.relu))
		self.model.add(tf.contrib.keras.layers.MaxPool2D(pool_size=2, strides=2))
		self.model.add(tf.contrib.keras.layers.Conv2D(64, 3, padding='same', activation=tf.contrib.keras.activations.relu))
		self.model.add(tf.contrib.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'))
		self.model.add(tf.contrib.keras.layers.Flatten())
		self.model.add(tf.contrib.keras.layers.Dense(500, activation=tf.contrib.keras.activations.relu))
		self.model.add(tf.contrib.keras.layers.Dense(10, activation=tf.contrib.keras.activations.softmax))

		self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])			
	

def tf_info():
	print('Tensorflow {}'.format(tf.__version__))

def get_mnist(folder='./data'):
	# Load pre-shuffled MNIST data into train and test sets
	(X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
	
	X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
	X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
	
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255.0
	X_test /= 255.0
	Y_train = tf.contrib.keras.utils.to_categorical(y_train, 10)
	Y_test = tf.contrib.keras.utils.to_categorical(y_test, 10)

	return X_train, Y_train, X_test, Y_test

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
	X_train, Y_train, X_test, Y_test = get_mnist()
	model = Net()

	print(model.model.summary())

	model.model.fit(X_train, Y_train, 
          batch_size=args.batch_size, epochs=args.epochs, verbose=1)
	score = model.model.evaluate(X_test, Y_test, verbose=0)
	loss, accuracy = score[0], score[1]
	print('Accuracy {:.2f}'.format(accuracy))	

if __name__ == "__main__":
	main()
