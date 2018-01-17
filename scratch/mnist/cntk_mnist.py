import argparse
from tqdm import tqdm

from data.data import *

import cntk
from cntk.device import try_set_default_device, gpu


def cntk_info():
	print('CNTK {}'.format(cntk.__version__))
	ret = try_set_default_device(gpu(0))
	if ret:
		print('GPU device set to 0')

def get_data_tmp(folder='./data'):
	prepare_data(folder)

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

class Net():

	def __init__(self):
		self.input_dim_model = (1, 28, 28)    # images are 28 x 28 with 1 channel of color (gray)
		self.input_dim = 28*28                # used by readers to treat input data as a vector
		self.num_output_classes = 10

		self.x = cntk.input_variable(self.input_dim_model)
		self.y = cntk.input_variable(self.num_output_classes)
	
		self.z = self.create_model(self.x/255)
		self.loss, self.errs = self.create_criterion_function(self.z, self.y)

	def create_model(self, features):
		with cntk.layers.default_options(init=cntk.glorot_uniform(), activation=cntk.relu):
			h = features
			
			h = cntk.layers.Convolution2D(filter_shape=(3,3), 
										num_filters=16, 
										strides=(1,1), 
										pad=True, name="first_conv")(h)
			h = cntk.layers.MaxPooling(filter_shape=(2,2), 
										strides=(2,2), name="first_max")(h)
			h = cntk.layers.Convolution2D(filter_shape=(3,3), 
										num_filters=32, 
										strides=(1,1), 
										pad=True, name="second_conv")(h)
			h = cntk.layers.MaxPooling(filter_shape=(2,2), 
										strides=(2,2), name="second_max")(h)
			h = cntk.layers.Convolution2D(filter_shape=(3,3), 
										num_filters=64, 
										strides=(1,1), 
										pad=True, name="third_conv")(h)
			h = cntk.layers.MaxPooling(filter_shape=(2,2), 
										strides=(2,2), name="third_max")(h)	
			h = cntk.layers.Dense(500, name="fc0")(h)
			r = cntk.layers.Dense(self.num_output_classes, activation = None, name="classify")(h)			
			return r

	def create_criterion_function(self, model, labels):
	    loss = cntk.cross_entropy_with_softmax(model, labels)
	    errs = cntk.classification_error(model, labels)
	    return loss, errs # (model, labels) -> (loss, error metric)

# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):
    
    ctf = cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(
          labels=cntk.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
          features=cntk.io.StreamDef(field='features', shape=input_dim, is_sparse=False)))
                          
    return cntk.io.MinibatchSource(ctf,
        randomize = is_training, max_sweeps = cntk.io.INFINITELY_REPEAT)

def test_data_available(folder):
	# Ensure the training and test data is available for this tutorial.
	# We search in two locations in the toolkit for the cached MNIST data set.
	
	data_found=False # A flag to indicate if train/test data found in local cache
	for data_dir in [os.path.join("..", "Examples", "Image", "DataSets", "MNIST"),
	                 os.path.join("data", "MNIST"),
	                 os.path.join(".", "data")]:
	    
	    train_file=os.path.join(data_dir, "Train-28x28_cntk_text.txt")
	    test_file=os.path.join(data_dir, "Test-28x28_cntk_text.txt")
	    
	    if os.path.isfile(train_file) and os.path.isfile(test_file):
	        data_found=True
	        break
	        
	if not data_found:
	    raise ValueError("Please generate the data by completing CNTK 103 Part A")
	    
	print("Data directory is {0}".format(data_dir))

# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
        
    return mb, training_loss, eval_error	

def train(epoch, model, train_reader, trainer, args):
	# Map the data streams to the input and labels.
	input_map={
		model.y  : train_reader.streams.labels,
		model.x  : train_reader.streams.features
	}

	training_progress_output_freq = 640
	num_minibatches_to_train = int(np.ceil(4*60000/args.batch_size))

	with tqdm(desc='Train Epoch', total=num_minibatches_to_train, unit=' examples') as pbar:
		for batch_idx in range(num_minibatches_to_train):
			# Read a mini batch from the training data file
			data=train_reader.next_minibatch(args.batch_size, input_map=input_map) 
			trainer.train_minibatch(data)
			#print_training_progress(trainer, batch_idx, training_progress_output_freq, verbose=1)
			loss = model.loss.eval(data)[-1][0]

			pbar.set_description('Train Epoch {:2}'.format(epoch))
			pbar.update(1)
			pbar.set_postfix(Loss=loss) 			

def test(epoch, model, test_reader, trainer, args):
	num_minibatches_to_test = int(np.ceil(10000/args.batch_size))

	test_input_map = {
	    model.y  : test_reader.streams.labels,
	    model.x  : test_reader.streams.features
	}	

	errors = 0

	for batch_idx in range(num_minibatches_to_test):
		# Read a mini batch from the training data file
		data=test_reader.next_minibatch(args.batch_size, input_map=test_input_map)
		eval_error = trainer.test_minibatch(data)
		errors += eval_error * args.batch_size

	# Average of evaluation errors of all test minibatches
	total_sample_size = num_minibatches_to_test * args.batch_size
	correct = total_sample_size - errors
	print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
		int(correct), total_sample_size,
		100. * correct / total_sample_size))

def main():
	cntk_info()
	args = arguments()
	get_data_tmp(folder='./data')
	test_data_available(folder='./data')

	model = Net()

	## CNTK code without formatting
	train_reader = create_reader(ctf_train_file, True, model.input_dim, model.num_output_classes)
	test_reader = create_reader(ctf_test_file, False, model.input_dim, model.num_output_classes)

	# Print the output shapes / parameters of different components
	print("Output Shape of the first convolution layer:", model.z.first_conv.shape)
	print("Bias value of the last dense layer:", model.z.classify.b.value)

	# Number of parameters in the network
	cntk.logging.log_number_of_parameters(model.z)

	# Instantiate the trainer object to drive the model training
	learning_rate = 0.2
	lr_schedule = cntk.learning_rate_schedule(learning_rate, cntk.UnitType.minibatch)
	learner = cntk.sgd(model.z.parameters, lr_schedule)
	trainer = cntk.Trainer(model.z, (model.loss, model.errs), [learner])

	for epoch in range(args.epochs):
		train(epoch, model, train_reader, trainer, args)
		test(epoch, model, test_reader, trainer, args)

if __name__ == "__main__":
	main()