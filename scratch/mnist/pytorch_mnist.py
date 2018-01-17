import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
		self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
		self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
		self.fc1 = nn.Linear(64*3*3, 500)
		self.fc2 = nn.Linear(500, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = nnf.max_pool2d(x, 2, 2)
		x = nnf.relu(x)
		x = self.conv2(x)
		x = nnf.max_pool2d(x, 2, 2)
		x = nnf.relu(x)
		x = self.conv3(x)
		x = nnf.max_pool2d(x, 2, 2)
		x = nnf.relu(x)
		x = x.view(-1, 64*3*3)
		x = self.fc1(x)
		x = nnf.relu(x)	
		x = self.fc2(x)
		return nnf.log_softmax(x)

def torch_info():
	print('PyTorch {}'.format(torch.__version__))
	if torch.cuda.is_available():
		print('Current CUDA device is {}'.format(torch.cuda.current_device()))
	else:
		print('CUDA NOT available.')

def get_mnist(folder='./data', batch_size=128):
	kwargs = {'num_workers': 1, 'pin_memory': True}

	train_loader = torch.utils.data.DataLoader(
	    datasets.MNIST(folder, train=True, download=True,
	                   transform=transforms.Compose([
	                       transforms.ToTensor(),
	                       transforms.Normalize((0.1307,), (0.3081,))
	                   ])),
	    batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
	    datasets.MNIST(folder, train=False, transform=transforms.Compose([
	                       transforms.ToTensor(),
	                       transforms.Normalize((0.1307,), (0.3081,))
	                   ])),
	    batch_size=batch_size, shuffle=True, **kwargs)
	return train_loader, test_loader

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
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	
	torch.manual_seed(args.seed)
	if args.cuda:
	    torch.cuda.manual_seed(args.seed)

	return args

def train(epoch, model, train_loader, optimizer, args):
	with tqdm(desc='Train Epoch', total=len(train_loader.dataset), unit=' examples') as pbar:
		for batch_idx, (data, target) in enumerate(train_loader):
			if args.cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			optimizer.zero_grad()
			output = model(data)
			loss = nnf.nll_loss(output, target)
			loss.backward()
			optimizer.step()

			pbar.set_description('Train Epoch {:2}'.format(epoch))
			pbar.update(len(data))
			pbar.set_postfix(Loss=loss.data[0])			

def test(model, test_loader, optimizer, args):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += nnf.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
	torch_info()
	args = arguments()
	train_loader, test_loader = get_mnist(batch_size=args.batch_size)

	model = Net()
	if args.cuda:
		model.cuda()

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

	print(model)
	for epoch in range(args.epochs):
		train(epoch, model, train_loader, optimizer, args)
		test(model, test_loader, optimizer, args)

if __name__ == "__main__":
	main()