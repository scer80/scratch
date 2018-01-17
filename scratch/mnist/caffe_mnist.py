import caffe


def caffe_info():
	print('Caffe {}'.format(caffe.__version__))

def main():
	caffe_info()

if __name__ == "__main__":
	main()