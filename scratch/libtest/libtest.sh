# Python3 Theano
echo ""
echo "--- Python3 Theano ---"
# python3 -c "import mxnet as mx; print('Python3 MxNet version: {} '.format(mx.__version__))"
# python3 -c "import mxnet as mx; a = mx.nd.ones((2, 3), mx.gpu()); b = a * 2 + 1; print(b.asnumpy())"

# Python3 MxNet
echo ""
echo "--- Python3 MxNet ---"
python3 -c "import mxnet as mx; print('Python3 MxNet version: {} '.format(mx.__version__))"
python3 -c "import mxnet as mx; a = mx.nd.ones((2, 3), mx.gpu()); b = a * 2 + 1; print(b.asnumpy())"

# Python2 Caffe2
echo ""
echo "--- Python2 Caffe ---"
python2 -c "import caffe; print('Python2 Caffe version: {} '.format(caffe.__version__))"

# Python2 Caffe2
echo ""
echo "--- Python2 Caffe2 ---"
python2 -c "from caffe2.python import core; print('Python2 Caffe2 version: {} '.format(core.__version__))"

# Python3 CNTK
echo ""
echo "--- Python3 CNTK ---"
python3 -c "import cntk; print('Python3 CNTK version: {} '.format(cntk.__version__))"

# Python3 PyTorch
echo ""
echo "--- Python3 PyTorch ---"
python3 -c "import torch; print('Python3 PyTorch version: {} CUDA {}'.format(torch.__version__, 'available' if torch.cuda.is_available() else 'NOT availbale'))"


# OpenCV Python3
echo ""
echo "--- Python3 OpenCV ---"
python3 -c "import cv2; print('Python3 OpenCV version: {} {}'.format(cv2.__version__, cv2.__file__))"

# OpenCV Python2
echo ""
echo "--- Python2 OpenCV ---"
python2 -c "import cv2; print('Python2 OpenCV version: {} {}'.format(cv2.__version__, cv2.__file__))"

# Tensorflow
echo ""
echo "--- Python3 Tensorflow ---"
python3 -c "import tensorflow as tf; print('Tensorflow version: {}'.format(tf.__version__))"
python3 -c "import tensorflow as tf; hello = tf.constant('Hello, TensorFlow!'); print(tf.Session().run(hello))"
