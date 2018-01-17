import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('str-arg', 'default_value', 'String argument example')
flags.DEFINE_integer('int-arg', 1, 'Integer argument example')
flags.DEFINE_float('float-arg', 1.0, 'Float argument example')


def tf_info():
	print('Tensorflow {}'.format(tf.__version__))

def main(_):
	tf_info()
	print(FLAGS.str_arg)
	print(FLAGS.int_arg)
	print(FLAGS.float_arg)

if __name__ == "__main__":
	tf.app.run()
