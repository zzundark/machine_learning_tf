import tensorflow as tf
import config as cfg

#모델 
class softmax_regression(object):	
		
	def __init__(self):
		self.num_class = cfg.num_class
		self.images = tf.placeholder(tf.float32, shape=[None,cfg.image_size,cfg.image_size,3], name = 'x_image') 
		self.labels = tf.placeholder(tf.float32, shape=[None, self.num_class], name = 'y_target')
		self.logits = self.build_network(x_image=self.images,num_class=self.num_class)
		
		
	def build_network(self,x_image,num_class):
	
		x_image_shape = x_image.get_shape().as_list()
		x_image=tf.reshape(x_image, [-1, x_image_shape[1]*x_image_shape[2]*x_image_shape[3]])
		
		x=x_image
		shape = x.get_shape().as_list()
		input_num= shape[1]
		
		xavier_initializer = tf.contrib.layers.xavier_initializer()
		
		W = tf.Variable(xavier_initializer([input_num, num_class]))
		b = tf.Variable(xavier_initializer([num_class]),name="bias")

		return tf.matmul(x, W) + b		
		
		
		
