import tensorflow as tf
import config as cfg
from tensorflow.contrib.factorization import KMeans

#모델 
class k_nn(object):	
		
	def __init__(self):
		self.num_class = cfg.num_class
		self.xtr_image = tf.placeholder(tf.float32, shape=[None,cfg.image_size,cfg.image_size,3], name = 'xtr_image') 
		self.xts_image = tf.placeholder(tf.float32, shape=[None,cfg.image_size,cfg.image_size,3], name = 'xts_image') 
		
		self.xtr_labels = tf.placeholder(tf.float32, shape=[None, cfg.num_class], name = 'xtr_target')
		self.logits = self.build_network(xtr_image=self.xtr_image,
										xts_image=self.xts_image,num_class=self.num_class)

		
	def build_network(self,xtr_image,xts_image,num_class):
	
		x_image_shape = xtr_image.get_shape().as_list()
		x_image=tf.reshape(xtr_image, [-1, x_image_shape[1]*x_image_shape[2]*x_image_shape[3]])
		x_data_train=x_image
	
		x_image_shape = xts_image.get_shape().as_list()
		x_image=tf.reshape(xts_image, [-1, x_image_shape[1]*x_image_shape[2]*x_image_shape[3]])
		x_data_test=x_image


		k = 1
		distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), axis=2)
		# Get min distance index (Nearest neighbor)
		top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
		prediction_indices = tf.gather(self.xtr_labels, top_k_indices)

		# Predict the mode category
		count_of_predictions = tf.reduce_sum(prediction_indices, axis=1)
		prediction = tf.argmax(count_of_predictions, axis=1)		

		return prediction		
		
		
		
