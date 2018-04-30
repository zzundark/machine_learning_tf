import tensorflow as tf
import config as cfg
from tensorflow.contrib.factorization import KMeans

#모델 
class k_mean(object):	
		
	def __init__(self):
		self.num_class = cfg.num_class
		self.images = tf.placeholder(tf.float32, shape=[None,cfg.image_size,cfg.image_size,3], name = 'x_image') 
		self.labels = tf.placeholder(tf.float32, shape=[None, cfg.num_class], name = 'y_target')
		self.k=10
		self.logits = self.build_network(x_image=self.images,num_class=self.num_class)

		
	def build_network(self,x_image,num_class):
	
		x_image_shape = x_image.get_shape().as_list()
		x_image=tf.reshape(x_image, [-1, x_image_shape[1]*x_image_shape[2]*x_image_shape[3]])
		x=x_image
		
		# K-Means 파라미터
		x = KMeans(inputs=x, num_clusters=self.k, distance_metric='cosine',
					use_mini_batch=True)

		return x		
		
		
		
