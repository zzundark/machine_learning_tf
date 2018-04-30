
import tensorflow as tf
import config as cfg
import inputs as input
import os
import sys
from timer import Timer
import numpy as np

#solver 클래스 : 모델의 학습, 테스트를 관리 
class solver(object):

	def __init__(self,net,input,logs_path):
		if not os.path.exists(logs_path):
			os.makedirs(logs_path)
		self.logs_path=logs_path
		self.net = net
		self.input=input
		

		#GPU 설정
		self.config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True))
		self.sess = tf.Session(config=self.config)
	
		#sigle op에 모든 summery merge함 
		self.summary_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(logs_path, flush_secs=60)
		self.writer.add_graph(self.sess.graph)	
		

	def train_and_test(self):
		
		train_timer = Timer()
		load_timer = Timer()
		f=open(os.path.join(self.logs_path, 'log.txt'), 'w')
		display_step=1
		min_distance=0
		min_index=0
		min_idx=0
		init_vars = tf.global_variables_initializer()
		self.sess.run(init_vars)
		
		x_train,y_train, x_val,y_val,x_test,y_test=self.input.get_dataset()
		size=len(y_train)
		xtr_images,xtr_labels = self.input.get(size,x_train,y_train,0)
		
		iteration=int(len(y_test)/float(cfg.batch_size))
		accuracy=0
		for step in range(iteration):		
			xts_images,xts_labels = self.input.get(cfg.batch_size,x_test,y_test,step)
			
			feed_dicts = {self.net.xtr_image: xtr_images,
							self.net.xtr_labels: xtr_labels,
							self.net.xts_image: xts_images,
							}
			test_output = []
			actual_vals = []		
			predictions = self.sess.run(self.net.logits, feed_dict=feed_dicts)
			test_output.extend(predictions)
			actual_vals.extend(np.argmax(xts_labels, axis=1))
			acc = sum([1./float(cfg.batch_size) for i in range(cfg.batch_size) if test_output[i]==actual_vals[i]])	
			accuracy+=acc
			print('Accuracy on test set: ' +"("+str(step)+"/"+str(iteration)+")/"+str(accuracy/float(iteration)))
		
		
		
		accuracy=accuracy/float(iteration)
		line=('Accuracy on test set: ' + str(accuracy))		
		print(line)
		f.write(line)
		f.close()
		