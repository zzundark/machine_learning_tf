
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
		
		training_graph=self.net.logits.training_graph()
		# k_mean 그래프 생성 
		if len(training_graph) > 6: # Tensorflow 1.4+
			(self.all_scores, cluster_idx, self.scores, self.cluster_centers_initialized,
			 self.cluster_centers_var, self.init_op, self.train_op) = training_graph
		else:
			(self.all_scores, cluster_idx, self.scores, self.cluster_centers_initialized,
			 self.init_op, self.train_op) = training_graph
			 
			 
		self.avg_distance = tf.reduce_mean(self.scores)
		self.cluster_idx = cluster_idx[0]				
		self.idx=None
		
		self.t_vars = tf.trainable_variables()
		
		#웨이트 저장 
		self.saver = tf.train.Saver(max_to_keep=None,var_list=self.t_vars)
		
		#GPU 설정
		self.config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True))
		self.sess = tf.Session(config=self.config)
	
		#sigle op에 모든 summery merge함 
		self.summary_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(logs_path, flush_secs=60)
		self.writer.add_graph(self.sess.graph)	
		
		
	#loss와 auccuracy 연산
	def cal_loss_accuracy(self,epoch,total_size,batch_size,data,label,loss_tag,accuracy_tag):
		cal_loss=0
		cal_acc=0
		
		assert (total_size > batch_size), '배치 사이즈(%d)가 데이터 셋 크기(%d)보다 작습니다'%(batch_size,total_size)
		
		# 각 중심점을 라벨에 할당
		counts = np.zeros(shape=(self.net.k, cfg.num_class))	
		for i in range(len(self.idx)):
			c_label=np.zeros((cfg.num_class))
			c_label[self.input.y_train[i]]=1.0
			counts[self.idx[i]] += c_label
			
			
		# 중심점을 할당한 라벨중에서 빈번한 것을 할당
		labels_map = [np.argmax(c) for c in counts]
		labels_map = tf.convert_to_tensor(labels_map)

		# 평가
		# 룩업(Lookup): centroid id -> label
		cluster_label = tf.nn.embedding_lookup(labels_map, self.cluster_idx)
		
		# 정확도 계산
		correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(self.net.labels, 1), tf.int32))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	
				
		total_iteration = int(total_size/batch_size)
		
		for step in range(total_iteration):
			images,labels = self.input.get(cfg.batch_size,data,label,step)
			feed_dicts = {self.net.images: images,
							self.net.labels: labels}
			acc = self.sess.run(
				accuracy,
				feed_dict=feed_dicts)
			cal_acc += acc
			
		cal_acc/=float(total_iteration)
		
		_summary = tf.Summary(value=[tf.Summary.Value(tag=accuracy_tag, simple_value=cal_acc)])

		self.writer.add_summary(_summary,epoch)
		return cal_loss,cal_acc
		
	def train_and_test(self):
		
		train_timer = Timer()
		load_timer = Timer()
		f=open(os.path.join(self.logs_path, 'log.txt'), 'w')
		display_step=1
		min_distance=0
		min_index=0
		min_idx=0
		
		init_vars = tf.global_variables_initializer()
			
		x_train,y_train, x_val,y_val,x_test,y_test=self.input.get_dataset()
		size=len(y_train)
		images,labels = self.input.get(size,x_train,y_train,0)
		total_iteraton=size
		
		
		feed_dicts = {self.net.images: images}		
		# Run the initializer
		self.sess.run(init_vars, feed_dict=feed_dicts)
		self.sess.run(self.init_op,feed_dict=feed_dicts)		
		
		#epoch 루프
		for epoch in range(cfg.max_epoch+1):
			#학습
			_, d = self.sess.run([self.train_op, self.avg_distance],
							feed_dict=feed_dicts)
		


			# 스텝마다 로그 및 웨이트 저장 
			if epoch % display_step == 0:	
				line =("epoch %i, Avg Distance: %f\n" % (epoch, d))
				print(line)
				f.write(line)
					
				if d<=min_distance or epoch==0:
					min_distance=d
					min_index=epoch	
					
				self.saver.save(self.sess,cfg.weight_dir+"/w",epoch)	
		
		#저장한 weight restore
		self.saver.restore(self.sess,cfg.weight_dir+"/w-%d"%(min_index))
		self.idx = self.sess.run(self.cluster_idx,
						feed_dict=feed_dicts)

		#테스트 정확도 측정
		test_loss,test_acc=self.cal_loss_accuracy(1,len(y_test),cfg.batch_size,x_test,y_test,"test_loss","test_accuracy")
		line = "test_acc:%.9f, max_index:%d\n"%(test_acc,min_index)
		print(line)
		f.write(line)
		f.close()
		