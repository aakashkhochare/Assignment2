import tensorflow as tf
import numpy as np
import sys
import random
import time

learning_rate = 0.05
num_of_classes = 50
#weights = tf.Variable(tf.random_uniform([72639,49],minval=0,maxval=0.1))
weights = tf.Variable(tf.zeros([467758,num_of_classes]))

toClassify = tf.placeholder(tf.int32, shape=(None))
correctChoice = tf.placeholder(tf.float32, shape=(num_of_classes,))

correctChoiceVal = tf.placeholder(tf.int32,shape=(1))

length = tf.placeholder(tf.int32,shape=(1))
requiredWeights = tf.transpose(tf.gather(weights,toClassify))
#print(requiredWeights.shape)
sums_inter = tf.reduce_sum(requiredWeights, 1)
sums = tf.exp(sums_inter)
grads_inter = tf.divide(sums,tf.reduce_sum(sums))
#grads       = tf.sub(correctChoice,grads_inter)
grads       = tf.subtract(correctChoice,grads_inter)
# works grads       = tf.subtract(grads_inter,correctChoice)
inter = tf.tile(tf.transpose(grads),[length[0]])
inter1 = tf.reshape(inter,[length[0],num_of_classes])
inter2 = inter1 * learning_rate
update_grads = tf.scatter_add(weights,toClassify,inter2)
# works update_grads = tf.scatter_sub(weights,toClassify,inter2)

sums_red = tf.reduce_sum(sums)
probs = tf.divide(sums,sums_red)
sums_red1 = tf.gather(sums,correctChoiceVal)
inter_loss = tf.divide(sums_red1,sums_red)
loss = tf.multiply(tf.log(inter_loss),-1)

#print(grads.shape)
saver = tf.train.Saver([weights])
init_op = tf.global_variables_initializer()
def main(_):
	arrays_test = []
	asso_class_test = []
	#load validation dataset
	with open("../data/full_processed_test.txt","r") as f:
		for line in f:
			parts = line.strip().split("\t")[1].split(",")
			s = set(parts)
			a = list(s)
			arrays = np.array(a,dtype=np.int32)
			arrays = arrays[arrays>=0]
			arrays_test.append(arrays)
			#print(arrays)
			parts = line.strip().split("\t")[0].split(",")
			asso_class_test.append(np.array(parts[0]))

	with tf.Session() as sess:
		sess.run(init_op)
		saver.restore(sess, "../data/models/model_full.ckpt")
		err = 0
		loss_itr = 0
		for step in range(0,len(arrays_test)-1):
		#for step in range(0,5000):				
			val_rec = sess.run([loss,probs],feed_dict={toClassify:arrays_test[step], correctChoiceVal: asso_class_test[step].reshape(1),length:arrays_test[step].shape})
			loss_itr += val_rec[0]
			numpyarray = val_rec[1]
			list1 = np.argmax(numpyarray).tolist()
			list2 = asso_class_test[step].tolist()
			if str(list1) == str(list2):
				err = err
			else:
				err += 1
		print("Final error " + str(float(err)/len(arrays_test)))
		#print("Before any SGD error " + str(float(err)/5000))
		print("Final loss " + str(loss_itr))

if __name__ == "__main__":
	tf.app.run(main=main)
