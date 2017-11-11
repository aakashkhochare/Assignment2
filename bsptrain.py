import tensorflow as tf
import numpy as np
import sys
import random 
import argparse
import time

learning_rate = 0.05
FLAGS = None
#num_of_classes = 49
num_of_classes = 50
def main(_):
	ps_hosts = FLAGS.ps_hosts.split(",")
	worker_hosts = FLAGS.worker_hosts.split(",")
	
	# Create a cluster from the parameter server and worker hosts.
	cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
	# Create and start a server for the local task.
	server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

	if FLAGS.job_name == "ps":
		server.join()
	elif FLAGS.job_name == "worker":
		with tf.device(tf.train.replica_device_setter(worker_device="/job:ps/task:0",cluster=cluster)):
			#weights = tf.Variable(tf.random_uniform([72639,num_of_classes],minval=0,maxval=0.1))
			weights = tf.Variable(tf.zeros([467758,num_of_classes]))
			validating = tf.Variable(tf.zeros([1]))
			iteration  = tf.Variable(tf.zeros([4]))

		with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
			toClassify = tf.placeholder(tf.int32, shape=(None))
			correctChoice = tf.placeholder(tf.float32, shape=(num_of_classes,))	
			correctChoiceVal = tf.placeholder(tf.int32,shape=(1))

			length = tf.placeholder(tf.int32,shape=(1))
			requiredWeights = tf.transpose(tf.gather(weights,toClassify))
			sums_inter = tf.reduce_sum(requiredWeights, 1)
			sums = tf.exp(sums_inter)
			grads_inter = tf.divide(sums,tf.reduce_sum(sums))
			grads       = tf.subtract(correctChoice,grads_inter)
			inter = tf.tile(tf.transpose(grads),[length[0]])
			inter1 = tf.reshape(inter,[length[0],num_of_classes])
			inter2 = inter1 * learning_rate
			update_grads = tf.scatter_add(weights,toClassify,inter2,use_locking=False)

			sums_red = tf.reduce_sum(sums)
			probs = tf.divide(sums,sums_red)
			sums_red1 = tf.gather(sums,correctChoiceVal)
			inter_loss = tf.divide(sums_red1,sums_red)
			loss = tf.multiply(tf.log(inter_loss),-1)
			
			#test if the validation variable is set to 1
			testValidating = validating
			getIterations = iteration
			updateIterations = tf.scatter_add(iteration,[FLAGS.task_index],[1],use_locking=False)


		if FLAGS.task_index == 0:
			with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:0",cluster=cluster)):
				init_op = tf.global_variables_initializer()
				saver = tf.train.Saver([weights])
	
				testValidating = validating
				#set the validation variable to 1
				setValidating = tf.assign(validating,[1])
				#set the validation variable to 0
				unsetValidating = tf.assign(validating,[0])

		arrays_train = []
		asso_class_train = []
		#load training dataset based on FLAG.index
		if FLAGS.task_index == 0:
			fileToOpen = "../data/full_processed_train.split.txt0000"
		elif FLAGS.task_index == 1:
			fileToOpen = "../data/full_processed_train.split.txt0001"
		elif FLAGS.task_index == 2:
			fileToOpen = "../data/full_processed_train.split.txt0002"
		elif FLAGS.task_index == 3:
			fileToOpen = "../data/full_processed_train.split.txt0003"

		with open(fileToOpen,"r") as f:
			for line in f:
				parts = line.strip().split("\t")[1].split(",")
				s = set(parts)
				a = list(s)
				arrays = np.array(a,dtype=np.int32)
				arrays = arrays[arrays>=0]
				arrays_train.append(arrays)
				#print(arrays)
				parts = line.strip().split("\t")[0].split(",")
				asso_class_train.append(np.array(parts[0],dtype=np.int32))
	
		#load validation dataset
		if FLAGS.task_index == 0:
			arrays_validate = []
			asso_class_validate = []
			with open("../data/full_processed_devel.txt","r") as f:
				for line in f:
					parts = line.strip().split("\t")[1].split(",")
					s = set(parts)
					a = list(s)
					arrays = np.array(a,dtype=np.int32)
					arrays = arrays[arrays>=0]
					arrays_validate.append(arrays)
					#print(arrays)
					parts = line.strip().split("\t")[0].split(",")
					asso_class_validate.append(np.array(parts[0]))
		myIter = 0.0
		config = tf.ConfigProto(device_filters=["/job:ps"])
		with tf.Session(server.target,config=config) as sess:
			if FLAGS.task_index == 0:
				sess.run(init_op)
			while True:
				toShuffle = list(zip(arrays_train, asso_class_train))
				random.shuffle(toShuffle)
				arrays_train, asso_class_train = zip(*toShuffle)
				epoch_start_time = time.time()
				#for step in range(0,len(arrays_train)-1):
				for step in range(0,100):
					correct = np.zeros(num_of_classes)
					correct[asso_class_train[step]] = 1
					val = sess.run([getIterations])
					start_wait = time.time()
					#print(np.amin(val))
					#print(np.amin(val) - myIter)
					while np.amin(val) - myIter < 0.0:
						val = sess.run([getIterations])
						#print(np.amin(val))
					myIter += 1.0
					end_wait  = time.time()
					print("Time spent waiting is " + str(end_wait - start_wait))
					start_time = time.time()
					print("Start time for iteration " + str(myIter) + " is " + str(start_time))
					sess.run([update_grads],feed_dict={toClassify:arrays_train[step], correctChoice: correct,length:arrays_train[step].shape})
					end_time   = time.time()
					sess.run([updateIterations])
					print("Time for processing each tuple is " + str(end_time-start_time))
					print("Completed Iteration is "+str(myIter))		
				epoch_end_time = time.time()
				#only the master does the validation
				if FLAGS.task_index == 0:
					save_path = saver.save(sess, "../data/models/model_bsp_parallel_4.ckpt")
					print("Model Saved")
					#sess.run([setValidating])
					err = 0
					loss_itr = 0
					for step in range(0,len(arrays_validate)-1):				
						val_rec = sess.run([loss,probs],feed_dict={toClassify:arrays_validate[step], correctChoiceVal: asso_class_validate[step].reshape(1),length:arrays_validate[step].shape})
						loss_itr += val_rec[0]
						numpyarray = val_rec[1]
						list1 = np.argmax(numpyarray).tolist()
						list2 = asso_class_validate[step].tolist()
						if str(list1) == str(list2):
							err = err
						else:
							err += 1
					print(float(err)/len(arrays_validate))
					print(loss_itr)
					#sess.run([unsetValidating])


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
