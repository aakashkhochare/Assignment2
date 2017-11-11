# Assignment2
DATASET prep:
Please maintain the datasets in ../data
Run the final_pre-processing_train.py first and then final_pre-processing_test_and_validate.py later. This performs standard
preprocessing such as lemmatization and removal of stop words.
For distributed, split the dataset into 4 equal parts using the command given below
split -l 53749  full_processed_train.txt full_processed_train.split.txt -da 4

Running code:
local methods are just python files that can be executed from command line
The distributed code should be run with one parameter server and 4 worker nodes. Here's an example that runs aynchronous train:

To run on node10
python asynchronoustrain.py \
     --ps_hosts=10.24.1.210:2222 \
     --worker_hosts=10.24.1.204:2222,10.24.1.205:2222,10.24.1.206:2222,10.24.1.209:2222 \
     --job_name=ps --task_index=0


To run on node4
python asynchronoustrain.py \
     --ps_hosts=10.24.1.210:2222 \
     --worker_hosts=10.24.1.204:2222,10.24.1.205:2222,10.24.1.206:2222,10.24.1.209:2222 \
     --job_name=worker --task_index=0


To run on node5
python asynchronoustrain.py \
     --ps_hosts=10.24.1.210:2222 \
     --worker_hosts=10.24.1.204:2222,10.24.1.205:2222,10.24.1.206:2222,10.24.1.209:2222 \
     --job_name=worker --task_index=1

To run on node6
python asynchronoustrain.py \
     --ps_hosts=10.24.1.210:2222 \
     --worker_hosts=10.24.1.204:2222,10.24.1.205:2222,10.24.1.206:2222,10.24.1.209:2222 \
     --job_name=worker --task_index=2

To run on node9
python asynchronoustrain.py \
     --ps_hosts=10.24.1.210:2222 \
     --worker_hosts=10.24.1.204:2222,10.24.1.205:2222,10.24.1.206:2222,10.24.1.209:2222 \
     --job_name=worker --task_index=3
