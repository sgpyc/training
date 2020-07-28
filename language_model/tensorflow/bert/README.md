# Location of the input files 

This [GCS location](https://console.cloud.google.com/storage/browser/pkanwar-bert) contains the following.
* TensorFlow checkpoint (bert_model.ckpt) containing the pre-trained weights (which is actually 3 files).
* Vocab file (vocab.txt) to map WordPiece to word id.
* Config file (bert_config.json) which specifies the hyperparameters of the model.

# Download and preprocess datasets

Download the [wikipedia dump](https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2) and extract the pages
The wikipedia dump can be downloaded from this link in this directory, and should contain the following file:
enwiki-20200101-pages-articles-multistream.xml.bz2

Run [WikiExtractor.py](https://github.com/attardi/wikiextractor) to extract the wiki pages from the XML
The generated wiki pages file will be stored as <data dir>/LL/wiki_nn; for example <data dir>/AA/wiki_00. Each file is ~1MB, and each sub directory has 100 files from wiki_00 to wiki_99, except the last sub directory. For the 20200101 dump, the last file is FE/wiki_17.

Clean up
The clean up scripts (some references here) are in the scripts directory.
The following command will run the clean up steps, and put the results in ./results
./process_wiki.sh '<data dir>/*/wiki_??'

After running the process_wiki.sh script, for the 20200101 wiki dump, there will be 500 files, named part-00xxx-of-00500 in the ./results directory.

Exact steps (starting in the bert path)  

```shell
cd cleanup_scripts  
mkdir -p wiki  
cd wiki  
wget https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2    # Optionally use curl instead  
bzip2 -d enwiki-20200101-pages-articles-multistream.xml.bz2  
cd ..    # back to bert/cleanup_scripts  
git clone https://github.com/attardi/wikiextractor.git  
python3 wikiextractor/WikiExtractor.py wiki/enwiki-20200101-pages-articles-multistream.xml    # Results are placed in bert/cleanup_scripts/text  
./process_wiki.sh '<text/*/wiki_??'  
python3 extract_test_set_articles.py  
```
  
MD5sums:  
7f59165e21b7d566db610ff6756c926b - bert_config.json  
00d47075e0f583fb7c0791fac1c57cb3 - enwiki-20200101-pages-articles-multistream.xml.bz2   
50797acd537880bfb5a7ade80d976129  model.ckpt-28252.data-00000-of-00001
f97de3ae180eb8d479555c939d50d048  model.ckpt-28252.index
dbd16c731e8a8113bc08eeed0326b8e7  model.ckpt-28252.meta
64800d5d8528ce344256daf115d4965e - vocab.txt  

# Generate the BERT input dataset

The create_pretraining_data.py script duplicates the input plain text, replaces different sets of words with masks for each duplication, and serializes the output into the TFRecord file format. 

```shell
python3 create_pretraining_data.py \
   --input_file=<path to ./results of previous step>/part-XX-of-00500 \
   --output_file=<tfrecord dir>/part-XX-of-00500 \
   --vocab_file=<path to vocab.txt> \
   --do_lower_case=True \
   --max_seq_length=512 \
   --max_predictions_per_seq=76 \
   --masked_lm_prob=0.15 \
   --random_seed=12345 \
   --dupe_factor=10
```

The generated tfrecord has 500 parts, totalling to ~365GB.
The dataset was generated using Python 3.7.6 and tensorflow-gpu 1.15.2.

# Stopping criteria
The training should occur over a minimum of 3,000,000 samples. A valid submission will evaluate a masked lm accuracy >= 0.712. 

The evaluation will be on the first 10,000 consecutive samples of the training set. The evalution frequency is every 500,000 samples, starting from 3,000,000 samples. The evaluation can be either offline or online for v0.7. More details please refer to the training policy.

The generation of the evaluation set shard should follow the exact command shown above, using create_pretraining_data.py. In particular the seed (12345) must be set to ensure everyone evaluates on the same data.

# Running the model

## on NVIDIA Tesla V100-16GB GPUs
To run this model on GPUs, use the following command.

```shell

TF_XLA_FLAGS='--tf_xla_auto_jit=2' \
python run_pretraining.py \
  --bert_config_file=./bert_config.json \
  --output_dir=/tmp/output/ \
  --input_file="./uncased_seq_512/wikipedia.tfrecord*" \
  --nodo_eval \
  --do_train \
  --eval_batch_size=8 \
  --learning_rate=4e-05 \
  --init_checkpoint=./checkpoint/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=682666666 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=20833 \
  --start_warmup_step=0 \
  --num_gpus=8 \
  --train_batch_size=24

```

The above parameters are for a machine with 8 V100 GPUs with 16GB memory each; the hyper parameters (learning rate, warm up steps, etc.) are for testing only. The training script won’t print out the masked_lm_accuracy; in order to get masked_lm_accuracy, a separately invocation of run_pretraining.py with the following command with a V100 GPU with 16 GB memory:

```shell

TF_XLA_FLAGS='--tf_xla_auto_jit=2' \
python3 run_pretraining.py \
  --bert_config_file=./bert_config.json \
  --output_dir=/tmp/output/ \
  --input_file="<tfrecord dir>/part-*" \
  --do_eval \
  --nodo_train \
  --eval_batch_size=8 \
  --init_checkpoint=./checkpoint/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --learning_rate=4e-05 \
  --max_eval_steps=1250 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_gpus=1 \
  --num_train_steps=682666666 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=20833 \
  --start_warmup_step=0 \
  --train_batch_size=24 \
  --nouse_tpu
   
```

The model has been tested using the following stack:
- Debian GNU/Linux 9.12 GNU/Linux 4.9.0-12-amd64 x86_64
- NVIDIA Driver 440.64.00
- NVIDIA Docker 2.2.2-1 + Docker 19.03.8
- docker image tensorflow/tensorflow:2.2.0rc0-gpu-py3

## On Google TPU-v3

To run the model on TPUs by Google cloud, follow these steps:

1. create a GCP host instance

```shell

gcloud compute instances create <host name> \
  --boot-disk-auto-delete \
  --boot-disk-size 500 \
  --boot-disk-type pd-ssd \
  --format json \
  --image debian-10-tf-2-2-v20200508 \
  --image-project ml-images \
  --machine-type n1-highmem-96 \
  --min-cpu-platform skylake \
  --network default \
  --network-tier PREMIUM \
  --no-restart-on-failure \
  --project <GCP project> \
  --quiet \
  --scopes cloud-platform \
  --tags perfkitbenchmarker \
  --zone <GCP zone that the project has TPU-v3 quota in>

```

2. create train TPU instances

```shell

gcloud compute tpus create <host name>_train \
  --accelerator-type v3-32 \
  --format json \
  --network default \
  --project <the same project as the host> \
  --quiet \
  --range <IP range, e.g. 10.196.80.0/29> \
  --version 2.2 \
  --zone <the same zone as the host>

```

3. create eval TPU instances

```shell

gcloud compute tpus create <host name>_eval \
  --accelerator-type v3-8 \
  --format json \
  --network default \
  --project <the same project as the host> \
  --quiet \
  --range <IP range, e.g. 10.192.60.0/29> \
  --version 2.2 \
  --zone <the same zone as the host>

```

4. Download the model code on the host, and start training by

```shell

python3 run_pretraining.py \
  --bert_config_file=gs://pkanwar-bert/bs64k_32k_ckpt/bert_config.json \
  --nodo_eval \
  --do_train \
  --eval_batch_size=448 \
  --init_checkpoint=gs://pkanwar-bert/bs64k_32k_ckpt/model.ckpt-28252 \
  --input_file=<GCP path that stores the input file>/part-* \
  --iterations_per_loop=1157 \
  --lamb_beta_1=0.9 \
  --lamb_beta_2=0.9999 \
  --lamb_weight_decay_rate=0.01 \
  --learning_rate=0.0004 \
  --log_epsilon=-6 \
  --tpu_name=<host name>_train \
  --tpu_zone=<the same zone as the host> \
  --max_eval_steps=23 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_tpu_cores=32 \
  --num_train_steps=8103 \
  --num_warmup_steps=0 \
  --optimizer=lamb \
  --output_dir=<a GCP path to store the output checkpoints> \
  --save_checkpoints_steps=1157 \
  --start_warmup_step=0 \
  --train_batch_size=448 \
  --use_tpu

```

By default, the training code only keeps the most recent 5 checkpoints.

5. Evaluate the output checkpoints by

```shell

python3 run_pretraining.py \
  --bert_config_file=gs://pkanwar-bert/bs64k_32k_ckpt/bert_config.json \
  --do_eval \
  --nodo_train \
  --eval_batch_size=16 \
  --init_checkpoint=gs://pkanwar-bert/bs64k_32k_ckpt/model.ckpt-28252 \
  --input_file=<GCP path for the input dataset>/part-* \
  --iterations_per_loop=1157 \
  --lamb_beta_1=0.9 \
  --lamb_beta_2=0.9999 \
  --lamb_weight_decay_rate=0.01 \
  --learning_rate=0.0004 \
  --log_epsilon=-6 \
  --tpu_name=<host name>_eval \
  --tpu_zone=<the same zone as the host> \
  --max_eval_steps=5000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_tpu_cores=8 \
  --num_train_steps=8103 \
  --num_warmup_steps=0 \
  --optimizer=lamb \
  --output_dir=<GCP path that stores the output checkpoints> \
  --save_checkpoints_steps=1157 \
  --start_warmup_step=0 \
  --train_batch_size=448 \
  --use_tpu

```

By default, the evaluation uses the most recent checkpoint; to eval on
a previous checkpoint, change the first line of the "checkpoint" file
in the output directory to point to the desired checkpoint.

The model has been tested on TPU with the following software stack:
- Debian GNU/Linux 4.19.98-1 x86_64
- GCP image debian-10-tf-2-2-v20200508
- TF 2.2 on TPU 
- TF 2.2.0 on host
