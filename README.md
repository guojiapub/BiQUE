BiQUE: Biquaternionic Embeddings of Knowledge Graphs
===
This is the official implementation for "BiQUE: Biquaternionic Embeddings of Knowledge Graphs" (EMNLP 2021, Main Conference).

## Dependencies
- Python 3.6+
- PyTorch 1.0+
- NumPy 1.17.2+
- tqdm 4.41.1+

The folder structure is:

```
|-- .BiQUE
    |-- README.md
    |-- src_data
    |-- data
    |-- codes
    |-- ckpt    
```


### 1. Preprocess the Datasets
To preprocess the datasets, run the following commands.

```shell script
cd codes
python process_datasets.py
```

Now, the processed datasets are in the `data` directory.


### 2. Reproduce the Results 
To reproduce the reported results of BiQUE on WN18RR, FB15k237, YAGO3-10, CN-100K and ATOMIC, please run the following commands.

```shell script
cd codes
python reproduce.py dataset

dataset = ["WN18RR", "FB237", "YAGO3", "CN100K", "ATOMIC"]
```

### 3. Training BiQUE model

```shell script

# WN18RR
python learn.py --dataset WN18RR --model BiQUE --rank 128 --optimizer Adagrad --learning_rate 1e-1 --batch_size 300 --regularizer wN3 --reg 1.5e-1 --max_epochs 200 --valid 5 -train -id 0 -save -weight


# FB15K-237
python learn.py --dataset FB237 --model BiQUE --rank 128 --optimizer Adagrad --learning_rate 1e-1 --batch_size 500 --regularizer wN3 --reg 7e-2 --max_epochs 300 --valid 5 -train -id 0 -save


# YAGO3-10
python learn.py --dataset YAGO3-10 --model BiQUE --rank 128 --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer wN3 --reg 5e-3 --max_epochs 200 --valid 5 -train -id 0 -save


# CN-100k
python learn.py --dataset Concept100k --model BiQUE --rank 128 --optimizer Adagrad --learning_rate 1e-1 --batch_size 5000 --regularizer wN3 --reg 1e-1 --max_epochs 200 --valid 5 -train -id 0 -save


# ATOMIC
python learn.py --dataset Atomic --model BiQUE --rank 128 --optimizer Adagrad --learning_rate 1e-1 --batch_size 5000 --regularizer wN3 --reg 5e-3 --max_epochs 200 --valid 5 -train -id 0 -save
```

## Acknowledgement
We refer to the codes of [kbc](https://github.com/facebookresearch/kbc). Thanks for their contributions.
