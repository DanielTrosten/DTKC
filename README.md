# DTKC
Impplementation of the DTKC-model from ["Deep Image Clustering with Tensor Kernels and Unsupervised Companion Objectives"](https://arxiv.org/abs/2001.07026)

## Dependencies
```
h5py=2.9.0
hdf5=1.10.5
numpy=1.17.2
python=3.6.7
tensorflow-gpu=2.0.0
tensorflow-probability=0.7
```

## Training
To train the model, run `main.py` with the desired parameters.

```
optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET_NAME
                        Name of the dataset. Can be either 'mnist' or 'fmnist'
  --n_clusters N_CLUSTERS
                        Number of clusters
  --n_runs N_RUNS       Number of runs
  --n_epochs N_EPOCHS   Number of epochs
  --batch_size BATCH_SIZE
                        Batch size
  --sigma SIGMA         Scaling factor for the sigma hyperparameter
  --lam LAM             Lambda hyperparameter
  --n_hidden N_HIDDEN   Number of units in the hidden layer
  --hidden_activation HIDDEN_ACTIVATION
                        Activation function for the hidden layer
  --batch_norm BATCH_NORM
                        Use batch normalization after the hidden layer
  --clipnorm CLIPNORM   Gradient norm for gradient clipping
  --learning_rate LEARNING_RATE
                        Learning rate for the Adam optimizer
  --use_companion_losses USE_COMPANION_LOSSES
                        Enable companion objectives?
```

## Custom datasets
You can train the model on you own dataset by defining a dataset name and a loading function in the `LOADERS` dictionary, in `data.py`.
