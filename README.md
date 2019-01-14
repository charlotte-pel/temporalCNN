# Temporal Convolutional Neural Network
Training temporal Convolution Neural Netoworks (CNNs) on satelitte image time series.  
This code is supporting a paper submitted to IEEE Transactions on Geoscience and Remote Sensing (under review):  
https://arxiv.org/abs/1811.10166 - Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series

More information about our research at https://sites.google.com/site/charpelletier, http://www.francois-petitjean.com/Research/, and http://i.giwebb.com/

## Prerequisites
This code relies on Pyhton 3.6 (and should work on Python 2.7) and Keras with Tensorflow backend.


## Examples

### Running the models

- main architecture: `python run_main_archi.py`
- other experiments described in the related paper: 
```
python run_archi.py --sits_path ./ --res_path path/to/results --noarchi 0
```

The architecture will run by training the network on `train_dataset.csv` file and by testing it on `test_dataset.csv` file.  
Please note that both `train_dataset.csv` and `test_dataset.csv` files are a subsample of the data used in the paper: original data cannot be distributed.

Thoses files have no header, and contain one observation per row having the following format:
`[class,date1.NIR,date1.R,date1.G,date2.NIR,...,date149.G]`

### Changing network parameters

- Number of channels in the data: `n_channels = 3` (`run_archi.py`, L21).  
It will require to change functions contained in `readingsits.py`.
- Validation rate: `val_rate = 0.05` (`run_archi.py`, L22).
- Network hyperparameters are mainly defined in `architecture_features.py` file.


