# Temporal Convolutional Neural Network
Training temporal Convolution Neural Netoworks (CNNs) on satelitte image time series.
This code is supporting one paper submitted in IEEE Transactions on Geoscience and Remote Sensing (under review):
(https://arxiv.org/abs/1811.101660): Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series


## Prerequisities
This code relies Pyhton 3.6 (ans it should work on Python 2.7 and Keras aith Tensorflow backend.


## Examples

### Running the models:

- main architecture: `python run_main_archi.py`
- other experiments described in the related paper: `python run_archi.py --sits_path ./ --res_path path/to/results --noarchi 0`

The architecture will run in `train_dataset.csv` and `test_dataset.csv` files, that are provided in the reporsitory.
Please note both `train_dataset.csv` and `test_dataset.csv` files are a subsample of the data used in the paper: Original data cannot be distributed.
Thoses files have no header, one observation per row described as `[class,date1.NIR,date1.R,date1.G,date2.NIR,...,date149.G]`

### Changing network parameters

- Number of channels in the data: `n_channels = 3` (`run_archi.py`, L21). It will also requires to change functions contained in `readingsits.py`..
- Validation rate: `val_rate = 0.05` (`run_archi.py`, L22).
- Network hyperparameters are mainly defined in `architecture_features.py` file.
