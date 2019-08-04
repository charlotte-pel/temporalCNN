# Temporal Convolutional Neural Network
Training temporal Convolution Neural Netoworks (CNNs) on satelitte image time series.  
This code is supporting by a paper published in Remote Sensing:
```
@article{Pelletier2019Temporal,
  title={Temporal convolutional neural network for the classification of satellite image time series},
  author={Pelletier, Charlotte and Webb, Geoffrey I and Petitjean, Fran{\c{c}}ois},
  journal={Remote Sensing},
  volume={11},
  number={5},
  pages={523},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute},
  note={https://www.mdpi.com/2072-4292/11/5/523}
}

```

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
`[class,polygonID,date1.NIR,date1.R,date1.G,date2.NIR,...,date149.G]`,
where `class` corresponds to the class label and `polygonID` to a unique polygon identifier for each plot of land.

### Changing network parameters

- Number of channels in the data: `n_channels = 3` (`run_archi.py`, L21).  
It will require to change functions contained in `readingsits.py`.
- Validation rate: `val_rate = 0.05` (`run_archi.py`, L22).
- Network hyperparameters are mainly defined in `architecture_features.py` file.


### Maps

The produced map for TempCNNs and RFs are available in the `map` folder.


## Contributors
 - [Dr. Charlotte Pelletier](https://sites.google.com/site/charpelletier)
 - [Professor Geoffrey I. Webb](http://i.giwebb.com/)
 - [Dr. Francois Petitjean](http://www.francois-petitjean.com/Research/)



