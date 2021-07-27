# Multi-view Chemical Variational Autoencoder

## Installation
A conda environment is necessary for installation. To create the enviroment, run the following command:
```
conda env create -f environment.yml
source activate multiview_chemvae
```
Jupyter notebook is also required to run the *.ipynb examples in the `experiments` directory.

## Creating datasets
To create the datasets for training, run
```
python make_char_dataset.py

python make_grammar_dataset.py

python make_features_dataset.py
```

The datasets will be named `char_dataset.h5`, `grammar_dataset.h5`, and `features_dataset.h5` respectively.
## Training
To train a model, run

```
python train.py  --model_type <type>
```

where `<type>` is either "Grammar" or "Character". 


Additional flags include:
* `--two_tower`: specifies whether to train a two-tower model or not
* `--load_model`: load weights from an existing model
* `--latent_dim`: specifies dimensionality of latent space
* `--epochs`: specifies number of training epochs

For example:

```
python train.py  --model_type=Grammar --two_tower --latent_dim=128 --epochs=100
```


## Experiments
The `experiments` directory contains the following Jupyter notebooks for evaluating the performance of the models:
* Prior validity: `experiments/prior_validity.ipynb`
* Reconstruction accuracy: `experiments/reconstruction_accuracy.ipynb`

* Property prediction: `experiments/property_prediction.ipynb`
