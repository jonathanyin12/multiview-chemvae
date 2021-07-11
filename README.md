# Multi-view Chemical Variational Autoencoder

## Installation


## Creating datasets
To create the datasets for training, run
* ```python make_char_dataset.py```
* ```python make_grammar_dataset.py```
* ```python make_features_dataset.py```

The datasets will be named `char_dataset.h5`, `grammar_dataset.h5`, and `features_dataset.h5` respectively.
## Training
To train a model, run

```python train.py  --model_type <type>```

where `<type>` is either "Grammar" or "Character". 


Additional flags include:
* `--two_tower`: specifies whether to train a two-tower model or not
* `--load_model`: load weights from an existing model
* `--latent_dim`: specifies dimensionality of latent space
* `--epochs`: specifies number of training epochs

For example:

```python train.py  --model_type=Grammar --two_tower --latent_dim=128 --epochs=100```


## Experiments

### Prior validity

### Reconstruction accuracy

### Property Prediction
