import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import zinc_grammar as G
from models.two_tower_grammar_vae import MoleculeVAE
import os
import matplotlib.pyplot as plt
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--load_model', type=str, metavar='N', default="")
    parser.add_argument('--epochs', type=int, metavar='N', default=100,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=128,
                        help='Dimensionality of the latent representation.')
    return parser.parse_args()


def main():
    args = get_arguments()

    # Load data
    grammar_data_path = 'grammar_dataset.h5'
    h5f = h5py.File(grammar_data_path, 'r')
    grammar_data = h5f['data'][:]
    h5f.close()

    features_data_path = 'features_dataset.h5'
    h5f = h5py.File(features_data_path, 'r')
    rdkit_features = h5f['rdkit2d_normalized'][:]
    h5f.close()

    rdkit_features = np.expand_dims(rdkit_features, axis=2)

    # Delete molecules where RDKit features are NaN
    nan_indices = np.unique(np.argwhere(np.isnan(rdkit_features))[:,0])
    grammar_data = np.delete(grammar_data, nan_indices, axis=0) 
    rdkit_features = np.delete(rdkit_features, nan_indices, axis=0)
    print(grammar_data.shape, rdkit_features.shape)


    # save 5000 for testing
    grammar_data=grammar_data[5000:]
    rdkit_features=rdkit_features[5000:]
    grammar_data_train, grammar_data_test, rdkit_features_train, rdkit_features_test = train_test_split(grammar_data, rdkit_features, test_size=0.05, random_state=42)
    # print(rdkit_features_train[0][0]) #confirm split is replicable-- should print "array([0.74291674])"


    rules = G.gram.split('\n')
    BATCH = 256

    model_save = './weights/Two_tower___GrammarVAE_L{}.hdf5'.format(str(args.latent_dim))

    print(model_save)
    model = MoleculeVAE()

    if os.path.isfile(args.load_model):
        print('Loading...')
        model.load(rules, args.load_model, latent_rep_size = args.latent_dim)
        print('Done loading!')
    else:
        print('Making new model...')
        model.create(rules, latent_rep_size = args.latent_dim)
        print('New model created!')


    checkpointer = ModelCheckpoint(filepath = model_save,
                                    verbose = 2,
                                    save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                    factor = 0.5,
                                    patience = 3,
                                    min_lr = 0.0001)

    early_stop = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            mode='min')

    history=model.autoencoder.fit(
        [grammar_data_train, rdkit_features_train],
        [grammar_data_train, rdkit_features_train],
        batch_size=BATCH,
        shuffle=True,
        epochs=args.epochs,
        initial_epoch=0,
        verbose=1,
        callbacks = [checkpointer, reduce_lr, early_stop],
        validation_data=([grammar_data_test, rdkit_features_test], [grammar_data_test, rdkit_features_test]))


    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()