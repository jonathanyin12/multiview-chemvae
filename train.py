from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import numpy as np
import h5py
import os


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--model_type', choices=['Grammar', 'Character'], default ='Character',
                        help='Which type of model to train')
    parser.add_argument('--two_tower', action='store_true', 
                        help='Whether kind of model to train.')
    parser.add_argument('--load_model', type=str, metavar='N', default="")
    parser.add_argument('--epochs', type=int, metavar='N', default=100,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=128,
                        help='Dimensionality of the latent representation.')
    return parser.parse_args()


def main():
    args = get_arguments()

    if args.model_type == 'Grammar':
        smiles_data_path = 'grammar_dataset.h5'
    elif args.model_type == 'Character':
        smiles_data_path = 'char_dataset.h5'
    # Load data
    print('Loading smiles data.')
    h5f = h5py.File(smiles_data_path, 'r')
    smiles_data = h5f['data'][:]
    h5f.close()

    # Save 5000 for testing
    smiles_data=smiles_data[5000:]

    if args.two_tower:
        print('Loading features data.')
        features_data_path = 'features_dataset.h5'
        h5f = h5py.File(features_data_path, 'r')
        rdkit_features = h5f['data'][:]
        h5f.close()
        
        # Save 5000 for testing
        rdkit_features=rdkit_features[5000:]

        # Delete molecules where RDKit features are NaN
        nan_indices = np.unique(np.argwhere(np.isnan(rdkit_features))[:,0])
        smiles_data = np.delete(smiles_data, nan_indices, axis=0) 
        rdkit_features = np.delete(rdkit_features, nan_indices, axis=0)

        smiles_data_train, smiles_data_test, rdkit_features_train, rdkit_features_test = train_test_split(smiles_data, rdkit_features, test_size=0.05, random_state=42)

        train_data = [smiles_data_train, rdkit_features_train]
        test_data = [smiles_data_test, rdkit_features_test]
    else:
        smiles_data_train, smiles_data_test, dummy_train, dummy_test = train_test_split(smiles_data, smiles_data, test_size=0.05, random_state=42)
        train_data = smiles_data_train
        test_data = smiles_data_test
   

    # Load model
    if args.model_type == 'Grammar':
        import zinc_grammar as G
        rules = G.gram.split('\n')
        if args.two_tower:
            from models.two_tower_grammar_vae import MoleculeVAE
            model_save = './weights/Two_tower_GrammarVAE_L{}.hdf5'.format(str(args.latent_dim))
        else: 
            from models.grammar_vae import MoleculeVAE       
            model_save = './weights/GrammarVAE_L{}.hdf5'.format(str(args.latent_dim))
    elif args.model_type == 'Character':
        rules = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[', '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/', '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
        if args.two_tower:
            from models.two_tower_char_vae import MoleculeVAE
            model_save = './weights/Two_tower_CharVAE_L{}.hdf5'.format(str(args.latent_dim))
        else: 
            from models.char_vae import MoleculeVAE
            model_save = './weights/CharVAE_L{}.hdf5'.format(str(args.latent_dim))
    

    print('Saving weights to', model_save)
    model = MoleculeVAE()
    if os.path.isfile(args.load_model):
        print('Loading...')
        model.load(rules, args.load_model, latent_rep_size = args.latent_dim)
        print('Done loading!')
    else:
        print('Making new model...')
        model.create(rules, latent_rep_size = args.latent_dim)
        print('New model created!')

    checkpointer = ModelCheckpoint(filepath = model_save, verbose = 2, save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, min_lr = 0.0001)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min')

    history=model.autoencoder.fit(
        train_data,
        train_data,
        batch_size=256,
        shuffle=True,
        epochs=args.epochs,
        verbose=1,
        callbacks = [checkpointer, reduce_lr, early_stop],
        validation_data=(test_data, test_data))


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
