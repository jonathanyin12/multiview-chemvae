import h5py
import numpy as np

grammar_data_path = 'grammar_dataset.h5'
features_data_path = 'features_dataset.h5'

save_path='./weights/'
load_model=save_path+ ""


#144338 158009
h5f = h5py.File(grammar_data_path, 'r')
grammar_data = h5f['data'][:]
h5f.close()
grammar_data=np.delete(grammar_data, [144338, 158009], axis=0) 
print(grammar_data.shape)

h5f = h5py.File(features_data_path, 'r')
rdkit_features = h5f['rdkit2d_normalized'][:]
h5f.close()
rdkit_features=np.expand_dims(rdkit_features, axis=2)
rdkit_features=np.delete(rdkit_features, [144338, 158009], axis=0)
print(rdkit_features.shape)

# save 5000 for testing
grammar_data=grammar_data[5000:]
rdkit_features=rdkit_features[5000:]


from sklearn.model_selection import train_test_split

grammar_data_train, grammar_data_test, rdkit_features_train, rdkit_features_test = train_test_split(grammar_data, rdkit_features, test_size=0.05, random_state=42)
print(rdkit_features_train[0][0]) #confirm split is replicable-- should print "array([0.74291674])"

# """## Load Model"""


import zinc_grammar as G
from models.two_tower_grammar_vae import MoleculeVAE
import os

rules = G.gram.split('\n')
DIM = len(rules)
LATENT = 128
EPOCHS = 200
BATCH = 256

model_save = save_path+'Two_tower___GrammarVAE_L{}.hdf5'.format(str(LATENT))

print(model_save)
model = MoleculeVAE()
if os.path.isfile(load_model):
    print('Loading...')
    model.load(rules, load_model, latent_rep_size = LATENT)
    print('Done loading!')
else:
    print('Making new model...')
    model.create(rules, latent_rep_size = LATENT)
    print('New model created!')

# from tensorflow.keras.utils import plot_model
# plot_model(model.autoencoder, to_file='two_tower_plot_shapes.png', show_shapes=True)



from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

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
    epochs=EPOCHS,
    initial_epoch=0,
    verbose=1,
    callbacks = [checkpointer, reduce_lr, early_stop],
    validation_data=([grammar_data_test, rdkit_features_test], [grammar_data_test, rdkit_features_test]))

# import matplotlib.pyplot as plt

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()