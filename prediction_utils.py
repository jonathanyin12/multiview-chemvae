from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping


early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    mode='min'
)

def create_model(hidden_size=1024, classification=True, input_size=56, output_size=1, activation='sigmoid', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()]):
    model = Sequential()
    model.add(Input(shape=(input_size)))
    model.add(Dense(units=hidden_size, activation='relu', ))    
    model.add(Dropout(0.5))        
    model.add(Dense(units=output_size, activation=activation))

    model.compile(loss=loss, optimizer='adam', metrics=metrics)
    return model


def get_multilabel_ROC_score(X,y,latent_rep_size, output_size, folds=10):
    kfold = KFold(n_splits=folds, shuffle=True, random_state=0)
    aucs = []
    for train, test in tqdm(kfold.split(X, y), total=folds):
        model=create_model(input_size=latent_rep_size, output_size=output_size)
        model.fit(X[train], y[train], epochs=250, batch_size=256, validation_split=0.1, verbose=0, callbacks = [early_stop])
        preds=model.predict(X[test], batch_size=256)
       
        try:
            score=roc_auc_score(y[test], preds)
            aucs.append(score)
        except:
            continue
    print("Mean AUC-ROC: {} =/- {}".format(np.average(aucs), np.std(aucs)))


def rmse(y_pred, y_true):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def get_regression_loss(X,y,latent_rep_size, loss_metric, folds=10):
    kfold = KFold(n_splits=folds, shuffle=True, random_state=0)
    losses = []
    for train, test in tqdm(kfold.split(X, y), total=folds):
        if loss_metric=='rmse':
            model=create_model(input_size=latent_rep_size, activation="linear", loss=rmse)
        else:
            model=create_model(input_size=latent_rep_size, activation="linear", loss=loss_metric)
        model.fit(X[train], y[train], epochs=250, batch_size=256, validation_split=0.1, verbose=0, callbacks = [early_stop])
        loss=model.evaluate(X[test], y[test], verbose=0)
        losses.append(loss)

    print("Mean {}: {} +/- {}".format(loss_metric, np.average(losses), np.std(losses)))    


def get_ROC_score(X, y, latent_rep_size, folds=10):
    base_fpr, mean_tprs, mean_auc, std_auc, tprs_lower, tprs_upper = get_average_ROC_curve(X, y, latent_rep_size, folds=10)
    print("Mean AUC-ROC: {} +/- {}".format(mean_auc, std_auc))


def get_average_ROC_curve(X, y, latent_rep_size, folds=10): 
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)

    interp_tprs = []
    tprs = []
    fprs = []
    aucs = []
    base_fpr = np.linspace(0, 1, 101)    
    for train, test in tqdm(kfold.split(X, y), total=folds):
        model=create_model(input_size=latent_rep_size)

        model.fit(X[train], y[train], epochs=250, batch_size=256, validation_split=0.1,  verbose=0, callbacks = [early_stop])

        preds=model.predict(X[test], batch_size=256)
        fpr, tpr, _ = roc_curve(y[test], preds)
        tprs.append(tpr)
        fprs.append(fpr)

        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        interp_tprs.append(tpr)

    interp_tprs = np.array(interp_tprs)
    mean_tprs = interp_tprs.mean(axis=0)
    std = interp_tprs.std(axis=0)

    mean_auc = auc(base_fpr, mean_tprs)
    std_auc = np.std(aucs)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    return base_fpr, mean_tprs, mean_auc, std_auc, tprs_lower, tprs_upper
 

def plot_average_ROC_curves(roc_curves, dataset_name, model_names, folds=10,):

    plt.figure(figsize=(8, 6))
    colors = ['darkblue', 'blue', 'darkorange', 'orange', 'darkgreen', 'green']

    for i in range(len(roc_curves)):
        base_fpr, mean_tprs, mean_auc, std_auc, tprs_lower, tprs_upper = roc_curves[i]
        color=colors[i]
        model_name=model_names[i]  
        plt.plot(base_fpr, mean_tprs, color, alpha = 0.9, label=model_name+ r' Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = color, alpha = 0.1)
   
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.title('{} ROC curve using stratified {}-fold cross-validation '.format(dataset_name, folds))
    plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'black', label = 'Random chance', alpha= 0.8)
    plt.show()