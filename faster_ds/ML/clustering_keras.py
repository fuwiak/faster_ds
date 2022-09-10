from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout , Flatten
from keras.layers import BatchNormalization, Input, Lambda
from keras import regularizers
from keras.losses import mse, categorical_crossentropy
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import os


class Clustering:
    """
    Clustering class in Keras for unsupervised learning of dataframe

    """
    def __init__(self, df, n_clusters=2, n_epochs=100, batch_size=32, verbose=0):
        """
        :param df: dataframe
        :param n_clusters: number of clusters
        :param n_epochs: number of epochs
        :param batch_size: batch size
        :param verbose: verbose
        """
        self.df = df
        self.n_clusters = n_clusters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _encoder(self):
        """
        :return: encoder model

        """
        input_dim = self.df.shape[1]
        encoding_dim = 32
        input_layer = Input(shape=(input_dim, ))
        encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
        model = Model(inputs=input_layer, outputs=encoder)
        return model
    def _decoder(self):
        """
        :return: decoder model

        """
        input_dim = self.df.shape[1]
        encoding_dim = 32
        input_layer = Input(shape=(int(encoding_dim / 2), ))
        decoder = Dense(encoding_dim, activation='tanh')(input_layer)
        decoder = Dense(input_dim, activation='relu')(decoder)
        model = Model(inputs=input_layer, outputs=decoder)
        return model
    def _autoencoder(self):
        """
        :return: autoencoder model

        """
        input_dim = self.df.shape[1]
        autoencoder = Sequential()
        autoencoder.add(Dense(32, input_shape=(input_dim, ), activation='tanh', activity_regularizer=regularizers.l1(10e-5)))
        autoencoder.add(Dense(int(32 / 2), activation='relu'))
        autoencoder.add(Dense(32, activation='tanh'))
        autoencoder.add(Dense(input_dim, activation='relu'))
        return autoencoder
    def _clustering_layer(self, x, n_clusters, alpha=1.0):
        """
        :param x: input
        :param n_clusters: number of clusters
        :param alpha: alpha
        :return: clustering layer

        """
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        kmeans.fit(x)
        y_pred = kmeans.predict(x)
        y_pred_last = np.copy(y_pred)
        model = Model(inputs=x, outputs=y_pred)
        model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        model.compile(loss='kld', optimizer='adam')
        index = 0
        index_array = np.arange(x.shape[0])
        tol = 0.001
        while True:
            if index + self.batch_size <= x.shape[0]:
                loss = model.train_on_batch(x=x[index_array[index:index + self.batch_size]],
                                            y=None)
                index = index + self.batch_size
            else:
                loss = model.train_on_batch(x=x[index_array[index:]],
                                            y=None)
                index = 0
            y_pred = model.predict(x, verbose=0)
            if y_pred is None:
                y_pred = y_pred_last
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if index == 0:
                if delta_label < tol:
                    break
        return model
    def _clustering_model(self):
        """
        :return: clustering model

        """
        input_dim = self.df.shape[1]
        encoding_dim = 32
        input_layer = Input(shape=(input_dim, ))
        encoder = self._encoder()
        clustering_layer = self._clustering_layer(encoder(input_layer), self.n_clusters)
        model = Model(inputs=input_layer, outputs=clustering_layer(encoder(input_layer)))
        return model
    def _pretrain(self):
        """
        :return: pretrain model

        """
        autoencoder = self._autoencoder()
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(self.df, self.df, epochs=self.n_epochs, batch_size=self.batch_size, verbose=self.verbose)
        return autoencoder
    def fit(self):
        """
        :return: clustering model

        """
        pretrain = self._pretrain()
        clustering = self._clustering_model()
        clustering.get_layer(name='clustering').set_weights(pretrain.get_layer(name='dense_3').get_weights())
        clustering.compile(loss='kld', optimizer='adam')
        clustering.fit(self.df, self.df, epochs=self.n_epochs, batch_size=self.batch_size, verbose=self.verbose)
        return clustering
    def predict(self, model):
        """
        :param model: clustering model
        :return: predicted labels

        """
        return model.predict(self.df)
    def plot(self, model)->None:
        """
        :param model: clustering model
        :return: None

        """
        plt.scatter(self.df.iloc[:, 0], self.df.iloc[:, 1], c=model.predict(self.df))
        plt.show()

        return model.get_layer(name='clustering').get_weights()[0]

