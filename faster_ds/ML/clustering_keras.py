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
 

# X.shape(150, 3)
# y.shape(150,1)

  
  
batch_size = 8
num_epochs = 50
num_classes = 3

model = Sequential()
model.add(Dense(2, activation='relu',input_shape=X.shape[1:])) 
model.add(Flatten())
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(y.shape[1], activation = "softmax"))
opt = keras.optimizers.Adam(learning_rate=1e-50)

model.compile(loss="categorical_crossentropy",metrics=['accuracy'],optimizer=opt)

model.summary() 

model.fit(x=X_train, y=y_train,epochs=num_epochs, batch_size=batch_size, shuffle=True,validation_data=(X_test,y_test),verbose=1)
