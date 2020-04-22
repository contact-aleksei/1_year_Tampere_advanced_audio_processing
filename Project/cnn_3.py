import librosa, os, re
import numpy as np
import soundfile as sf
import pandas as pd


###############################################################################
##### here below, we have X as data of MFCC and corresponding labels as y #####
###############################################################################


def get_files_from_dir_with_os(dir_name):
    return os.listdir(dir_name)

dir_name='ESC-50-master/audio/'
files=get_files_from_dir_with_os(dir_name)
train_features, train_class_labels, test_features, test_class_labels=[],[],[],[]
validation_features, validation_class_labels=[],[]

i=0

for file_name in files:
    X, sample_rate = sf.read(dir_name+file_name, dtype='float32')     
    melspectrogram = librosa.feature.melspectrogram(X,n_mels=40, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0)
    #melgram = librosa.core.amplitude_to_db(melspectrogram, ref=1.0, amin=1e-05, top_db=80.0)[np.newaxis,np.newaxis,:,:]
    #melspectrogram = melspectrogram.T
    i=i+1
    print("%.2f" %(i*100/2000), ' percent of melspectrogram features are read from audio files')
    class_label=str(file_name[-6:])    
    temp = re.findall(r'\d+', class_label) 
    class_label = list(map(int, temp))
    
    if file_name[0]=="1":
        train_features.append([melspectrogram, class_label])
    if file_name[0]=="2":
        train_features.append([melspectrogram, class_label])
    if file_name[0]=="3":
        train_features.append([melspectrogram, class_label])
    if file_name[0]=="4":
        validation_features.append([melspectrogram, class_label])
    if file_name[0]=="5":
        test_features.append([melspectrogram, class_label])
#        
        
test_features_df = pd.DataFrame(test_features, columns=['feature','class_label'])
X_test = np.array(test_features_df.feature.tolist())
y_test = np.array(test_features_df.class_label.tolist())


validation_features_df = pd.DataFrame(validation_features, columns=['feature','class_label'])
X_validation = np.array(validation_features_df.feature.tolist())
y_validation = np.array(validation_features_df.class_label.tolist())

train_features_df = pd.DataFrame(train_features, columns=['feature','class_label'])
X_train = np.array(train_features_df.feature.tolist())
y_train = np.array(train_features_df.class_label.tolist())

###############################################################################
######################### here below,  we defined CNN #########################
###############################################################################

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.models import Sequential
import pandas as pd
import numpy as np

model = Sequential()
#model.add(BatchNormalization()))
model.add(Conv2D(filters=32, kernel_size=4, padding='same',input_shape=(40,431,1)))
#model.add(Conv2D(filters=32, kernel_size=4, padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Activation('relu'))
model.add(Dropout(0.10))
#
model.add(Conv2D(filters=64, kernel_size=4, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Activation('relu'))
model.add(Dropout(0.10))
#
model.add(Conv2D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Activation('relu'))   
model.add(Dropout(0.10))
#
model.add(Conv2D(filters=256, kernel_size=2, activation='relu'))
#model.add(MaxPooling2D(pool_size=2))
model.add(Activation('relu'))
model.add(Dropout(0.10))

model.add(GlobalAveragePooling2D())
model.add(Dense(50, activation='softmax'))

model.summary()


###############################################################################
#################### here below,  we train CNN with K-fold ####################
###############################################################################
    

import keras
y_train = keras.utils.to_categorical(y_train,50)
y_validation = keras.utils.to_categorical(y_validation,50)
y_test = keras.utils.to_categorical(y_test,50)

X_train = np.expand_dims(X_train, axis=2)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],X_train.shape[3],1))

X_validation= np.expand_dims(X_validation, axis=2)
X_validation = np.reshape(X_validation,(X_validation.shape[0],X_validation.shape[1],X_validation.shape[3],1))

X_test = np.expand_dims(X_test, axis=2)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],X_test.shape[3],1))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train,y_train,batch_size= 8, epochs = 150, validation_data=(X_validation,y_validation))

results=model.evaluate(x=X_test, y=y_test, batch_size=8, verbose=1)
print('test loss, test acc:', results)