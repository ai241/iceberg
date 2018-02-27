
# start my homework
import pandas as pd 
import numpy as np 
import cv2 # Used to manipulated the images 
np.random.seed(1337)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import concatenate as keras_concat
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

df_train = pd.read_json('J:/kaggle/iceberg/train.json') #read the data

def get_scaled_imgs(df): #reshape the data and create a new image channel
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 - band_2 # band_3 = band_1 + band_2 in original model

        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)

Xtrain = get_scaled_imgs(df_train) 
Ytrain = np.array(df_train['is_iceberg'])

#train only with (inc_angle not NA)
idx_tr = np.where(df_train.inc_angle!="na")
Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]

def get_more_images(imgs): #flip, increase samples
    
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
      
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
        
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
       
    more_images = np.concatenate((imgs,v,h))
    
    return more_images

Xtr_more = get_more_images(Xtrain)
Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))

def getModel_inception():
    #google incepyion with residual learning
    input_img = Input(shape=(75, 75, 3))
    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    x = keras_concat([input_img, tower_1, tower_2, tower_3], axis=-1)

    #add Conv and MaxPooling layers to reduce parameters
    x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # You must flatten the data for the dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    # output
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_img, outputs=outputs)
    model.compile(optimizer=Adam(lr=0.001,decay=0.0),loss='binary_crossentropy',metrics=['accuracy'])
        
    return model


# fit getModel_inception to be an example
model = getModel_inception()
model.summary()

batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('model_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)

model.load_weights(filepath = 'model_wts.hdf5')

score = model.evaluate(Xtrain, Ytrain, verbose=1)
print('Train score:', score[0])
print('Train accuracy:', score[1])

df_test = pd.read_json('J:/kaggle/iceberg/test.json')
Xtest = (get_scaled_imgs(df_test))
pred_test = model.predict(Xtest)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
print(submission.head(10))

#save submission
submission.to_csv('J:/kaggle/iceberg/submission.csv', index=False) 

