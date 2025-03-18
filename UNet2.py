#!/usr/bin/env python
# coding: utf-8

# # Kidney Stone Segmentation Using U-Net and Fully Convolutional Networks

# ## Importing Libraries

# In[1]:


import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import os
import random
import numpy as np
 
from tqdm import tqdm 

from PIL import Image
import cv2
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# ## Parameters and Variable

# In[2]:


seeds = [13, 42, 1, 83, 76]
np.random.seed = seeds[0]

num_classes = 1
split_size = 0.5


IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1

DATA_PATH = 'data/'

data_ids = next(os.walk(DATA_PATH+'/image'))[2]

X = np.zeros((len(data_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
y = np.zeros((len(data_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR']= 'cuda_malloc_async'


# ## Read Dataset

# In[3]:


print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(data_ids), total=len(data_ids)):   
    path = DATA_PATH
    img = imread(path + '/image/' + id_)[:,:]
    img = img.reshape(img.shape[0], img.shape[1], IMG_CHANNELS)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X[n] = img  #Fill empty X_train with values from img
    
    mask = imread(path + 'label/' + id_)
    mask = (mask >= 250)
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
    y[n] = mask 
    #plt.axis("off")
    #imshow(y[n])
    #plt.show()


# ## Sample Image and Ground-Truth Label from Dataset

# In[4]:


image_x = random.randint(0, len(X))
plt.axis("off")
imshow(X[image_x])
plt.show()
plt.axis("off")
imshow(np.squeeze(y[image_x]))
plt.show()


# In[5]:


def unet():
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    b1 = tf.keras.layers.BatchNormalization()(c1)
    r1 = tf.keras.layers.ReLU()(b1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)
    
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    b2 = tf.keras.layers.BatchNormalization()(c2)
    r2 = tf.keras.layers.ReLU()(b2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)
     
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    b3 = tf.keras.layers.BatchNormalization()(c3)
    r3 = tf.keras.layers.ReLU()(b3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)
     
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    b4 = tf.keras.layers.BatchNormalization()(c4)
    r4 = tf.keras.layers.ReLU()(b4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(r4)
     
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    b5 = tf.keras.layers.BatchNormalization()(c5)
    r5 = tf.keras.layers.ReLU()(b5)
    c5 = tf.keras.layers.Dropout(0.3)(r5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.BatchNormalization()(u6)
    u6 = tf.keras.layers.ReLU()(u6)
    
     
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.BatchNormalization()(u7)
    u7 = tf.keras.layers.ReLU()(u7)
    
     
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.BatchNormalization()(u8)
    u8 = tf.keras.layers.ReLU()(u8)
     
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    u9 = tf.keras.layers.BatchNormalization()(u9)
    u9 = tf.keras.layers.ReLU()(u9)
    
     
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(u9)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

    return model


# In[ ]:


acc = []
jacc = []
f1 = []
prec = []
rec = []


for f in range(3, len(seeds)):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_size, random_state=seeds[f])


    model = unet()
    
    checkpoint_filepath = 'model1_' + str(f+1)+'fold.h5'
    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs'),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                monitor='val_recall',
                mode='max',
                save_best_only=True,
                verbose=1)]
    
    model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=16, epochs=300, callbacks=callbacks)

    for i in range(0, len(X_val)):
        sample_image = X_val[i]
        sample_mask = y_val[i].astype(np.uint8).flatten()
        prediction = model.predict(sample_image[tf.newaxis, ...],verbose=0)[0]
        predicted_mask = (prediction > 0.5).astype(np.uint8).flatten()
            
        acc.append(accuracy_score(sample_mask, predicted_mask))
        jacc.append(jaccard_score(sample_mask, predicted_mask))
        f1.append(f1_score(sample_mask, predicted_mask))
        prec.append(precision_score(sample_mask, predicted_mask))
        rec.append(recall_score(sample_mask, predicted_mask))

    del model 

    tf.keras.backend.clear_session()
    
    model2 = unet()
    
    checkpoint_filepath = 'model2_' + str(f+1)+'fold.h5'
    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs'),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                monitor='val_recall',
                mode='max',
                save_best_only=True,
                verbose=1)]
    
    model2.fit(X_val, y_val, validation_data=(X_train,y_train), batch_size=16, epochs=300, callbacks=callbacks)

    for i in range(0, len(X_train)):
            sample_image = X_train[i]
            sample_mask = y_train[i].astype(np.uint8).flatten()
            prediction = model2.predict(sample_image[tf.newaxis, ...],verbose=0)[0]
            predicted_mask = (prediction > 0.5).astype(np.uint8).flatten()
                
            acc.append(accuracy_score(sample_mask, predicted_mask))
            jacc.append(jaccard_score(sample_mask, predicted_mask))
            f1.append(f1_score(sample_mask, predicted_mask))
            prec.append(precision_score(sample_mask, predicted_mask))
            rec.append(recall_score(sample_mask, predicted_mask))
        
    del model2
    tf.keras.backend.clear_session()
    
print("Accuracy: "+ np.mean(acc) + "+- " + np.std(acc))
print("Jaccard: "+ np.mean(jacc) + "+- " + np.std(jacc))
print("Dice: "+ (2*np.mean(jacc))/(1+np.mean(jacc)) + "+- " + np.std(2*jacc/(1+jacc)))
print("F1 Score: "+ np.mean(f1)+ "+- " + np.std(f1))
print("Precision: "+ np.mean(prec) + "+- " + np.std(prec))
print("Recall: "+ np.mean(rec) + "+- " + np.std(rec))


# In[ ]:


loss = model.history.history['loss']
val_loss = model.history.history['val_loss']

plt.figure()
plt.plot( loss, 'r', label='Training loss')
plt.plot( val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()


# In[ ]:


model = tf.keras.models.load_model('model_20_06_2023.h5')


# In[ ]:


model.evaluate(X_val, y_val)


# In[ ]:


def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input image', 'True mask', 'Predicted mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]), cmap='gray')
    plt.axis('off')
  plt.show()
  
i = random.randint(0, len(X_val))
sample_image = X_val[i]
sample_mask = y_val[i]
prediction = model.predict(sample_image[tf.newaxis, ...])[0]
predicted_mask = (prediction > 0.5).astype(np.uint8)
display([sample_image, sample_mask,predicted_mask])


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

y_test_score = y_val[i].astype(np.uint8).flatten()
predicted_mask_score = predicted_mask.flatten()
print("Acuraccy: ", accuracy_score(y_test_score, predicted_mask_score))
print("Jaccard: ", jaccard_score(y_test_score, predicted_mask_score))
print("F1 Score: ", f1_score(y_test_score, predicted_mask_score))
print("Precision: ", precision_score(y_test_score, predicted_mask_score))
print("Recall: ", recall_score(y_test_score, predicted_mask_score))


# In[ ]:


y_pred = []
acc = []
jacc = []
f1 = []
prec = []
rec = []
for i in range(0, len(X_val)):
    sample_image = X_val[i]
    sample_mask = y_val[i].astype(np.uint8).flatten()
    prediction = model.predict(sample_image[tf.newaxis, ...],verbose=0)[0]
    predicted_mask = (prediction > 0.5).astype(np.uint8).flatten()
        
    acc.append(accuracy_score(sample_mask, predicted_mask))
    jacc.append(jaccard_score(sample_mask, predicted_mask))
    f1.append(f1_score(sample_mask, predicted_mask))
    prec.append(precision_score(sample_mask, predicted_mask))
    rec.append(recall_score(sample_mask, predicted_mask))


print("Accuracy: ", np.mean(acc))
print("Jaccard: ", np.mean(jacc))
print("Dice: ", (2*np.mean(jacc))/(1+np.mean(jacc)))
print("F1 Score: ", np.mean(f1))
print("Precision: ", np.mean(prec))
print("Recall: ", np.mean(rec))


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

y_test_score = y_val.astype(np.uint8).flatten()
y_pred = y_pred.flatten()
#print("Acuraccy: ", accuracy_score(y_test_score, y_pred))
print("Jaccard: ", jaccard_score(y_test_score, y_pred))
print("F1 Score: ", f1_score(y_test_score, y_pred))
print("Precision: ", precision_score(y_test_score, y_pred))
print("Recall: ", recall_score(y_test_score, y_pred))


# Acuraccy:  0.9998064804077148
# Jaccard:  0.7504550150031974
# F1 Score:  0.8574399325558522
# Precision:  0.9923892538866844
# Recall:  0.7547991292301603

# In[ ]:





# In[ ]:




