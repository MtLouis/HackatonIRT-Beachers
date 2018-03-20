
# coding: utf-8

# # Hackathon
# 
# Some utilities

# ## Import Utils

# In[1]:


ls


# In[2]:


get_ipython().system('pip install keras')


# In[ ]:


# cd ..


# In[19]:


import keras
import h5py as h5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
PATH_DATA = 'data/full.h5'
PATH_PREDICT_WITHOUT_GT = 'data/pred_eighties_from_full_1_without_gt.h5'
#PATH_SUBMIT = 'data/submit/pred_eighties_from_half_1_AWESOMEGROUP.h5'
#PATH_PREDICT_WITH_GT = 'data/pred_teachers/pred_eighties_from_half_1.h5'


# In[45]:


BATCH_SIZE = 32
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout
import keras.layers.normalization 
from keras.callbacks import Callback
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


f = h5.File(PATH_DATA)


# In[6]:


range(len(f['S2']))


# In[7]:


def get_idxs(h5_path):
    f = h5.File(h5_path)
    return range(len(f['S2']))

def shuffle_idx(sample_idxs):
    return list(np.random.permutation(sample_idxs))

def split_train_val(sample_idxs, proportion):
    n_samples = len(sample_idxs)
    return sample_idxs[:int((1.-proportion)*n_samples)], sample_idxs[int((1.-proportion)*n_samples):]

def get_batch_count(idxs, batch_size):
    batch_count = int(len(idxs)//batch_size)
    remained_samples = len(idxs)%batch_size
    if remained_samples > 0:
        batch_count += 1

    return batch_count


# In[8]:


def generator(h5_path, batch_size, idxs):
    f = h5.File(h5_path, 'r')
    while True : 
        idxs = shuffle_idx(idxs)
        batch_count = get_batch_count(idxs, batch_size)
        for b in range(batch_count):
            batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
            batch_idxs = sorted(batch_idxs)
            X = f['S2'][batch_idxs, :,:,:]
            Y = f['TOP_LANDCOVER'][batch_idxs, :]
            yield np.array(X), keras.utils.np_utils.to_categorical(np.array(Y), 23)


# In[9]:


idxs = get_idxs(PATH_DATA)
shuffled_idxs = shuffle_idx(idxs)
train_idxs, val_idxs = split_train_val(shuffled_idxs, 0.2)


# In[10]:


train_gen = generator(PATH_DATA, BATCH_SIZE, train_idxs)
train_batch_count = get_batch_count(train_idxs, BATCH_SIZE)

val_gen = generator(PATH_DATA, BATCH_SIZE, val_idxs)
val_batch_count = get_batch_count(val_idxs, BATCH_SIZE)


# In[11]:


countLabels = np.zeros([23,2])

for i in range(22):
    countLabels[i+1][0] = countLabels[i][0] + 1

for i in range(len(f['TOP_LANDCOVER'])):
    countLabels[int(f['TOP_LANDCOVER'][i])][1] += 1


# In[ ]:


countLabels


# In[ ]:


# import pickle
# print("countLabels saved")
# fileLabels = open('fileCountLabels.pickle', 'wb')
# pickle.dump(countLabels, fileLabels)
# fileLabels.close()


# In[12]:


# Read from file
import pickle
fileLabels = open('pickle.pickle', 'rb')
countLabels = pickle.load(fileLabels)  # variables come out in the order you put them in
fileLabels.close()
countLabels


# In[13]:


tot = np.sum(countLabels[:,1])


# In[121]:


countlbl = np.zeros([23,2])
for i in range(22):
    countlbl[i+1][0] = countlbl[i][0] + 1
    
countlbl[:,1] = 100 * countLabels[:,1]/tot #np.around(100 * countLabels[:,1]/tot, decimals = 3)

countlbl


# In[127]:


tailleDB = 2000000
DB = np.zeros([tailleDB, 16, 16, 4, 1])

# Correspondance : chaque élément est de type [i,j] où j est la classe originale reliée désormais à l'indice i 
cor = np.zeros([12,2], dtype=int)
for i in range(11):
    cor[i+1][0] = cor[i][0] + 1

tempLab = 0
for i in range(23):
    if countlbl[i,1] > 0.2:
        cor[tempLab][1] = i
        tempLab += 1

lblAdded = np.zeros([23,2], dtype = int)
cor


# In[124]:


int(f['TOP_LANDCOVER'][2])


# In[128]:


lblComplete = np.zeros([23,2], dtype=int)
for i in range(22):
    lblComplete[i+1][0] = lblComplete[i][0] + 1

for i in range(22):
    if i in cor[:,1]:
        lblComplete[i,1] = countlbl[i,1]/100 * tailleDB
    else:
        lblComplete[i,1] = 0
    
lblComplete
np.sum(lblComplete[:,1])


# In[22]:


plt.plot(f['TOP_LANDCOVER'][:2000000])


# In[23]:


dataBatch = f['S2'][10:10010]
dataClass = f['TOP_LANDCOVER'][10:10010]


# In[24]:


type(dataClass)


# In[26]:


import random


# In[ ]:


tailleDB = 1998403
picTot = 0

DB = np.zeros([tailleDB, 16, 16, 4])
classY = np.zeros(tailleDB, dtype=int)

countlbl = np.zeros([23,2])
for i in range(22):
    countlbl[i+1][0] = countlbl[i][0] + 1

batchLen = 1000

while picTot < tailleDB:
    ra = np.random.randint(0,18698240)
    dataBatch = f['S2'][ra:ra+batchLen]
    classBatch = f['TOP_LANDCOVER'][ra:ra+batchLen]
    pic = 0
    while pic < batchLen:
        if (countlbl[int(classBatch[pic]),1] < lblComplete[int(classBatch[pic]),1]) and (picTot < tailleDB):
            DB[picTot] = dataBatch[pic]
            classY[picTot] = classBatch[pic]
            picTot +=1
            countlbl[int(classBatch[pic]),1] +=1
            if picTot%1000==0:
                print(picTot)
        pic += 1


# In[37]:


import numpy.fft as ft
i=0
DB[i][:,:,0].shape


# In[41]:


fourrier = np.zeros((tailleDB,16,16,4))
for i in range(tailleDB):
    if i%3400==0:
        print(i)
    for canal in range((4)):
        fourrier[i][:,:,canal]=np.absolute(ft.fft2(DB[i][:,:,canal]))


# In[54]:


fourrier[12].shape



# In[49]:


input_shape=(16,16,4)

model2 = Sequential()
model2.add(BatchNormalization(input_shape=input_shape))

model2.add(Conv2D(32, (5, 5), input_shape=input_shape))

model2.add(Conv2D(32, (5, 5)))
model2.add(Activation('relu'))

model2.add(Conv2D(32, (4, 4)))
model2.add(Activation('relu'))

model2.add(Conv2D(64, (3, 3)))
model2.add(Activation('relu'))

model2.add(Conv2D(64, (3, 3)))
model2.add(Activation('relu'))


model2.add(Flatten())
model2.add(Dense(64))
model2.add(Activation('relu'))
model2.add(Dropout(0.01))
model2.add(Dense(23))
model2.add(Activation('softmax'))


# In[50]:


optim = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model2.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[71]:


def generator(h5_path, batch_size, coucou):
    f = h5.File(h5_path, 'r')
    while True : 
        idxs = coucou
        batch_count = get_batch_count(idxs, batch_size)
        for b in range(batch_count):
            batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
            batch_idxs = sorted(batch_idxs)
            X = fourrier[batch_idxs, :,:,:]
            Y = classY[batch_idxs]
            yield np.array(X), keras.utils.np_utils.to_categorical(np.array(Y), 23)


# In[ ]:



idxs_BD = range(tailleDB)
idxs_test = range(taille_test_db)


# In[72]:


shuffled_idxs = shuffle_idx(idxs)
train_idxs, val_idxs = split_train_val(shuffled_idxs, 0.2)


# In[73]:


train_gen = generator(PATH_DATA, BATCH_SIZE, train_idxs)
train_batch_count = get_batch_count(train_idxs, BATCH_SIZE)

val_gen = generator(PATH_DATA, BATCH_SIZE, val_idxs)
val_batch_count = get_batch_count(val_idxs, BATCH_SIZE)


# In[74]:


len(train_idxs)


# In[82]:


model2.fit_generator(train_gen,steps_per_epoch=1000, epochs=10, verbose=1, validation_data=val_gen, nb_val_samples=100)


# In[84]:


test = h5.File('data/pred_eighties_from_full_1_without_gt.h5')


# In[85]:


test['S2']


# In[87]:


taille_test_db = len(test['S2'])


# In[88]:


fourrier_test = np.zeros((taille_test_db,16,16,4))
for i in range(taille_test_db):
    if i%3400==0:
        print(i)
    for canal in range((4)):
        fourrier[i][:,:,canal]=np.absolute(ft.fft2(DB[i][:,:,canal]))


# In[93]:


def prediction_generator(batch_size, idxs):
    

    batch_count = get_batch_count(idxs, batch_size)
    
    for b in range(batch_count):
        batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
        batch_idxs = sorted(batch_idxs)
        X = fourrier_test[batch_idxs, :,:,:]
        yield np.array(X)


# In[96]:


pred_gen = prediction_generator(BATCH_SIZE,idxs_test)


# In[99]:


resultat = model2.predict_generator(pred_gen,steps=get_batch_count(idxs_test, BATCH_SIZE), verbose=1)


# In[101]:


class_prediction = np.argmax(resultat, axis = 1)
tosubmit = pd.DataFrame([idxs_test,class_prediction]).transpose()
tosubmit.columns=["ID","TOP_LANDCOVER"]
to_submit_csv = tosubmit.to_csv('FFT_2.csv',sep=',',index= False)


# In[103]:


tosubmit.head()
len(tosubmit)


# In[106]:


input_shape=(16,16,4)

model3 = Sequential()
model3.add(BatchNormalization(input_shape=input_shape))

model3.add(Conv2D(32, (5, 5), input_shape=input_shape))

model3.add(Conv2D(32, (5, 5)))
model3.add(Activation('relu'))

model3.add(Conv2D(32, (4, 4)))
model3.add(Activation('relu'))

model3.add(Conv2D(64, (3, 3)))
model3.add(Activation('relu'))

model3.add(Conv2D(64, (3, 3)))
model3.add(Activation('relu'))


model3.add(Flatten())
model3.add(Dense(64))
model3.add(Activation('relu'))
model3.add(Dropout(0.01))
model3.add(Dense(23))
model3.add(Activation('softmax'))


# In[109]:


def generator(h5_path, batch_size, coucou):
    f = h5.File(h5_path, 'r')
    while True : 
        idxs = coucou
        batch_count = get_batch_count(idxs, batch_size)
        for b in range(batch_count):
            batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
            batch_idxs = sorted(batch_idxs)
            X = DB[batch_idxs, :,:,:]
            Y = classY[batch_idxs]
            yield np.array(X), keras.utils.np_utils.to_categorical(np.array(Y), 23)


# In[ ]:


shuffled_idxs = shuffle_idx(idxs)
train_idxs, val_idxs = split_train_val(shuffled_idxs, 0.2)


# In[ ]:


train_gen = generator(PATH_DATA, BATCH_SIZE, train_idxs)
train_batch_count = get_batch_count(train_idxs, BATCH_SIZE)

val_gen = generator(PATH_DATA, BATCH_SIZE, val_idxs)
val_batch_count = get_batch_count(val_idxs, BATCH_SIZE)


# In[111]:


optim = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model3.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model3.fit_generator(train_gen,steps_per_epoch=1000, epochs=100, verbose=1, validation_data=val_gen, nb_val_samples=100)


# In[115]:


def prediction_generator(batch_size, idxs):
    

    batch_count = get_batch_count(idxs, batch_size)
    
    for b in range(batch_count):
        batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
        batch_idxs = sorted(batch_idxs)
        X = test['S2'][batch_idxs, :,:,:]
        yield np.array(X)


# In[116]:


pred_gen = prediction_generator(BATCH_SIZE,idxs_test)


# In[117]:


resultat = model3.predict_generator(pred_gen,steps=get_batch_count(idxs_test, BATCH_SIZE), verbose=1)


# In[118]:


class_prediction = np.argmax(resultat, axis = 1)
tosubmit = pd.DataFrame([idxs_test,class_prediction]).transpose()
tosubmit.columns=["ID","TOP_LANDCOVER"]
to_submit_csv = tosubmit.to_csv('model3_v1.csv',sep=',',index= False)


# In[119]:





# In[ ]:





# In[ ]:


import h5py

h5f = h5py.File('Db_proportions.h5', 'w')
h5f.create_dataset('dataset_prop', data=DB)
h5f.close()


# In[ ]:


DB[0,0,0,0]


# In[ ]:


verif = h5py.File('Db_proportions.h5')


# In[ ]:


for element in verif.items():
    print(element[0])
    print(element[1])
    print(element[1].name)
verif.close()


# In[ ]:


list_elmts = [key for key in verif['/'].keys()]
for key in list_elmts:
    print(key)
    print(type(verif['/'][key]))
    print(verif['/'][key])
    print([key for key in verif['/'][key].keys()])


# In[ ]:


# idxs_test = get_idxs(PATH_PREDICT_WITHOUT_GT)


# In[ ]:


print(train_batch_count, val_batch_count)


# # Instanciation du model

# In[ ]:


#model 1
input_shape = (16,16,4)
model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Conv2D(8,(5,5),activation='relu',input_shape =(16,16,4)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(8,(5,5),activation='relu',input_shape =(16,16,4)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(200,activation ='relu'))
model.add(Dropout(0.01))

model.add(Dense(23,activation ='softmax'))


# In[ ]:


#model 2
input_shape=(16,16,4)
model2 = Sequential()
model2.add(BatchNormalization(input_shape=input_shape))

model2.add(Conv2D(32, (5, 5), input_shape=input_shape))
model2.add(Activation('relu'))

model2.add(Conv2D(32, (5, 5)))
model2.add(Activation('relu'))

model2.add(Conv2D(32, (4, 4)))
model2.add(Activation('relu'))

model2.add(Conv2D(64, (3, 3)))
model2.add(Activation('relu'))

model2.add(Conv2D(64, (3, 3)))
model2.add(Activation('relu'))


model2.add(Flatten())
model2.add(Dense(64))
model2.add(Activation('relu'))
model2.add(Dropout(0.01))
model2.add(Dense(23))
model2.add(Activation('softmax'))


# # Fit

# In[ ]:


# optim = keras.optimizers.Adam(lr=0.001)
optim = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model2.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model2.fit_generator(train_gen, steps_per_epoch=100, epochs=4, verbose=1, validation_data=val_gen, nb_val_samples=100)


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))

for i, (name, values) in enumerate(history.history.items()):
    plt.subplot(1, len(history.history.items()), i+1)
    plt.plot(values)
    plt.title(name)


# ## Prediction routines
# 
# In order to submit a result here are some gits

# In[ ]:


import os 
def prediction_generator(h5_path, batch_size, idxs):
    f = h5.File(h5_path, 'r')

    batch_count = get_batch_count(idxs, batch_size)
    
    for b in range(batch_count):
        batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
        batch_idxs = sorted(batch_idxs)
        X = f['S2'][batch_idxs, :,:,:]
        yield np.array(X)

def build_h5_pred_file(pred, h5_output_path):
    if os.path.exists(h5_output_path):
        os.remove(h5_output_path)
    f = h5.File(h5_output_path, 'w')
    top_landcover_submit = f.create_dataset("TOP_LANDCOVER", (len(pred), 1), maxshape=(None, 1))
    top_landcover_submit[:, 0] = pred
    f.close()
    
    return 1


# In[ ]:


pred_idx = get_idxs(PATH_PREDICT_WITHOUT_GT)
print(len(pred_idx))
pred_gen = prediction_generator(PATH_PREDICT_WITHOUT_GT, BATCH_SIZE, pred_idx)
prediction = model2.predict_generator(pred_gen, steps=get_batch_count(pred_idx, BATCH_SIZE), verbose=1)
print(prediction)
#build_h5_pred_file(np.argmax(prediction, axis = 1), PATH_SUBMIT)


# In[ ]:


def resultat_avec_csv(modele,name,path):

    pred_idx = get_idxs(path)
    pred_gen = prediction_generator(path, BATCH_SIZE, pred_idx)
    prediction = modele.predict_generator(pred_gen, steps=get_batch_count(pred_idx, BATCH_SIZE), verbose=1)
    class_prediction = np.argmax(prediction, axis = 1)
    tosubmit = pd.DataFrame([pred_idx,class_prediction]).transpose()
    tosubmit.columns=["ID","TOP_LANDCOVER"]
    to_submit_csv = tosubmit.to_csv('%s.csv'%(name),sep=',',index= False)
    return prediction

def resultat(modele,path):

    pred_idx = get_idxs(path)
    pred_gen = prediction_generator(path, BATCH_SIZE, pred_idx)
    prediction = modele.predict_generator(pred_gen, steps=get_batch_count(pred_idx, BATCH_SIZE), verbose=1)


    return prediction


# In[ ]:


essai =resultat(model2,PATH_DATA)


# In[ ]:


resultat_avec_csv(model2,"jubois_palmi",PATH_PREDICT_WITHOUT_GT)


# ## Some ideas for monitoring

# In[ ]:


7700/32


# In[ ]:


def gt_generator(h5_path, batch_size, idxs):
    f = h5.File(h5_path, 'r')

    batch_count = get_batch_count(idxs, batch_size)
    print(batch_count)
    for b in range(batch_count):
        if (b+1)*batch_size<
        batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
        batch_idxs = sorted(batch_idxs)
        print(max(batch_idxs))
        Y = f['TOP_LANDCOVER'][batch_idxs, :]
        yield keras.utils.np_utils.to_categorical(np.array(Y), 23)

gt_gen = gt_generator(PATH_DATA, BATCH_SIZE, pred_idx)
gt = []
for elem in gt_gen:
    gt.append(elem)
gt = np.vstack(gt)


# In[ ]:


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' #if normalize else '.i'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=7)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


def clean_confusion_matrix(confusion_matrix, classes):
    real_classes = []
    for c in range(len(classes)):
        if np.sum(confusion_matrix[:,c])+np.sum(confusion_matrix[c, :]) != 0:
            real_classes.append(c)
    real_confusion_matrix = np.empty((len(real_classes), len(real_classes)))  
    for c_index in range(len(real_classes)):
        real_confusion_matrix[c_index,:] = confusion_matrix[real_classes[c_index], real_classes]
    return real_confusion_matrix, real_classes


# In[ ]:


list_top=list(f['TOP_LANDCOVER'])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.metrics import confusion_matrix
y_true = np.array(list_top)
y_pred = np.argmax(essai, axis = 1)

real_cnf_matrix, real_classes = clean_confusion_matrix(confusion_matrix(y_true, y_pred, labels= range(23)), range(23))
plot_confusion_matrix(real_cnf_matrix, classes = real_classes, normalize=True)


# In[ ]:


list_top[:20][0][0]


# In[ ]:


len(list_top)

