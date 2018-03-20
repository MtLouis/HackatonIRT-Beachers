
# coding: utf-8

# # Hackathon
# 
# Some utilities

# ## Import Utils

# In[4]:


get_ipython().system('pip install keras')


# In[5]:


ls


# In[6]:


import keras
import h5py as h5
import numpy as np

PATH_DATA = 'data/train/hackathon_data_train_eightieth.h5'
PATH_PREDICT_WITHOUT_GT = 'data/pred_students/pred_eighties_from_half_1_without_gt.h5'
PATH_SUBMIT = 'data/submit/pred_eighties_from_half_1_AWESOMEGROUP.h5'
PATH_PREDICT_WITH_GT = 'data/pred_teachers/pred_eighties_from_half_1.h5'


# In[471]:


BATCH_SIZE = 32
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout
import keras.layers.normalization 
from keras.callbacks import Callback
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[8]:


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


# In[9]:


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


# In[10]:


cd Hackathon


# In[255]:


idxs = get_idxs(PATH_DATA)
shuffled_idxs = shuffle_idx(idxs)
train_idxs, val_idxs = split_train_val(shuffled_idxs, 0.2)


# In[14]:


f = h5.File(PATH_DATA)


# In[48]:


countLabels = np.zeros([23,2])

for i in range(22):
    countLabels[i+1][0] = countLabels[i][0] + 1

for i in range(len(f['TOP_LANDCOVER'])):
    countLabels[int(f['TOP_LANDCOVER'][i])][1] += 1


# In[387]:


countLabels


# In[198]:


import matplotlib.pyplot as plt
imgplot = plt.imshow(im_I)
print(imgplot)


# In[20]:


import pandas as pd


# In[71]:


train_gen = generator(PATH_DATA, BATCH_SIZE, train_idxs)
train_batch_count = get_batch_count(train_idxs, BATCH_SIZE)

val_gen = generator(PATH_DATA, BATCH_SIZE, val_idxs)
val_batch_count = get_batch_count(val_idxs, BATCH_SIZE)


# In[116]:


print(train_batch_count, val_batch_count)


# # Analyse statistique

# In[51]:


D = np.zeros([23,70000,16,16,4])


# In[32]:





# In[52]:


countSaMere = np.zeros([23])

for i in range(len(f['S2'])):
    D[int(f['TOP_LANDCOVER'][i]),int(countSaMere[int(f['TOP_LANDCOVER'][i])]),:,:,:] = f['S2'][i]
    countSaMere[int(f['TOP_LANDCOVER'][i])] +=1


# In[54]:


np.mean(D[4,:int(countSaMere[4]),:,:,0])


# In[57]:


D[21,:,:,:,:]


# ### Calcul de moyenne

# In[55]:


meanColors = np.zeros([23,4])
for i in range(23):
    for j in range(4):
        meanColors[i][j] = np.mean(D[i,:int(countSaMere[i]),:,:,j])
    


# In[ ]:


test_idxs = get_idxs()


# In[76]:


np.around(meanColors, decimals=0)


# ### Calcul d'écart-type

# In[59]:


varColors = np.zeros([23,4])
for i in range(23):
    for j in range(4):
        varColors[i][j] = np.var(D[i,:int(countSaMere[i]),:,:,j])


# In[75]:


np.around(varColors,decimals=0)


# ### Affichage boxplot

# In[66]:


import matplotlib.pyplot as plt


# In[239]:


fourrier = np.zeros((len(f['S2']),4,16*16))


# ## Passage en spectrale 2D
# 

# In[125]:


import numpy.fft as ft
image_12 = (f['S2'][12][:,:,:])


# In[187]:


im_I=


# In[393]:


fft_I = ft.fft2(np.array(f['S2'][i][:,:,canal]))


# In[394]:


fft_I


# In[395]:


np.absolute(ft.fft2(np.array(f['S2'][1][:,:,0]))).reshape(1,16*16)[0]


# In[396]:


fourrier = np.zeros((len(f['S2']),4,256))
for i in idxs:
    if i%3400:
        print(i)
    for canal in range((4)):
        fourrier[i][canal]=np.absolute(ft.fft2(np.array(f['S2'][i][:,:,canal]))).reshape(1,16*16)[0]


# In[249]:


from sklearn.ensemble import RandomForestClassifier
for i in range(len(f['TOP_LANDCOVER'])):
    countLabels[int(f['TOP_LANDCOVER'][i])][1] += 1


# In[479]:


np.absolute(ft.fft2(np.array(f['S2'][0][:,:,canal]))).shape


# In[483]:


fourrier.reshape(234000,4,16,16)


# In[294]:


imp_class=[1,2,3,4,5,10,12,20]
indice = list()
for i in idxs:
    if f['TOP_LANDCOVER'][i]  in imp_class:
        indice.append(i)
        
imp_class_idxs=np.array(indice)

shuffled_idxs = shuffle_idx(imp_class_idxs)
train_idxs, val_idxs = split_train_val(shuffled_idxs, 0.2)


# In[291]:


num = np.zeros((1,4))
num[0,2]=3
np.delete(num,3)


# In[487]:



fourrier_train = fourrier[train_idxs]


# In[489]:


y_train = np.array(f['TOP_LANDCOVER'])[train_idxs]
fourrier_train.reshape(187104, 4, 16,16)


# In[486]:


rf512 = RandomForestClassifier(n_estimators= 80, criterion='entropy',max_depth=8,max_features=512,n_jobs=-1,verbose = 1)
rf512.fit(fourrier_train,y_train)


# In[409]:


y_test = np.array(f['TOP_LANDCOVER'])[val_idxs]
fourrier_test = fourrier[val_idxs].reshape(len(fourrier[val_idxs]),4*256)


# In[410]:


score512 = rf512.score(fourrier_test,y_test)


# In[412]:


score512


# In[305]:


len(train_idxs)


# In[445]:


y_pred = rf512.predict(fourrier_test)


# In[286]:


len(rf.feature_importances_)
rf.get_params(deep=True)


# ## Stats sur les fft 

# In[402]:


fft_R=np.zeros((234000))
fft_G=np.zeros((234000))
fft_B=np.zeros((234000))
fft_I=np.zeros((234000))

for i in range(234000):
    fft_R[i]=np.mean(fourrier[i,0,:])
    fft_G[i]=np.mean(fourrier[i,1,:])
    fft_B[i]=np.mean(fourrier[i,2,:])
    fft_I[i]=np.mean(fourrier[i,3,:])
    
    


# In[403]:



# Canal R

class_ = pd.DataFrame([f['TOP_LANDCOVER'][()].reshape(234000),fft_R,fft_G,fft_B,fft_I],index=['Class','Mean_R ','Mean_G ','Mean_B ','Mean_I ']).transpose()



# In[404]:


import matplotlib.pyplot as plt
plt.figure()
class_.boxplot(column='Mean_R ', by = 'Class', figsize=(12,6))
class_.boxplot(column='Mean_G ', by = 'Class', figsize=(12,6))
class_.boxplot(column='Mean_B ', by = 'Class', figsize=(12,6))
class_.boxplot(column='Mean_I ', by = 'Class', figsize=(12,6))



# In[405]:


fft_Ra=np.zeros((234000))
fft_Ga=np.zeros((234000))
fft_Ba=np.zeros((234000))
fft_Ia=np.zeros((234000))

for i in range(234000):
    fft_Ra[i]=np.var(fourrier[i,0,:])
    fft_Ga[i]=np.var(fourrier[i,1,:])
    fft_Ba[i]=np.var(fourrier[i,2,:])
    fft_Ia[i]=np.var(fourrier[i,3,:])
    
    


# In[390]:



# Canal R

class_ = pd.DataFrame([f['TOP_LANDCOVER'][()].reshape(234000),fft_Ra,fft_Ga,fft_Ba,fft_Ia],index=['Class','var_R ','var_G ','var_B ','var_I ']).transpose()



# In[391]:


import matplotlib.pyplot as plt
class_.boxplot(column='var_R ', by = 'Class', figsize=(12,6))
class_.boxplot(column='var_G ', by = 'Class', figsize=(12,6))
class_.boxplot(column='var_B ', by = 'Class', figsize=(12,6))
class_.boxplot(column='var_I ', by = 'Class', figsize=(12,6))


# ## RF sur mean et var

# In[ ]:


conca_fft=np.array([fft_R,fft_G,fft_B,fft_I,fft_Ra,fft_Ga,fft_Ba,fft_Ia])

rfM_V = RandomForestClassifier(n_estimators= 80, criterion='entropy',max_depth=8,max_features=1024,n_jobs=-1,verbose = 1)
rfM_V.fit(conca_fft,y_train)


# 
# ## SVM sur FFT

# In[ ]:



from sklearn.svm import SVC

SVM_FFT = SVC()
SVM_FFT.fit(fourrier_train,y_train) 





# # Instanciation du model

# In[272]:


resultat


# In[254]:


rf.predict()


# In[69]:


input_shape = (16,16,4)
model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Flatten())
model.add(Dense(23))
model.add(Activation('softmax'))


# # Fit

# In[72]:


# optim = keras.optimizers.Adam(lr=0.001)
optim = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_gen, steps_per_epoch=100, epochs=2, verbose=1, validation_data=val_gen, nb_val_samples=100)


# ## Prediction routines
# 
# In order to submit a result here are some gits

# In[257]:


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


# In[134]:


pred_idx = get_idxs(PATH_PREDICT_WITHOUT_GT)
print(len(pred_idx))
pred_gen = prediction_generator(PATH_PREDICT_WITHOUT_GT, BATCH_SIZE, pred_idx)
prediction = model.predict_generator(pred_gen, steps=get_batch_count(pred_idx, BATCH_SIZE), verbose=1)
print(len(prediction))
build_h5_pred_file(np.argmax(prediction, axis = 1), PATH_SUBMIT)


# In[433]:


f2 = h5.File(PATH_PREDICT_WITHOUT_GT)


# In[439]:


f2['S2']


# In[442]:


fourrier_test = np.zeros((len(f2['S2']),4,256))
for i in pred_idxs:
    if i%3400==0:
        print(i)
    for canal in range((4)):
        fourrier_test[i][canal]=np.absolute(ft.fft2(np.array(f2['S2'][i][:,:,canal]))).reshape(1,16*16)[0]


# In[413]:


def resultat_avec_csv(modele,name,path):

    pred_idxs = int(get_idxs(path))
    pred_gen = prediction_generator(path, BATCH_SIZE, pred_idx)
    prediction = modele.predict(pred_gen verbose=1)
    class_prediction = np.argmax(prediction, axis = 1)
    tosubmit = pd.DataFrame([pred_idx,class_prediction]).transpose()
    tosubmit.columns=["ID","TOP_LANDCOVER"]
    to_submit_csv = tosubmit.to_csv('%s.csv'%(name),sep=',',index= False)
    return prediction


# In[469]:


resultat_2 = rf.predict(fourrier_test.reshape(len(fourrier_test),4*256))
tosubmit_FFT = pd.DataFrame([pred_idxs,(resultat_512).astype(int)]).transpose()
tosubmit_FFT.columns=["ID","TOP_LANDCOVER"]


# In[468]:



to_submit_csv_FFT = tosubmit_FFT.to_csv('RF_2.csv',sep=',',index= False)


# In[430]:


prediction_generator(PATH_PREDICT_WITHOUT_GT, BATCH_SIZE, pred_idxs)


# In[441]:


pred_idx = get_idxs(PATH_PREDICT_WITHOUT_GT)
pred_idx


# ## Some ideas for monitoring

# In[135]:


def gt_generator(h5_path, batch_size, idxs):
    f = h5.File(h5_path, 'r')

    batch_count = get_batch_count(idxs, batch_size)
    
    for b in range(batch_count):
        batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
        batch_idxs = sorted(batch_idxs)
        Y = f['TOP_LANDCOVER'][batch_idxs, :]
        yield keras.utils.np_utils.to_categorical(np.array(Y), 23)

gt_gen = gt_generator(PATH_PREDICT_WITH_GT, BATCH_SIZE, pred_idx)
gt = []
for elem in gt_gen:
    gt.append(elem)
gt = np.vstack(gt)


# In[547]:


def generator(h5_path, batch_size, coucou):
    f = h5.File(h5_path, 'r')
    while True : 
        idxs = coucou
        batch_count = get_batch_count(idxs, batch_size)
        for b in range(batch_count):
            batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
            batch_idxs = sorted(batch_idxs)
            X = fourrier[batch_idxs, :,:,:]
            Y = f['TOP_LANDCOVER'][batch_idxs, :]
            yield np.array(X), keras.utils.np_utils.to_categorical(np.array(Y), 8)



# In[548]:


a = generator(PATH_DATA,BATCH_SIZE, train_idxs )
val_gen = generator(PATH_DATA, BATCH_SIZE, val_idxs)
# val_batch_count = get_batch_count(val_idxs, BATCH_SIZE)


# In[279]:


import matplotlib.pyplot as plt
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

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=7)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[276]:


def clean_confusion_matrix(confusion_matrix, classes):
    real_classes = []
    for c in range(len(classes)):
        if np.sum(confusion_matrix[:,c])+np.sum(confusion_matrix[c, :]) != 0:
            real_classes.append(c)
    real_confusion_matrix = np.empty((len(real_classes), len(real_classes)))  
    for c_index in range(len(real_classes)):
        real_confusion_matrix[c_index,:] = confusion_matrix[real_classes[c_index], real_classes]
    return real_confusion_matrix, real_classes


# In[303]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.metrics import confusion_matrix

real_cnf_matrix, real_classes = clean_confusion_matrix(confusion_matrix(y_test, y_pred, labels= imp_class), imp_class)
plot_confusion_matrix(real_cnf_matrix, classes = real_classes, normalize=True)


# In[544]:


#model 2

input_shape=(4,16,16)
model2 = Sequential()
model2.add(BatchNormalization(input_shape=input_shape))

model2.add(Conv2D(32, (5, 5), input_shape=input_shape,data_format="channels_first"))
model2.add(Activation('relu'))

model2.add(Conv2D(32, (5, 5),data_format="channels_first"))
model2.add(Activation('relu'))

model2.add(Conv2D(32, (4, 4),data_format="channels_first"))
model2.add(Activation('relu'))

model2.add(Conv2D(64, (3, 3),data_format="channels_first"))
model2.add(Activation('relu'))

model2.add(Conv2D(64, (3, 3),data_format="channels_first"))
model2.add(Activation('relu'))


model2.add(Flatten())
model2.add(Dense(64))
model2.add(Activation('relu'))
model2.add(Dropout(0.01))
model2.add(Dense(8))
model2.add(Activation('softmax'))


# In[545]:




# optim = keras.optimizers.Adam(lr=0.001)
optim = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model2.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[549]:


model2.fit_generator(a,steps_per_epoch=1000, epochs=10, verbose=1, validation_data=val_gen, nb_val_samples=100)


# In[568]:


countlabels = np.array([[0.000000e+00, 0.000000e+00],
       [1.000000e+00, 3.827110e+06],
       [2.000000e+00, 1.250253e+06],
       [3.000000e+00, 2.315736e+06],
       [4.000000e+00, 7.769660e+05],
       [5.000000e+00, 2.167443e+06],
       [6.000000e+00, 6.330000e+02],
       [7.000000e+00, 0.000000e+00],
       [8.000000e+00, 0.000000e+00],
       [9.000000e+00, 3.709000e+03],
       [1.000000e+01, 1.112499e+06],
       [1.100000e+01, 7.657050e+05],
       [1.200000e+01, 4.054392e+06],
       [1.300000e+01, 1.281000e+03],
       [1.400000e+01, 6.341420e+05],
       [1.500000e+01, 3.847000e+03],
       [1.600000e+01, 0.000000e+00],
       [1.700000e+01, 8.647100e+04],
       [1.800000e+01, 3.829000e+03],
       [1.900000e+01, 1.154414e+06],
       [2.000000e+01, 5.382200e+05],
       [2.100000e+01, 1.590000e+03],
       [2.200000e+01, 0.000000e+00]])


# In[569]:


tot = np.sum(countlabels[:,1])


# In[570]:




countlbl = np.zeros([23,2])
for i in range(22):
    countlbl[i+1][0] = countlbl[i][0] + 1
    
countlbl[:,1] = 100 * countlabels[:,1]/tot #np.around(100 * countLabels[:,1]/tot, decimals = 3)

countlbl



# In[571]:


tailleDB = 700000
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


# In[572]:




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



# In[573]:


import random


# In[574]:


tailleDB = 699438
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


# In[576]:


(countlbl[int(classBatch[0]),1])

