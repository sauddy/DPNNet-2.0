#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sauddy/DPCNet/blob/main/DPCNet_V1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## DPCNet_Build -- 26 November  2020 Colab Compatible

# In[1]:


# '''
#     Please note this version of the code is compatible with Google colab 
#     and will also run on Local machine  

# '''
# mount the drive if running from Colab
# from google.colab import drive
# drive.mount('/content/drive')


# In[2]:


##### IDEA Behind this notebook : ###########
## Author : Sayantan 
## Created : 26 November 2020
## This notebook is adopted from the CNN_DPNNet V3 and V5 (old versions)
## RESNET50, ALEXNET, VGG16 are implemented by Ramit Dey


## We develop a modular notebook that does the following:
## Import all the customized Modules from Modules_DPCNet 
## For data processing we use data_processing.py script 
## A function module to call the different networks independently. (deep_models.py, other_cnn.py)



# In[3]:


# import the necessary packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import cv2
import os
import csv

## Modules to check the performance of the code
from time import process_time 
# !pip install memory_profiler ## When running from Google Colab
# import memory_profiler as mem_profile
# print('Memory (Before): {}Mb'.format(mem_profile.memory_usage()))


## Importing the necessary TesnorFLow modules modules
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')

from tensorflow import keras
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score ## form calcualting the r2 score



# In[4]:


############ Please provide the same path to the code directory if using Colab################

Path_gdrive= '/content/drive/MyDrive/CNN_DPNNET/DPCNET_dev/' ## Comment out this line if using local computer

## Importing the Modules from Modules_DPCNet
import sys
try: ## tries to find the modules in the local directory first
  current_directory = os.getcwd()
  path = current_directory + '/' # For local computer 
#   path = '' # For local computer  
  sys.path.append(path+'MODULES_DPCNeT')
  import data_processing as dp
  import deep_models as dm
  import other_cnns as ocn

  ########### Folders to save the processed data, files and figures when using Local computer ##############
  output_folder_list = ['data_folder1','figures1','saved_model1']
  for file in output_folder_list:
    try:
        os.makedirs(file)
    except OSError:
        print ("Creation of the directory %s failed/ not needed as it already exit" % file)
    else:
        print ("Successfully created the directory %s" % file)
  
except ModuleNotFoundError:
  
  # #For Colab use:
  # #Point to the path containing the modules in the above section
  #(data folder are a directory above the directory containing the notebook)
  try:
    path = Path_gdrive
    print(path)
    sys.path.append(path+'MODULES_DPCNeT')
    import data_processing as dp
    import deep_models as dm
    import other_cnns as ocn

    ########### Folders to save the processed data, files and figures when using GDRIVE ##############
    import os
    os.chdir(path)
    print("Creating the folders")
    get_ipython().system('mkdir -p data_folder')
    get_ipython().system('mkdir -p figures ## to save the figurs')
    get_ipython().system('mkdir -p figures_paper')
    get_ipython().system('mkdir -p saved_model')
  except ModuleNotFoundError:
    print("The path to the modules is incorrect-- Provide current path")

print("[INFO] Modules imported")


# ## Load data csv (including the path to images) from multiple time-instances

# In[5]:


############# Address to the data folder ###################
list_of_orbits = ['150','140','130','120']

print("[INFO]: Importing files from the datafolder")
## Note the data folder are a directory above the code directory
## Please contact the authors if you need access to the data
folder_address = path + "../analysis_output_"

# The idea is to generate a dataframe with the parameters and the path to the images
data_complete = dp.parse_time_series_data(folder_address,list_of_orbits,path)


# ## Preparing data 

# In[6]:


## partition the data csv file into training and testing splits using 85% of
## the data for training and the remaining 15% for testing
split = train_test_split(data_complete, test_size=0.15, random_state=42)
(train, test) = split

## Save the train and the test data for future use as well.
test.to_csv(path+'data_folder/test_dataset.csv')
train.to_csv(path+'data_folder/train_dataset.csv')

## Generate the Normalized data
normed_train_data, normed_test_data, train_labels, test_labels = dp.process_the_disk_attributes(train, test, path)


#### Desired Image resoltuion #####
X_res = Y_res = 32

## Generate the training and the test images 

trainImagesX = dp.load_disk_images(train, X_res, Y_res, Type = "Train")
testImagesX = dp.load_disk_images(test, X_res, Y_res, Type = "Test")

Validation_split = 0.15 # 15 percent of the training data is used for validation
# print('Memory (After Loading): {}Mb'.format(mem_profile.memory_usage()))
print('There are {} Train, {} Validation and {} Test images'.format(int((1-Validation_split)*len(normed_train_data)),int(Validation_split*len(normed_train_data)),len(normed_test_data)))## check the numbers in each chategory


# ## Training the CNN 

# In[7]:


## Hyper-Parameter to define
batch_size = 20 ## the best was for 200 last run
valid_batch_size = 20
epochs=200 ## best was 100
init_lr = 1e-5

# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.2, patience=20, verbose=1, mode='min',restore_best_weights=True)


# In[8]:


## Select the Network type

# NETWORK = "Vanilla"
# NETWORK = "ALEXNET"
# NETWORK = "VGG"
NETWORK = "RESNET50"

print('INFO: Currently training using the {} NETWORK'.format(NETWORK))
if NETWORK == "Vanilla":
    CNN = dm.build_cnn(X_res, Y_res, 3, regress=True)
elif NETWORK == "ALEXNET":
    CNN = ocn.alexnet(X_res, Y_res, 3, regress=True)
elif NETWORK == "VGG":
    CNN = ocn.cnn_vgg(X_res, Y_res, 3, regress=True)
elif NETWORK == "RESNET50":
    CNN = ocn.ResNet50(X_res, Y_res, 3)
# optimizer = tf.keras.optimizers.Adam(lr_schedule)
optimizer = tf.keras.optimizers.Adam(init_lr, decay=init_lr/2000)
CNN.compile(loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mean_absolute_error', 'mean_squared_error'])
CNN_history = CNN.fit(x=trainImagesX, y=train_labels,
                  validation_split = 0.15,epochs=epochs, batch_size=batch_size,callbacks=[early_stop])
# print('Memory (After Training): {}Mb'.format(mem_profile.memory_usage()))


# In[ ]:


## The plots for the Validation and the Testing loss
dp.plot_history(CNN_history,path, Model = "CNN")
hist_df = pd.DataFrame(CNN_history.history)  ## converting to dataframe
## Saving the history
hist_df.to_csv(path+'data_folder/'+NETWORK+'_'+str(X_res)+'_history.csv')


# ## Saving the network for future use

# In[ ]:


# uncomment the following lines if you want to update your model
CNN.save(path+'saved_model/'+NETWORK+'_'+str(X_res)+'_model')


# In[ ]:


## Loading the model
CNN = tf.keras.models.load_model(path+'saved_model/'+NETWORK+'_'+str(X_res)+'_model')
##Check its architecture
# CNN.summary()


# ## Model Evaluation for DPCNet

# In[ ]:


loss, mae, mse_CNN = CNN.evaluate(testImagesX, test_labels, verbose=0)
print("Testing set Mean Square Error for {}: {:5.2f} ".format(NETWORK,mse_CNN))
print("Testing set Root Mean Square Error for {}: {:5.2f} M_Earth".format( NETWORK,np.sqrt(mse_CNN)))
print("Testing set Mean Abs Error for {} : {:5.2f} M_Earth ".format(NETWORK,mae))
print("Testing set Loss for {}: {:5.2f} M_Earth".format(NETWORK,loss))


# ## Implementing the hybrid model (multi-input), i.e., DPCNet + DPNNet##

# In[ ]:


DPNNet = dm.DPNNet_build(normed_train_data.shape[1], regress=False)



if NETWORK == "Vanilla":
    CNN_ = dm.build_cnn(X_res, Y_res, 3, regress=False)
elif NETWORK == "ALEXNET":
    CNN_ = ocn.alexnet(X_res, Y_res, 3, regress=False)
elif NETWORK == "VGG":
    CNN_ = ocn.cnn_vgg(X_res, Y_res, 3, regress=False)
elif NETWORK == "RESNET50":
    CNN_ = ocn.ResNet50(X_res, Y_res, 3)

combinedInput = concatenate([DPNNet.output,CNN_.output])
# our final FC layer head will have two dense layers, the final one being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)


# ## Training the Hybrid Model (DPCNet + DPNNet)

# In[ ]:


epochs=200 ## best was 100
init_lr = 1e-4
hybrid_model = Model(inputs=[DPNNet.input, CNN_.input], outputs=x)
# optimizer = tf.keras.optimizers.Adam(lr_schedule)
optimizer = tf.keras.optimizers.Adam(init_lr,decay=init_lr /200) #, 
hybrid_model.compile(loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mean_absolute_error', 'mean_squared_error'])
print('INFO: Currently training using the {} NETWORK and DPNNet'.format(NETWORK))
history_hybrid = hybrid_model.fit(x=[normed_train_data, trainImagesX], y=train_labels,
    validation_split = 0.15,verbose=1,
    epochs=epochs, batch_size=batch_size,callbacks=[early_stop])


# In[ ]:


dp.plot_history(history_hybrid,path,Model = "Hybrid")
hist_hybrid = pd.DataFrame(history_hybrid.history) 
hist_hybrid.to_csv(path+'data_folder/'+NETWORK+'_'+str(X_res)+'_history_hybrid.csv')


# In[ ]:


hybrid_model.save(path+'saved_model/'+NETWORK+'_'+str(X_res)+'_hybrid_model') 
hybrid_model = tf.keras.models.load_model(path+'saved_model/'+NETWORK+'_'+str(X_res)+'_hybrid_model')


# ## Model Evaluation for Hybrid Model

# In[ ]:


loss, mae, mse_hybrid = hybrid_model.evaluate([normed_test_data,testImagesX], test_labels, verbose=0)
print("Testing set Mean Square Error for {} with DPNNet: {:5.2f} ".format(NETWORK,mse_hybrid))
print("Testing set Root Mean Square Error for {} with DPNNet: {:5.2f} M_Earth".format( NETWORK,np.sqrt(mse_hybrid)))
print("Testing set Mean Abs Error for {} with DPNNet: {:5.2f} M_Earth ".format(NETWORK,mae))
print("Testing set Loss for {} with DPNNet: {:5.2f} M_Earth".format(NETWORK,loss))


# ## Model Predictions and Results

# In[ ]:


pred_CNN = CNN.predict(testImagesX)
np.shape(pred_CNN)

pred_Hybird = hybrid_model.predict([normed_test_data,testImagesX])
np.shape(pred_Hybird)


# In[ ]:


plt.style.use('classic')
plt.figure(figsize = (5,5))
# test_predictions = model.predict(normed_test_data).flatten()
plt.scatter(test_labels,pred_CNN.flatten(),s=30,marker='d',color='r')

score_CNN = r2_score(test_labels,pred_CNN.flatten())
plt.text(20,110,r" r2 = {:.3f}".format(score_CNN), fontsize =14)
plt.xlabel(r'True values of planet mass($M_\oplus$)', fontsize=15)
plt.ylabel(r'Predicted planet mass($M_\oplus$)',fontsize=15)
plt.title("DPCNet Prediction")
plt.axis('equal')
plt.axis('square')
plt.xlim(5,125)
plt.ylim(5,125)

# plt.xlim([0.6,plt.xlim()[1]])
# plt.ylim([0.6,plt.xlim()[1]])
_ = plt.plot([0, 120], [0, 120],linewidth=2)
plt.minorticks_on() 
plt.tight_layout()
plt.savefig('figures/predicted_correlation_CNN.pdf',format='pdf',dpi=300)

plt.tick_params(labelsize=15)

plt.tick_params(axis='both', which='major',length=6, width=2)
plt.tick_params(axis='both', which='minor',length=3, width=1.3)
plt.figure(figsize = (5,5))
score_HYBRID = r2_score(test_labels,pred_Hybird.flatten())
plt.text(20,110,r"r2 = {:.3f}".format(score_HYBRID),fontsize =14)
plt.scatter(test_labels,pred_Hybird.flatten(),s=20,marker='d',color='r')
plt.title("Hybrid Prediction")
plt.xlabel(r'True values of planet mass($M_\oplus$)', fontsize=15)
plt.ylabel(r'Predicted planet mass($M_\oplus$)',fontsize=15)
plt.axis('equal')
plt.axis('square')
plt.xlim(5,125)
plt.ylim(5,125)

# plt.xlim([0.6,plt.xlim()[1]])
# plt.ylim([0.6,plt.xlim()[1]])
_ = plt.plot([0, 120], [0, 120],linewidth=2)

plt.minorticks_on() 
plt.tick_params(labelsize=15)
plt.tick_params(axis='both', which='major',length=6, width=2)
plt.tick_params(axis='both', which='minor',length=3, width=1.3)
plt.tight_layout()
plt.savefig('figures/predicted_correlation_hybrid.pdf',format='pdf',dpi=300)
# print("{} r2 score is {}".format(NETWORK,score_CNN))
# print("{} + DPPNET r2 score is {}".format(NETWORK,score_HYBRID))


# In[ ]:


csv_path = path+'data_folder'
output_filename = os.path.join(csv_path,'res_error'+'.csv')
csv_file = open(output_filename,'a')
csv_writer = csv.writer(csv_file)
# csv_writer.writerow(['Resolution','MSE_CNN','R2_CNN', 'MSE_HYBRID','R2_HYBRID'])
csv_writer.writerow([X_res,mse_CNN,score_CNN,mse_hybrid,score_HYBRID])
csv_file.close()


# In[ ]:




