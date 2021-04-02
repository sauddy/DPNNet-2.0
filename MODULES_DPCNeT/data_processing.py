# import the necessary packages

import pandas as pd
import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow import keras as k
print("We are currently using the Modules_DPCNet")



def parse_dataset(dataset_path, filtering=True):
    '''
    Input : Address to the data folder
    Outpur : Return a csv with parameter data and path to the image

    '''

    dataset = load_csv(dataset_path, filtering=filtering)
    images = []
    for i in dataset["Sample#"]:
        imagePath = os.path.sep.join([dataset_path, "Disk_dust_plots/dust1_gap_{}.jpg".format(i)])
        images.append(imagePath)
    dataset["file"] = images
    return dataset


def load_csv(folder_address, filtering=True):

    dataset0 = pd.concat(map(pd.read_csv, glob.glob(folder_address + '*.csv')), ignore_index=True)

    ################## DATA Filtering #############
    if filtering is True:

        # Filtering 1
        dataset0 = dataset0[dataset0['Dust_gap_1'] > 0.05]  # filtering out very narrow gaps
        dataset = pd.concat([dataset0], ignore_index=True).sort_index()  # important when merging multiple datasets
        # df = shuffle(dataset)
        # dataset = df.reset_index(drop=True)
        dataset['Planet_Mass'] = dataset['Planet_Mass'] / (3 * 10**-6)  # writing in unit of earth mass

        # Filtering 2 (removing simualation with more than two gaps)
        dataset = dataset[dataset['#_DG'] < 2]  # keeping one and two dust gap disks
        # dataset_filtered = dataset.drop(columns=['Sample#']) # dropping the Sample#

        # dataset_filtered.to_csv('data_folder/dataset_filered.csv')   # saving the filtered data as csv file for future reference
        dataset = dataset.drop(columns=['Gas_gap_1', 'Dust_depth_1', 'Dust_gap_1', 'Dust_gap_2', 'Dust_depth_2', 'Gas_depth_1', '#_DG', '#_GG'])  # droping the unimportant columns
        # dataset.to_csv('../data_folder/dataset_filered.csv')
        # dataset = dataset[['Sample#','Planet_Mass']] # droping the unimportant columns
    #     dataset = dataset.sort_values(by="Sample#")

        ## cleaning the data##
        dataset.isna().sum()  # summing the number of na
        dataset0 = dataset.dropna()

    return dataset0


def parse_time_series_data(folder_addrss, list_of_orbits,path, data_filters=True):

    ''' Input : Address to the data folder and ahte orbits
        Output : Return a conacted csv with parameter data and path to the image with parse_dataset function
    '''
    dataset_complete = []
    # appended_data = []
    print("[INFO] preparing the dataframe from differnt times...")
    for i in range(len(list_of_orbits)):

        folder_address = folder_addrss + list_of_orbits[i] + '/'

        print("Reading the image paths and data from folder:", folder_address)
        # Loading the dataset for a given orbit
        dataset = parse_dataset(folder_address)

        # Appending the data from the pandas dataframe for each orbits
        dataset_complete.append(dataset)
    print("[INFO] The concatination of dataframes from differnt times are now complete")
    dataset_complete = pd.concat(dataset_complete, ignore_index=True, axis=0)
    dataset_complete.to_csv(path+'data_folder/dataset_complete.csv')  # saving as dataset_complete.csv
    return dataset_complete


def process_the_disk_attributes(train, test, path):

    ''' Input : train or test dataset
        path : to store the stats file
        Output : Return normalized data z normalization is used
    '''

    print("[INFO] preparing the normalized data training/testing split...")
    try: 
        train  = train.drop(columns=['Sample#', 'file']) ## dropping the necessary files
        test   = test.drop(columns=['Sample#', 'file']) ## dropping the necessary files
    except KeyError :
        pass
    
    train_stats = train.describe()
    train_stats.pop("Planet_Mass")
    train_stats = train_stats.transpose()

    train_stats.to_csv(path+'data_folder/train_stats.csv')

    ## The labels are not normalized
    train_labels = train.pop("Planet_Mass")
    test_labels = test.pop("Planet_Mass")


    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    normed_train_data = norm(train)

    normed_test_data = norm(test)
#     print(normed_train_data)
    print("[INFO] Done...")
    return normed_train_data, normed_test_data, train_labels, test_labels

def process_data_for_test(test, path):

    ''' 
    ### ADDED ON 7 JANUARY TO TEST DPCNET ON HIGH MASS DATA####

        Input : CSV containing data and path to the image
        path : to get the stat files from the original trained data
        Output : Return normalized data (z normalization is used)
    '''

    print("[INFO] preparing the normalized data TEST...")
    try: 
        # train  = train.drop(columns=['Sample#', 'file']) ## dropping the necessary files
        test   = test.drop(columns=['Sample#', 'file']) ## dropping the necessary files
    except KeyError :
        pass
    
    

    train_stats = pd.read_csv(path+'data_folder/train_stats.csv',index_col=0)
    # print(train_stats)
    ## The labels are not normalized
    
    test_labels = test.pop("Planet_Mass")


    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    

    normed_test_data = norm(test)
#     print(normed_train_data)
    print("[INFO] Done...")
    return normed_test_data, test_labels



def load_disk_images(dataset, X_res, Y_res, Type):

    ''' Input : dataset with path to the images 
        Output : Images set for either test or train
    '''

    print("[INFO] Loading images from {} data..".format(Type))
    images = []
    for image_path in dataset["file"]:    

        # dimensions for cropping the image
        left = 44
        top = 44
        right = 556
        bottom = 556    
        ## read the image corresponding to the path
        try:
            imagePath = image_path ## for regular code 
            # imagePath = '..'+image_path[33:] ## when reading path from the ones gnerated in COLAB as the address in COLAB gets modified
            image = cv2.imread(imagePath)  
            crop_image = image[left:right, top:bottom]
        except TypeError:
            print("you may need to specify the correct path to the images")
            imagePath = '..'+image_path[33:] ## when reading path from the ones gnerated in COLAB as the address in COLAB gets modified
            image = cv2.imread(imagePath) 
            crop_image = image[left:right, top:bottom]
        
        crop_image = image[left:right, top:bottom]

        crop_image = cv2.resize(crop_image, (X_res, Y_res))  # downsizing the image
        # crop_image = crop_image/255.0 # scaling

        ## ADDED for image normalization (standadization) on 31 Jan 2021
        crop_image = k.preprocessing.image.img_to_array(crop_image) ## changing to numpy array
        datagen = k.preprocessing.image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True,rescale= 1.0/255.0)
        crop_image = datagen.standardize(np.copy(crop_image))
        images.append(crop_image)
    print("{} Images are loaded".format(Type))
    return np.array(images)




def plot_history(history, path, Model,Network= None,res =None):
    try:
        hist = pd.DataFrame(history.history) ## is the data asalready a dataframe no need to convert
        path1 = path+'figures'
    except AttributeError:
        hist = history
        path1 = path+'figures_paper'
    hist['epoch'] = hist.index
    print(path1)
    plt.figure(figsize=(5, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error ($M_\oplus$)')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train ')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim([0, 30])
    plt.legend()
    
    if Model == 'CNN':
        plt.title("Single-input DPCNet")
        if Network == None:
           plt.savefig(path1+'/MAEvalidation_loss_CNN.pdf', format='pdf', dpi=300)
        else:
           plt.savefig(path1+'/MAEvalidation_loss_{}_{}.pdf'.format(Network,str(res)), format='pdf', dpi=300)
    else:
        plt.title("Multi-input DPCNet")
        if Network == None:
           plt.savefig(path1+'/MAEvalidation_loss_hybrid.pdf', format='pdf', dpi=300)
        else:
           plt.savefig(path1+'/MAEvalidation_loss_{}_{}_hybrid.pdf'.format(Network,str(res)), format='pdf', dpi=300)
       # plt.savefig(path1+'/MAEvalidation_loss_Hybrid.pdf', format='pdf', dpi=300)

    plt.figure(figsize=(5, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error ($M_\oplus^2$)')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                 label='Validation ')
    plt.ylim([0, 600])
    #   plt.xlim([0,700])
    #   plt.yscale("log")
    plt.legend()
    plt.tick_params(labelsize=10)
    plt.tick_params(axis='both', which='major', length=6, width=2)
    plt.tick_params(axis='both', which='minor', length=3, width=1.3)
    plt.tight_layout()
    if Model == 'CNN':
        plt.title("Single-input DPCNet")
        if Network == None:
           plt.savefig(path1+'/MSEvalidation_loss_CNN.pdf', format='pdf', dpi=300)
        else:
           plt.savefig(path1+'/MSEvalidation_loss_{}_{}.pdf'.format(Network,str(res)), format='pdf', dpi=300)
       # plt.savefig(path1+'/MSEvalidation_loss_CNN.pdf', format='pdf', dpi=300)     
    else:
        plt.title("Multi-input DPCNet")
        if Network == None:
           plt.savefig(path1+'/MSEvalidation_loss_hybrid.pdf', format='pdf', dpi=300)
        else:
           plt.savefig(path1+'/MSEvalidation_loss_{}_{}_hybrid.pdf'.format(Network,str(res)), format='pdf', dpi=300)
        #plt.savefig(path1+'/MSEvalidation_loss_Hybrid.pdf', format='pdf', dpi=300)

    plt.show()

