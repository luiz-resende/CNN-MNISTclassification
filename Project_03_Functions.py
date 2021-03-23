# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# -*- coding: utf-8 -*-
"""
@author: Luiz Resende Silva
"""
############################################################################################################################
'''                                           IMPORTING GENERAL LIBRARIES                                                '''
############################################################################################################################
import pandas as pd
import numpy as np
import scipy
import seaborn as sb #Graphical plotting library
import matplotlib.pyplot as plt #Graphical plotting library
import pickle as pkl #Pickle format library
import time #Library to access time and assess running performance of the NN
import pdb #Library to create breakpoints

from scipy.sparse import hstack
############################################################################################################################
'''                                IMPORT SCIKIT-LEARN PREPROCESSING AND METRICS MODULES                                 '''
############################################################################################################################
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

############################################################################################################################
'''                                         IMPORT PYTORCH MODULES/LIBRARY                                               '''
############################################################################################################################
import torch as th
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as nf
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

##################################################################################################################################
'''                               GENERAL FUNCTIONS TO READ, WRITE, PLOT AND HABDLE DATA                                       '''
##################################################################################################################################
def Read_File_DF(File_Name, separation=",", head=None, replace=[], drop=False):
    """ Function created to read dataset files and import information as a Pandas DataFrame
    INPUT: Dataset file name, character delimiter in the file (default ','), flag for file header (default 0 - no header head=None),
            list of strings to find possible malformations (default empty) and flag to drop or not lines/columns
            containing such values (default False)"""
    try:
        separation = separation.lower()
        if(separation == "space" or separation == "tab"):
            separation = "\t"
        Raw_Data_Set = pd.read_csv(File_Name, delimiter=separation, header=head, na_values=replace)
        RawRowsColumns = Raw_Data_Set.shape
        if(replace != None):
            Missing = Raw_Data_Set.isnull().sum().sum()
            print("Total number of missing/anomalous 'entries' in the data set: ",Missing)
            if(drop == True):
                Raw_Data_Set.dropna(axis=0, how='any', inplace=True)
                CleanRowsColumns = Raw_Data_Set.shape
                print("Number of examples with missing values deleted from data set: ",(RawRowsColumns[0]-CleanRowsColumns[0]))
        return Raw_Data_Set
    except:
        print("READ_FILE_ERROR\n")

def Write_File_DF(Data_Set, File_Name="Predictions", separation=",", head=True, ind=False, dec='.'):
    """ Function created to write a Pandas DataFrame containing prediciton made by classifiers to submit to competition
    INPUT: DataFrame with IDs and predictions, file name (default 'Predictions'), character delimiter in the file (default ','), 
            flag to include file header (default True), flag to include column of indices (default False)
            and character for decimals (default '.') """
    try:
        separation = separation.lower()
        if(separation == "space" or separation == "tab"):
            separation = "\t"
        timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
        name = timestr+File_Name+".csv"
        Data_Set.to_csv(path_or_buf=name, sep=separation, na_rep='', float_format=None, columns=None, header=head, index=ind,
                        index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"',
                        line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal=dec)
        return print(File_Name+" exported as .csv file")
    except:
        print("WRITE_FILE_ERROR\n")

def To_PD_DF(Data_Set):
    """ Function created to convert NumPy array to Pandas DataFrame 
    INPUT: NumPy array/matrix containing data """
    try:
        Data_Set_DF = pd.DataFrame(Data_Set, index = (list(range(0,Data_Set.shape[0]))), columns = (list(range(0,Data_Set.shape[1]))))
        return Data_Set_DF
    except:
        print("DATAFRAME_CONVERT_ERROR\n")
    
def To_NP_Array(Data_Set):
    """ Function created to convert Pandas DataFrame to NumPy array/matrix 
    INPUT: Pandas DataFrame containing data """
    try:
        Data_Set_NP = Data_Set.to_numpy(copy = True)
        return Data_Set_NP
    except:
        print("NP_CONVERT_ERROR\n")

def Data_Stats(Data, QQ_DD=True, show=False):
    """ Function created to calculate and show/save some basic statistics and correlation about the dataser
    INPUT: DataFrame dataset, flag for quartiles or deciles (default True=quartiles) and flag for printing information to screen
            (default False) """
    try:
        Data_Set = pd.DataFrame(Data, index = (list(range(0,Data.shape[0]))), columns = (list(range(0,Data.shape[1]))))
        if(QQ_DD == True):           
            quantiles = [0.00, 0.25, 0.50, 0.75] #Calculating quartiles
        else:
            quantiles = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00] #Calculating quartiles

        Describ = Data_Set.describe(percentiles = quantiles) #Data set general stats description
        Correlation = Data_Set.corr('spearman') #Computing pairwise feature correlation through Spearman rank correlation
        name = ("GeneralStats.xlsx")
        with pd.ExcelWriter(name) as writer: #Outputting Excel file with statistics
            Describ.to_excel(writer, sheet_name='Data_Description')
            Correlation.to_excel(writer, sheet_name='Column_Correlation')
        if(show == True):
            print(Data_Set)
            print(Describ) #Printing statistics to screen
            print(Correlation) #Printing statistics to screen
    except:
        print("STATS_FUNCTION_ERROR\n")

##################################################################################################################
'''                                       PLOTTING/GRAPHICS FUNCTIONS                                          '''
##################################################################################################################

def Plot_Multi_Curves(Data, Xlabel="X Axis", Ylabel="Y Axis", Title="My Plot", Xlim=True, Xlim1=0, Xlim2=100,
                      Ylim=True, Ylim1=0, Ylim2=100, save=False):
    """ Function created to plot different curves to represent the impact of changing the different parameters such as learning rate
        in the evolution of the functions (e.g. loss function)
    INPUT: DataFrame structure with first column being the constant parameter, label for x axis, label for y axis, title, flag for setting
            custom limit for x axis (default True), lower x limit, upper x limit, flag for setting custom limit for y axis (default True),
            lower y limit, upper y limit and flag to save the figure (default False) """
    try:
        columns  = list(Data.columns.values)
        colors = ['green','blue','yellow','magenta','black','cyan','red']
        
        fig, ax = plt.subplots(figsize = [15,10])
        ax.grid(True)
        for i in range(len(columns)-1):
            plt.plot(columns[0], columns[i+1], data=Data, marker='', markerfacecolor=None, markersize=None, color=colors[i], linewidth=2)
#        plt.plot("Number Iterations", "10e-4", data=Data, marker='', markerfacecolor=None, markersize=None, color='green', linewidth=2)
#        plt.plot("Number Iterations", "10e-5", data=Data, marker='', markerfacecolor=None, markersize=None, color='cyan', linewidth=2)
#        plt.plot('Number Iterations', "10e-6", data=Data, marker='', markerfacecolor=None, markersize=None, color='magenta', linewidth=2)
#        plt.plot("Number Iterations", "10e-7", data=Data, marker='', markerfacecolor=None, markersize=None, color='yellow', linewidth=2)
#        plt.plot('Number Iterations', "10e-8", data=Data, marker='', markerfacecolor=None, markersize=None, color='black', linewidth=2)
        plt.suptitle(Title,fontsize = 26)
        if(Xlabel=="X Axis"):
            Xlabel=columns[0]
        plt.xlabel(Xlabel, fontsize = 22)
        plt.ylabel(Ylabel, fontsize = 22)
        if(Xlim):
            plt.xlim(Xlim1,Xlim2)
        if(Ylim):
            plt.ylim(Ylim1,Ylim2)
        plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, fontsize='xx-large')
        if(save):
            timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
            plt.savefig(timestr+Title+".png")
    except:
        print("MULTI_PLOT_ERROR\n")

def Count_Plot(Column=None, Dataset=None, Cond=None, TitleName="MyPlot", Color=None, save=False, Size=(25,20)):
    """ Function imports method to analyse the DataFrame or DataSeries distribution
    INPUT: Column containing the classes, dataset, hue to a third value (default None), title, color to be used (default Rainbow),
			flag to save or not the figure (default False) and tuple for the figure size (default (25,20)) """
    try:
        sb.set(style="whitegrid", color_codes=True)
        sb.set(rc={"figure.figsize":Size})
        if(Column!=None):
            ax = sb.countplot(x=Column,data=Dataset, hue=Cond, color=Color) #Create a countplot and define hue if Cond!=None
        else:
            ax = sb.countplot(Dataset, hue=Cond, color=Color)
        ax.set_title(TitleName, fontsize = 32)
        plt.show()
        if(save==True):
            timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
            fig = ax.get_figure()
            fig.savefig(timestr+TitleName+".png")
    except:
        print("COUNTPLOT_GENERATION_ERROR\n")

def ClassReport_Graph(Classif, Data_train, Target_train, Data_test, Target_test, Class, ModelName='Classifier', Accur=False, Predict=None):
    """ Function imports method to report and analyse predictions from different scikit-learn model implementations
    INPUT: training examples' features, training examples' outputs, testing examples' features, testing examples' outputs
            and list with the names of the classes """
    try:
        from yellowbrick.classifier import ClassificationReport
        
        if(Accur==True):
            print((ModelName+" accuracy: %0.4f")%(metrics.accuracy_score(Target_test, Predict, normalize=True)))
        
        view_graph = ClassificationReport(Classif, classes=Class, size=(900, 720)) #Object for classification model and visualization
        view_graph.fit(Data_train, Target_train) # Fit the training data to the visualizer
        view_graph.score(Data_test, Target_test) # Evaluate the model on the test data
        graph = view_graph.poof() # Draw/show/poof the data
        return graph
    except:
        print("CLASSIFICATION-REPORT_ERROR\n")

def Learn_Perform(DataF, LabelX='Classifiers', LabelY1='Accuracy', LabelY2='Run Time', TitleName="Resulting Scores", Size=(16,12), save=False):        
    """ Function designed to plot the performance of the best results and parameters for the different learning model fed to
		to the Pipeline in the GridSearch function 
    INPUT: DataFrame containing the results (accuracies and times), name of column containing learner names, name of accuracy axis,
		name for the running time axis, title for the plot, size of the plot and flag for saving or not (default False)
			and flag to save or not the figure (default False)"""
    try:
        DF = pd.melt(DataF, id_vars=LabelX, var_name='Variables', value_name='value_numbers')
        fig, ax1 = plt.subplots(figsize=Size)
        graph = sb.barplot(x=LabelX, y='value_numbers', hue='Variables', data=DF, ax=ax1)
        ax2 = ax1.twinx()
        ax1.set_title(TitleName, fontsize = 24)
        ax1.set_xlabel(LabelX, fontsize=18)
        ax1.set_ylabel((LabelY1+" (%)"), fontsize=18)
        ax1.set_ylim(0.0,100.0)
        ax2.set_ylabel((LabelY2+" (s)"), fontsize=18)
        ax2.set_ylim(0,DataF[LabelY2].max())
        plt.setp(ax1.get_legend().get_texts(), fontsize='16') # for legend text
        plt.show()
        if(save==True):
            timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
            graph = ax1.get_figure()
            graph.savefig(timestr+TitleName+".png")
    except:
        print("LEARNER-PERFORM_ERROR\n")

def Get_ConfusionMatrix(TrueLabels, PredictedLabels, Classes, Normal=False, Title='Confusion matrix', ColorMap='rainbow',
                        FigSize=(30,30), save=False):
    """ Function designed to plot the confusion matrix of the predicted labels versus the true leabels 
    INPUT: vector containing the actual true labels, vector containing the predicted labels, flag for normalizing the data (default False),
            name of the title for the graph, color map (default winter) and flag to save or not the figure (default False).
    OUTPUT: function returns a matrix containing the confusion matrix values """
#   Colormap reference -> https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    try:   
        ConfMatrix = metrics.confusion_matrix(TrueLabels, PredictedLabels) #Calculating confusion matrix
    
        if(Normal==True):
            ConfMatrix = ConfMatrix.astype('float') / ConfMatrix.sum(axis=1)[:, np.newaxis]

        ConfMatrix_DF = pd.DataFrame(data=ConfMatrix, index=Classes, columns=Classes)                     
        fig, ax = plt.subplots(figsize=FigSize)
        sb.heatmap(ConfMatrix_DF, annot=True, cmap=ColorMap)
        ax.set_title(Title, fontsize=26)
        ax.set_xlabel('Predicted labels', fontsize = 20)
        ax.set_ylabel('True labels', fontsize = 20)
        ax.set_ylim(len(ConfMatrix)+0.25, -0.25)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()
    
        if(save==True):
            timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
            fig = ax.get_figure()
            fig.savefig(timestr+Title+".png")
    
        return ConfMatrix_DF
    except:
       print("CONFUSION-MATRIX_ERROR\n") 

def View_Image(Matrix, Is_NumPy=False, Is_DF=False, Multiple=True):
    """ Function designed to plot figures from the data set 
    INPUT: structure containing the images/matrices, flag if the structure is numpy (default False), flag if the strucure is pandas DF or
            tensor (pandas=True or tensor=False) and flag if there are multiple images to be shown (default True) """
    try:       
        if(Is_NumPy!=True):
            if(Is_DF):
                Matrix = Matrix.to_numpy(copy=True)
            elif(Is_DF==False):
                Matrix = Matrix.numpy()
        if(Multiple):
            for i in range(Matrix.shape[0]):
                plt.figure()
                plt.imshow(Matrix[i].squeeze(), cmap='gray_r')
        else:
            plt.imshow(Matrix.squeeze(), cmap='gray_r')
    except:
       print("PRINT-IMAGE_ERROR\n") 

def OneHotEncoder(Labels, Array=True):
    """ Function designed to perform preprocessing of dataset labels and perform one hot encoding of the label 
    INPUT: structure containing the labels, flag if returning NumPy array structure (True) or list (False) - (default True)
    OUTPUT: function returns a matrix containing the confusion matrix values """
    try:
        # Defining the matrix of one-hots for the labels from 0 to 9           
        MatrixPrior = [[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], #0
                       [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], #1
                       [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], #2
                       [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0], #3
                       [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], #4
                       [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], #5
                       [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0], #6
                       [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0], #7
                       [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0], #8
                       [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],] #9
        
        # Encoding the inputted labels in a list of lists
        Encoded = [MatrixPrior[index] for index in Labels]
        if(Array): # Converting to NumPy array if flag true
            Encoded = np.array(Encoded)
        return Encoded
    except:
       print("ONE-HOT-ENCODER_ERROR\n") 

def Image_Thresholding(Matrix, threshold_px=0):
    """ Function designed to perform the image thresholding as described in the report 
    INPUT: images to be applied the thresholding and the threshold value T
    OUTPUT: treated images """
    try:
        Matrix = (Matrix>=threshold_px) #Retaining only pixels that are greater or equal to the given value
        Matrix = Matrix.astype(int) #Converting back to integers
        return Matrix
    except:
        print("IMAGE-THRESHOLDING_ERROR\n")
        
def Image_Normalization(Matrix):
    """ Function designed to perform the image normalization as described in the report 
    INPUT: images to be applied the normalization
    OUTPUT: images with normalized pixels """
    try:
        Matrix = Matrix/np.max(Matrix) #Retaining only pixels that are black    
        return Matrix
    except:
        print("IMAGE-NORM_ERROR\n")
    
def Down_Resolution(Matrix, Multiple=True, Number_Times=1):
    """ Function designed to perform the decrease in the image resolution by averaging the pixel values 
    INPUT: images to be applied the downsize resolution, flag if there are multiple images and the number of times to downsize
    OUTPUT: treated images """
    try:
        if(Multiple==False):
            for t in range(Number_Times):
                Temp1 = []
                for i in range(0,Matrix.shape[0],2):
                    Temp2 = []
                    for j in range(0,Matrix.shape[1],2):
                        Temp2.append((Matrix[i][j]+Matrix[i][j+1]+Matrix[i+1][j]+Matrix[i+1][j+1])/4)
                    Temp1.append(np.array(Temp2))
                Matrix = np.array(Temp1)
        else:
            for t in range(Number_Times):
                Temp1 = []
                for k in range(Matrix.shape[0]):
                    Temp2 = []
                    for i in range(0,Matrix.shape[1],2):
                        Temp3 = []
                        for j in range(0,Matrix.shape[2],2):
                            Temp3.append((Matrix[k][i][j]+Matrix[k][i][j+1]+Matrix[k][i+1][j]+Matrix[k][i+1][j+1])/4)
                        Temp2.append(np.array(Temp3))
                    Temp1.append(np.array(Temp2))
                Matrix = np.array(Temp1)
        return Matrix
    except:
        print("DOWN-RESOLUTION_ERROR\n")
    

##################################################################################################################################
'''                                    CONVOLUTIONAL NEURAL NETWORK CLASSES DESIGNED                                          '''
##################################################################################################################################
       
class ConvNN_G23_Full(nn.Module):
    """ The class constructor builds a neural network with convolutional layers based on the VGG Net architecture, a CNN model with simple
        architecture and high accuracy. This proposed class uses the total number of convolutional layers, 13, but increases the number of
        fully connected layers, from 3 to 5. The kernel size of 3x3 convolutions was maintained, using a stride of 1 and a unit (1)
        zero-padding in the borders. Rectified Linear Unit (ReLU) was used as activation function and the first 4 out of 5 fully connected
        layers have dropout introduced to exclude a given % of the activation units
        INPUT PARAMETERS: when instantiating the class, the user should provide six different parameters, being:
                        num_classes = the number of different labels in the classificaiton
                        input_ratio = the image size (e.g. 128) divided by the 8 (referent to three max poolings of 2x2). Default 16
                        soft_max = boolean falg to apply or not soft max funciton in the end of the model. Default False. If True, 
                                    the torch.max must be changed in the evaluation function.
                        drop_out = boolean flag to indicate the use or not of drop out regularization. Default True
                        drop_proba = the probability of dropping out hidden units if drop_out=True. Default 0.5
                        FC5_relu = boolean value to flag the use or not of ReLU function after the last fully connected layer.
                                    Default True. However, ReLU is applied in all other fully connected layers """
    
    def __init__(self, num_classes=10, input_ratio=16, soft_max=False, drop_out=True, drop_prob=0.5, FC5_relu=True):
        super(ConvNN_G23_Full, self).__init__()
        
        self.num_classes = num_classes
        self.input_ratio = input_ratio
        self.soft_max = soft_max
        self.drop_out = drop_out
        self.drop_prob = drop_prob
        self.FC5_relu = FC5_relu
        
        # Max pooling of 2x2 filter and stride of 2 - this max pooling will be used in all layers
        self.Pooling = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        #Using Rectified Linear Unit activation function in all convolutional layers of the neural network
        self.relu = nn.ReLU()
        
        #Using Soft max after the last fully connected layer
        self.softmax = nn.Softmax(dim=-1)
        
        #First "super" layer: composed of two 2D convolutional layers with 128 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size   
        Conv1_1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=(3,3), stride=(1,1), padding=1)
        bn1_1 = nn.BatchNorm2d(32)
        Conv1_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=(3,3), stride=(1,1), padding=1)
        bn1_2 = nn.BatchNorm2d(32)
        self.SuperLayer1 = nn.Sequential(
            Conv1_1,
            self.relu,
            bn1_1,
            Conv1_2,
            self.relu,
            bn1_2,
            self.Pooling)
        
        #Second "super" layer: composed of two 2D convolutional layers with 256 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size 
        Conv2_1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3,3), stride=(1,1), padding=1)
        bn2_1 = nn.BatchNorm2d(64)
        Conv2_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3,3), stride=(1,1), padding=1)
        bn2_2 = nn.BatchNorm2d(64)
        self.SuperLayer2 = nn.Sequential(
            Conv2_1,
            self.relu,
            bn2_1,
            Conv2_2,
            self.relu,
            bn2_2,
            self.Pooling)
        
        #Third "super" layer: composed of three 2D convolutional layers with 512 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size 
        Conv3_1 = nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size=(3,3), stride=(1,1), padding=1)
        bn3_1 = nn.BatchNorm2d(256)
        Conv3_2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3,3), stride=(1,1), padding=1)
        bn3_2 = nn.BatchNorm2d(256)
        Conv3_3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3,3), stride=(1,1), padding=1)
        bn3_3 = nn.BatchNorm2d(256)
        self.SuperLayer3 = nn.Sequential(
            Conv3_1,
            self.relu,
            bn3_1,
            Conv3_2,
            self.relu,
            bn3_2,
            Conv3_3,
            self.relu,
            bn3_3,
            self.Pooling)
    
        #Fourth "super" layer: composed of three 2D convolutional layers with 1024 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size; No max pooling in the end
        Conv4_1 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=(3,3), stride=(1,1), padding=1)
        bn4_1 = nn.BatchNorm2d(512)
        Conv4_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,3), stride=(1,1), padding=1)
        bn4_2 = nn.BatchNorm2d(512)
        Conv4_3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,3), stride=(1,1), padding=1)
        bn4_3 = nn.BatchNorm2d(512)
        self.SuperLayer4 = nn.Sequential(
            Conv4_1,
            self.relu,
            bn4_1,
            Conv4_2,
            self.relu,
            bn4_2,
            Conv4_3,
            self.relu,
            bn4_3,
            self.Pooling)
        
        #Fifth "super" layer: composed of three 2D convolutional layers with 1024 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size; No max pooling in the end
        Conv5_1 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,3), stride=(1,1), padding=1)
        bn5_1 = nn.BatchNorm2d(512)
        Conv5_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,3), stride=(1,1), padding=1)
        bn5_2 = nn.BatchNorm2d(512)
        Conv5_3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,3), stride=(1,1), padding=1)
        bn5_3 = nn.BatchNorm2d(512)
        self.SuperLayer5 = nn.Sequential(
            Conv5_1,
            self.relu,
            bn5_1,
            Conv5_2,
            self.relu,
            bn5_2,
            Conv5_3,
            self.relu,
            bn5_3,
            self.Pooling)
        
        #Fully connected (FC) layers: first FC of size input_ratio^2 times the out_channels size of the last convolutional layer;
        #second FC of size 4096 and out 1024; third FC input 1024 and out num_classes; Batch normalization in the first two FCs  
        self.FC1 = nn.Linear(512*self.input_ratio*self.input_ratio, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
    
        self.FC2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
    
        self.FC3 = nn.Linear(256,64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        
        self.FC4 = nn.Linear(64,16)
        self.bn_fc4 = nn.BatchNorm1d(16)
        
        self.FC5 = nn.Linear(16,self.num_classes)


    def forward(self,x):
        
        out = self.SuperLayer1(x)
        out = self.SuperLayer2(out) 
        out = self.SuperLayer3(out)
        out = self.SuperLayer4(out)
        out = self.SuperLayer5(out)

        out = out.view(-1,512*self.input_ratio*self.input_ratio)
        
        #Entering fully connected layers
        out = self.relu(self.FC1(out))
        out = self.bn_fc1(out)
        
        #Performing dropout to avoid overfitting
        if(self.drop_out==True):   
            out = nf.dropout(out, p=self.drop_prob)

        out = self.relu(self.FC2(out))
        out = self.bn_fc2(out)
        
        if(self.drop_out==True):
            out = nf.dropout(out, p=self.drop_prob)
            
        out = self.relu(self.FC3(out))
        out = self.bn_fc3(out)
        
        if(self.drop_out==True):
            out = nf.dropout(out, p=self.drop_prob)
            
        out = self.relu(self.FC4(out))
        out = self.bn_fc4(out)
        
        if(self.drop_out==True):
            out = nf.dropout(out, p=self.drop_prob)

        if(self.FC5_relu==True):
            out = self.relu(self.FC5(out))
        else:
            out = self.FC5(out)
            
        if(self.soft_max==True):
            out = self.softmax(out)
        
        return out      

##############################################################################

class ConvNN_G23_Std(nn.Module):
    """ The class constructor builds a neural network with convolutional layers based on the architecture of ConvNN_G23_Full, but with
        less layers to enable faster tests. This proposed class modifies the total number of layers, decreaseing from 13 to 10 convolutional
        layers. The kernel size of 3x3 convolutions was maintained, using a stride of 1 and a unit (1) zero-padding in the borders.
        Rectified Linear Unit (ReLU) was used as activation function and the first 3 out of 4 fully connected layers have dropout option
        introduced to exclude a given % of the activation units
        INPUT PARAMETERS: when instantiating the class, the user should provide six different parameters, being:
                        num_classes = the number of different labels in the classificaiton
                        input_ratio = the image size (e.g. 128) divided by the 8 (referent to three max poolings of 2x2). Default 16
                        soft_max = boolean falg to apply or not soft max funciton in the end of the model. Default False. If True, 
                                    the torch.max must be changed in the evaluation function.
                        drop_out = boolean flag to indicate the use or not of drop out regularization. Default True
                        drop_proba = the probability of dropping out hidden units if drop_out=True. Default 0.5
                        FC4_relu = boolean value to flag the use or not of ReLU function after the last fully connected layer.
                                    Default True. However, ReLU is applied in all other fully connected layers """
    
    def __init__(self, num_classes=10, input_ratio=16, soft_max=False, drop_out=True, drop_prob=0.5, FC4_relu=True):
        super(ConvNN_G23_Std, self).__init__()
        
        self.num_classes = num_classes
        self.input_ratio = input_ratio
        self.soft_max = soft_max
        self.drop_out = drop_out
        self.drop_prob = drop_prob
        self.FC4_relu = FC4_relu
        
        # Max pooling of 2x2 filter and stride of 2 - this max pooling will be used in all layers
        self.Pooling = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        #Using Rectified Linear Unit activation function in all convolutional layers of the neural network
        self.relu = nn.ReLU()
        
        #Using Soft max after the last fully connected layer
        self.softmax = nn.Softmax(dim=-1)
        
        #First "super" layer: composed of two 2D convolutional layers with 128 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size   
        Conv1_1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=(3,3), stride=(1,1), padding=1)
        bn1_1 = nn.BatchNorm2d(32)
        Conv1_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=(3,3), stride=(1,1), padding=1)
        bn1_2 = nn.BatchNorm2d(32)
        self.SuperLayer1 = nn.Sequential(
            Conv1_1,
            self.relu,
            bn1_1,
            Conv1_2,
            self.relu,
            bn1_2,
            self.Pooling)
        
        #Second "super" layer: composed of two 2D convolutional layers with 256 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size 
        Conv2_1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3,3), stride=(1,1), padding=1)
        bn2_1 = nn.BatchNorm2d(64)
        Conv2_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3,3), stride=(1,1), padding=1)
        bn2_2 = nn.BatchNorm2d(64)
        self.SuperLayer2 = nn.Sequential(
            Conv2_1,
            self.relu,
            bn2_1,
            Conv2_2,
            self.relu,
            bn2_2,
            self.Pooling)
        
        #Third "super" layer: composed of three 2D convolutional layers with 512 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size 
        Conv3_1 = nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size=(3,3), stride=(1,1), padding=1)
        bn3_1 = nn.BatchNorm2d(256)
        Conv3_2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3,3), stride=(1,1), padding=1)
        bn3_2 = nn.BatchNorm2d(256)
        Conv3_3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3,3), stride=(1,1), padding=1)
        bn3_3 = nn.BatchNorm2d(256)
        self.SuperLayer3 = nn.Sequential(
            Conv3_1,
            self.relu,
            bn3_1,
            Conv3_2,
            self.relu,
            bn3_2,
            Conv3_3,
            self.relu,
            bn3_3,
            self.Pooling)
    
        #Fourth "super" layer: composed of three 2D convolutional layers with 1024 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size; No max pooling in the end
        Conv4_1 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=(3,3), stride=(1,1), padding=1)
        bn4_1 = nn.BatchNorm2d(512)
        Conv4_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,3), stride=(1,1), padding=1)
        bn4_2 = nn.BatchNorm2d(512)
        Conv4_3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,3), stride=(1,1), padding=1)
        bn4_3 = nn.BatchNorm2d(512)
        self.SuperLayer4 = nn.Sequential(
            Conv4_1,
            self.relu,
            bn4_1,
            Conv4_2,
            self.relu,
            bn4_2,
            Conv4_3,
            self.relu,
            bn4_3,
            self.Pooling)
        
        #Fully connected (FC) layers: first FC of size input_ratio^2 times the out_channels size of the last convolutional layer;
        #second FC of size 4096 and out 1024; third FC input 1024 and out num_classes; Batch normalization in the first two FCs  
        self.FC1 = nn.Linear(512*self.input_ratio*self.input_ratio, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
    
        self.FC2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
    
        self.FC3 = nn.Linear(256,16)
        self.bn_fc3 = nn.BatchNorm1d(16)
        
        self.FC4 = nn.Linear(16,self.num_classes)


    def forward(self,x):
        
        out = self.SuperLayer1(x)
        out = self.SuperLayer2(out) 
        out = self.SuperLayer3(out)
        out = self.SuperLayer4(out)

        out = out.view(-1,512*self.input_ratio*self.input_ratio)
        
        #Entering fully connected layers
        out = self.relu(self.FC1(out))
        out = self.bn_fc1(out)
        
        #Performing dropout to avoid overfitting
        if(self.drop_out==True):   
            out = nf.dropout(out, p=self.drop_prob)

        out = self.relu(self.FC2(out))
        out = self.bn_fc2(out)
        
        if(self.drop_out==True):
            out = nf.dropout(out, p=self.drop_prob)
            
        out = self.relu(self.FC3(out))
        out = self.bn_fc3(out)
        
        if(self.drop_out==True):
            out = nf.dropout(out, p=self.drop_prob)
            
        if(self.FC4_relu==True):
            out = self.relu(self.FC4(out))
        else:
            out = self.FC4(out)
            
        if(self.soft_max==True):
            out = self.softmax(out)
        
        return out

##############################################################################    
    
class ConvNN_G23_Mini(nn.Module):
    """ The class constructor builds a neural network with convolutional layers based on the architecture of ConvNN_G23_Std, but with
        less layers to enable faster tests. This proposed class modifies the total number of layers, decreaseing from 10 to 6 convolutional
        layers. The kernel size of 3x3 convolutions was maintained, using a stride of 1 and a unit (1) zero-padding in the borders.
        Rectified Linear Unit (ReLU) was used as activation function and the first 2 out of 3 fully connected layers have dropout option
        introduced to exclude a given % of the activation units
        INPUT PARAMETERS: when instantiating the class, the user should provide six different parameters, being:
                        num_classes = the number of different labels in the classificaiton
                        input_ratio = the image size (e.g. 128) divided by the 8 (referent to three max poolings of 2x2). Default 16
                        soft_max = boolean falg to apply or not soft max funciton in the end of the model. Default False. If True, 
                                    the torch.max must be changed in the evaluation function.
                        drop_out = boolean flag to indicate the use or not of drop out regularization. Default True
                        drop_proba = the probability of dropping out hidden units if drop_out=True. Default 0.5
                        FC3_relu = boolean value to flag the use or not of ReLU function after the last fully connected layer.
                                    Default True. However, ReLU is applied in all other fully connected layers """
    
    def __init__(self, num_classes=10, input_ratio=16, soft_max=False, drop_out=True, drop_prob=0.5, FC3_relu=True):
        super(ConvNN_G23_Mini, self).__init__()
        
        self.num_classes = num_classes
        self.input_ratio = input_ratio
        self.soft_max = soft_max
        self.drop_out = drop_out
        self.drop_prob = drop_prob
        self.FC3_relu = FC3_relu
        
        # Max pooling of 2x2 filter and stride of 2 - this max pooling will be used in all layers
        self.Pooling = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        #Using Rectified Linear Unit activation function in all convolutional layers of the neural network
        self.relu = nn.ReLU()
        
        #Using Soft max after the last fully connected layer
        self.softmax = nn.Softmax(dim=-1)
        
        #First "super" layer: composed of two 2D convolutional layers with 128 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size   
        Conv1_1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=(3,3), stride=(1,1), padding=1)
        bn1_1 = nn.BatchNorm2d(64)
        Conv1_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3,3), stride=(1,1), padding=1)
        bn1_2 = nn.BatchNorm2d(64)
        self.SuperLayer1 = nn.Sequential(
            Conv1_1,
            self.relu,
            bn1_1,
            Conv1_2,
            self.relu,
            bn1_2,
            self.Pooling)
        
        #Second "super" layer: composed of two 2D convolutional layers with 256 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size 
        Conv2_1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=(3,3), stride=(1,1), padding=1)
        bn2_1 = nn.BatchNorm2d(128)
        Conv2_2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=(3,3), stride=(1,1), padding=1)
        bn2_2 = nn.BatchNorm2d(128)
        self.SuperLayer2 = nn.Sequential(
            Conv2_1,
            self.relu,
            bn2_1,
            Conv2_2,
            self.relu,
            bn2_2,
            self.Pooling)
        
        #Third "super" layer: composed of three 2D convolutional layers with 512 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size 
        Conv3_1 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=(3,3), stride=(1,1), padding=1)
        bn3_1 = nn.BatchNorm2d(256)
        Conv3_2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3,3), stride=(1,1), padding=1)
        bn3_2 = nn.BatchNorm2d(256)
        self.SuperLayer3 = nn.Sequential(
            Conv3_1,
            self.relu,
            bn3_1,
            Conv3_2,
            self.relu,
            bn3_2,
            self.Pooling)
    
        #Fully connected (FC) layers: first FC of size input_ratio^2 times the out_channels size of the last convolutional layer;
        #second FC of size 4096 and out 1024; third FC input 1024 and out num_classes; Batch normalization in the first two FCs  
        self.FC1 = nn.Linear(256*self.input_ratio*self.input_ratio, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
    
        self.FC2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
    
        self.FC3 = nn.Linear(128,self.num_classes)

    def forward(self,x):
        
        out = self.SuperLayer1(x)
        out = self.SuperLayer2(out)
        out = self.SuperLayer3(out)
        
        out = out.view(-1,256*self.input_ratio*self.input_ratio)
        
        #Entering fully connected layers
        out = self.relu(self.FC1(out))
        out = self.bn_fc1(out)
              
        #Performing dropout to avoid overfitting
        if(self.drop_out==True):   
            out = nf.dropout(out, p=self.drop_prob)

        out = self.relu(self.FC2(out))
        out = self.bn_fc2(out)
        
        if(self.drop_out==True):
            out = nf.dropout(out, p=self.drop_prob)

        if(self.FC3_relu==True):
            out = self.relu(self.FC3(out))
        else:
            out = self.FC3(out)
        
        if(self.soft_max==True):
            out = self.softmax(out)
        
        return out 

##############################################################################
    
class ConvNN_G23_Micro(nn.Module):
    """ The class constructor builds a neural network with convolutional layers based on the architecture of ConvNN_G23_Mini, but with
        less layers to enable faster tests. This proposed class modifies the total number of layers, decreaseing from 6 to 3 convolutional
        layers. The kernel size of 3x3 convolutions was maintained, using a stride of 1 and a unit (1) zero-padding in the borders.
        Rectified Linear Unit (ReLU) was used as activation function and the first 2 out of 3 fully connected layers have dropout option
        introduced to exclude a given % of the activation units
        INPUT PARAMETERS: when instantiating the class, the user should provide six different parameters, being:
                        num_classes = the number of different labels in the classificaiton
                        input_ratio = the image size (e.g. 128) divided by the 8 (referent to three max poolings of 2x2). Default 16
                        soft_max = boolean falg to apply or not soft max funciton in the end of the model. Default False. If True, 
                                    the torch.max must be changed in the evaluation function.
                        drop_out = boolean flag to indicate the use or not of drop out regularization. Default True
                        drop_proba = the probability of dropping out hidden units if drop_out=True. Default 0.5
                        FC3_relu = boolean value to flag the use or not of ReLU function after the last fully connected layer.
                                    Default True. However, ReLU is applied in all other fully connected layers """
    
    def __init__(self, num_classes=10, input_ratio=16, soft_max=False, drop_out=True, drop_prob=0.5, FC3_relu=True):
        super(ConvNN_G23_Micro, self).__init__()
        
        self.num_classes = num_classes
        self.input_ratio = input_ratio
        self.soft_max = soft_max
        self.drop_out = drop_out
        self.drop_prob = drop_prob
        self.FC3_relu = FC3_relu
        
        # Max pooling of 2x2 filter and stride of 2 - this max pooling will be used in all layers
        self.Pooling = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        #Using Rectified Linear Unit activation function in all convolutional layers of the neural network
        self.relu = nn.ReLU()
        
        #Using Soft max after the last fully connected layer
        self.softmax = nn.Softmax(dim=-1)
        
        #First "super" layer: composed of two 2D convolutional layers with 128 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size   
        Conv1_1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=(3,3), stride=(1,1), padding=1)
        bn1_1 = nn.BatchNorm2d(64)
        self.SuperLayer1 = nn.Sequential(
            Conv1_1,
            self.relu,
            bn1_1,
            self.Pooling)
        
        #Second "super" layer: composed of two 2D convolutional layers with 256 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size 
        Conv2_1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=(3,3), stride=(1,1), padding=1)
        bn2_1 = nn.BatchNorm2d(128)
        self.SuperLayer2 = nn.Sequential(
            Conv2_1,
            self.relu,
            bn2_1,
            self.Pooling)
        
        #Third "super" layer: composed of three 2D convolutional layers with 512 filters, 3x3 kernels, 1 stride and 1 zero-padding;
        #each one followed by non-linear activation ReLU and batch normalization of kernel size 
        Conv3_1 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=(3,3), stride=(1,1), padding=1)
        bn3_1 = nn.BatchNorm2d(256)
        self.SuperLayer3 = nn.Sequential(
            Conv3_1,
            self.relu,
            bn3_1,
            self.Pooling)
    
        #Fully connected (FC) layers: first FC of size input_ratio^2 times the out_channels size of the last convolutional layer;
        #second FC of size 4096 and out 1024; third FC input 1024 and out num_classes; Batch normalization in the first two FCs  
        self.FC1 = nn.Linear(256*self.input_ratio*self.input_ratio, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
    
        self.FC2 = nn.Linear(256, 32)
        self.bn_fc2 = nn.BatchNorm1d(32)
    
        self.FC3 = nn.Linear(32,self.num_classes)

    def forward(self,x):
        
        out = self.SuperLayer1(x)
        out = self.SuperLayer2(out)
        out = self.SuperLayer3(out)
        
        out = out.view(-1,256*self.input_ratio*self.input_ratio)
        
        #Entering fully connected layers
        out = self.relu(self.FC1(out))
        out = self.bn_fc1(out)
        
        if(self.drop_out==True): #Performing dropout to avoid overfitting
            out = nf.dropout(out, p=self.drop_prob)

        out = self.relu(self.FC2(out))
        out = self.bn_fc2(out)
        
        if(self.drop_out==True): #Performing dropout to avoid overfitting
            out = nf.dropout(out, p=self.drop_prob)

        if(self.FC3_relu==True):
            out = self.relu(self.FC3(out))
        else:
            out = self.FC3(out)
        
        if(self.soft_max==True):
            out = self.softmax(out)
        
        return out  

    
class FFNN_G23(nn.Module):
    """ The class constructor builds a neural network with feed-forward layers """
    
    def __init__(self, num_classes=10, input_ratio=16, soft_max=False, drop_out=True, drop_prob=0.5, final_relu=True):
        super(FFNN_G23, self).__init__()  
    
        self.num_classes = num_classes
        self.input_ratio = input_ratio
        self.soft_max = soft_max
        self.drop_out = drop_out
        self.drop_prob = drop_prob
        self.final_relu = final_relu
        
        self.ffnn1 = nn.Sequential(nn.Linear(self.input_ratio*self.input_ratio, 1024),
                                  nn.ReLU(),
                                  nn.Linear(1024, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 128),
                                  nn.ReLU())
        
        self.ffnn2 = nn.Sequential(nn.Linear(128, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, self.num_classes))
    
    def forward(self,x):
        
        out = self.ffnn1(x)
        
        if(self.drop_out==True):
            out = nf.dropout(out, p=self.drop_prob)
        
        out = self.ffnn2(out)
        
        if(self.final_relu==True):
            out = nn.ReLU(out)
            
        if(self.soft_max==True):
            out = self.softmax(out)
        
        return out 
            
    
##################################################################################################################################
'''                                         TRAINING AND VALIDATION LOSS FUNCTIONS                                             '''
##################################################################################################################################
    
def TrainAccur(labels_real, predicted_out, UseGPU=True):
    """ Function designed to calculated the BOOLEAN comparison between true and predicted labels during the training process
        to assess the evolution of of training in each epoch
    INPUT: function requires a tensor contraining the true labels and another containing the predicted labels.
    OUTPUT: function returns the accuracy of the given set """
    
    proba, labels_predicted = th.max(predicted_out, 1)
    if(UseGPU):
        labels_predicted = labels_predicted.data.cpu().numpy() #Moving variable back to CPU and converting to NumPy if GPU flag True
    else:
        labels_predicted = labels_predicted.numpy() #Else, only converting to NumPy
    
    if(UseGPU):
        labels_real = labels_real.data.cpu().numpy() #Moving variable back to CPU and converting to NumPy if GPU flag True
    else:
        labels_real = labels_real.numpy() #Else, only converting to NumPy
    
    length = len(labels_real)
    
    compared = [1 if labels_predicted[i]==labels_real[i] else 0 for i in range(length)]
    compared = np.array(compared)
    
    return compared
     
   
def LossInTraining(NN, TrainingLoader, Criterion, Optimizer, TrainLength, BatchSize, Epoch, is_CNN=True, ImageSize=128, UseGPU=True,
                   PrintPartialLoss=True, PartialBatch=1000, log_file=None):
    """ Function designed to calculated the loss regardng the cost function selected (criterion) during the training
    INPUT: function requires the neural net model, the loss function, the optimization function, the length of the training set,
            batch size, the variable countig the current training epoch, flag to move or not the variables to GPU device (default True),
            flag for printing the current partial loss after a number os batchs trained (default True)
            and number after which to print current loss in current epoch.
    OUTPUT: function returns the loss of 1 training epoch per number of batchs trained """
    
    partial_running_loss = 0.0 #Partial loss for mini batches
    running_train_loss = 0.0 #Total accumulated training loss
    partial_counter = 0
    T_accur = []
    
    print(80*'=')
    print('TRAINING [EPOCH %d]'%(Epoch+1))
    
    if(log_file!=None):
        log_file.append(80*'=')
        log_file.append('TRAINING [EPOCH %d]'%(Epoch+1))
    
    for i, data in enumerate(TrainingLoader, 0):
        inputs, labels = data
        if(is_CNN==True):
            inputs = inputs.float()
            inputs.unsqueeze_(1) #Unsqueezing inputs to add "1D" between batch size and matrix dimensions
        else:
            inputs = inputs.view(-1,ImageSize*ImageSize)
        labels = labels.long()
        if(UseGPU):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) #Transforming in torch variables and moving inputs and labels to GPU
        
        Optimizer.zero_grad() #Restarting the gradients

        train_outputs = NN(inputs) #NN output
        
        T_accur.append(TrainAccur(labels_real=labels, predicted_out=train_outputs, UseGPU=UseGPU))

        loss = Criterion(train_outputs, labels) #Loss from the outputs and the true labels
        loss.backward() #Getting gradients from backpropagation
        Optimizer.step() #Increasing optimizer step

        running_train_loss += loss.item() #Increasing the total training loss
        partial_running_loss += loss.item() #Increase partial loss for the current batch size the print
        
        partial_counter += BatchSize

        if(PrintPartialLoss==True):
            if (partial_counter % PartialBatch == 0):    #Printing the loss after every set number of mini-batches  
                print('Epoch %d - batch size %d - Training loss: %.6f'%(Epoch+1, partial_counter, (partial_running_loss / PartialBatch)))
                if(log_file!=None):
                    log_file.append('Epoch %d - batch size %d - Training loss: %.6f'%(Epoch+1, partial_counter, (partial_running_loss / PartialBatch)))
                partial_running_loss = 0.0 #Restarting partial training loss
        
    T_accur = np.array(T_accur).ravel()
    return (running_train_loss/(TrainLength/BatchSize)), T_accur
    
def LossInValidation(NN, ValidationLoader, Criterion, ValidLength, BatchSize, is_CNN=True, ImageSize=128, UseGPU=True):
    """ Function designed to calculated the loss regardng the cost function selected (criterion) during the validation
    INPUT: function requires the neural net model, the loss function, the length of the validation set, batch size
            and flag to move or not the variables to GPU device (default True).
    OUTPUT: function returns the loss of 1 validated epoch per number of batchs validated """
    
    running_valid_loss = 0.0 #Initializing zero loos for the batch
    V_accur = []
    
    for i, data in enumerate(ValidationLoader, 0): #looping through the validation dataset
        
        inputs, labels = data
        if(is_CNN==True):
            inputs = inputs.float()
            inputs.unsqueeze_(1) #Unsqueezing inputs to add "1D" between batch size and matrix dimensions
        else:
            inputs = inputs.view(-1,ImageSize*ImageSize)
        labels = labels.long()
        if(UseGPU):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) #Moving variables to GPU
       
        valid_output = NN(inputs) #Output from the NN created
        loss_valid = Criterion(valid_output, labels)
        
        V_accur.append(TrainAccur(labels_real=labels, predicted_out=valid_output, UseGPU=UseGPU))
        
        running_valid_loss += loss_valid.item() #Loss in validation per batch

    V_accur = np.array(V_accur).ravel()
    return (running_valid_loss/(ValidLength/BatchSize)), V_accur


def GetPredsAccur(NeuralNet, DataLoader, DatasetType='Validation', is_CNN=True, ImageSize=128, UseGPU=True, PrintAccur=True, GetLebelsPreds=True, List=False, log_file=None):
    """ Function designed to calculated the predictions and accuracy for a given dataset using the NN built
    INPUT: function requires the neural net model, DataLoader containing the data wrapup, the title for the dataset passed, flag to move
            or not the variables to GPU device (default True), flag to print or not accuracy (default True), flag to return or not the
            predicted labels and flag to determine if the returned labels are in list of NumPy array structure (default False = np.array).
    OUTPUT: function returns the predicted labels for a given dataset (if GetLebelsPreds=True) """
    
    labels_predicted = []
    labels_real = []
    
    for batch_num, data in enumerate(DataLoader, 0): #Iterating through DataLoader
        inputs, labels = data #Getting inputs and labels for the given batch
        if(is_CNN==True):
            inputs = inputs.float() #Ensuring the inputs are float type
            inputs.unsqueeze_(1) #Adding dimension between batch size and matrix dimensions
        else:
            inputs = inputs.view(-1,ImageSize*ImageSize)
        labels = labels.long() #Ensuring the inputs are long type
        if(UseGPU):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) #Moving variables to GPU if flag True

        predicted_out = NeuralNet(inputs) #Getting predictions from the NN
        proba, predicted = th.max(predicted_out, 1) #Getting maximum probabilities and the labels associated to them for each input in the batch
		
        if(UseGPU):
            predicted = predicted.data.cpu().numpy() #Moving variable back to CPU and converting to NumPy if GPU flag True
        else:
            predicted = predicted.data.numpy() #Else, only converting to NumPy
		
        labels_predicted.append(predicted)
		
        if(UseGPU):
            labels_real.append(labels.data.cpu().numpy()) #Moving label variables to CPU and converting to NumPy if GPU flag True
        else:
            labels_real.append(labels.data.numpy()) #Else, only converting to NumPy
	
    length = len(labels_real) #Getting length of predicted vector
    labels_predicted = np.array(labels_predicted) #Converting lists to NumPy
    labels_real = np.array(labels_real)
	
    labels_predicted = labels_predicted.ravel()
    labels_real = labels_real.ravel()
		
    if(PrintAccur==True):
        compared = [1 if labels_predicted[i]==labels_real[i] else 0 for i in range(length)]
        compared = np.array(compared)
        
        print("Accuracy for %s dataset (instance of %d) of %0.5f"%(DatasetType,len(labels_real),(sum(compared)/length)))
        
    if(log_file!=None):
        log_file.append("Accuracy for %s dataset (instance of %d) of %0.5f"%(DatasetType,len(labels_real),(sum(compared)/length)))
	
    if(GetLebelsPreds==True):
        if(List==False):
            return labels_predicted
        elif(List==True):
            return labels_predicted.tolist()
			
#    else:
#        print("GET-ACCURACY_ERROR!")
            
def KagglePreds(NeuralNet, DataLoader, is_CNN=True, ImageSize=128, UseGPU=True, GetLebelsPreds=True):
    """ Function designed to calculated the predictions and accuracy for a given dataset using the NN built
    INPUT: function requires the neural net model, DataLoader containing the data wrapup, the title for the dataset passed, flag to move
            or not the variables to GPU device (default True), flag to print or not accuracy (default True), flag to return or not the
            predicted labels and flag to determine if the returned labels are in list of NumPy array structure (default False = np.array).
    OUTPUT: function returns the predicted labels for a given dataset (if GetLebelsPreds=True) """
    
    labels_predicted = []
    
    for batch_num, data in enumerate(DataLoader, 0): #Iterating through DataLoader
        inputs, labels = data #Getting inputs and labels for the given batch
        if(is_CNN==True):
            inputs = inputs.float() #Ensuring the inputs are float type
            inputs.unsqueeze_(1) #Adding dimension between batch size and matrix dimensions
        else:
            inputs = inputs.view(-1,ImageSize*ImageSize)
        labels = labels.long() #Ensuring the inputs are long type
        if(UseGPU):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) #Moving variables to GPU if flag True

        predicted_out = NeuralNet(inputs) #Getting predictions from the NN
        proba, predicted = th.max(predicted_out, 1) #Getting maximum probabilities and the labels associated to them for each input in the batch
		
        if(UseGPU):
            predicted = predicted.data.cpu().numpy() #Moving variable back to CPU and converting to NumPy if GPU flag True
        else:
            predicted = predicted.data.numpy() #Else, only converting to NumPy
		
        labels_predicted.append(predicted)
	
    labels_predicted = np.array(labels_predicted) #Converting lists to NumPy
	
    labels_predicted = labels_predicted.ravel()
    labels_predicted = labels_predicted.tolist()
    ids = [i for i in range(len(labels_predicted))]
    
    predictions = pd.DataFrame({"Id":ids, "Label":labels_predicted})
        
    if(GetLebelsPreds==True):
        return predictions
			
#    else:
#        print("GET-ACCURACY_ERROR!")

##################################################################################################################################
'''                                                              END                                                           '''
##################################################################################################################################
