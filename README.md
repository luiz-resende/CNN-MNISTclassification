# Neural Network for Image Classification on Modified MNIST Dataset

### **Authors**: *Resende Silva, Luiz, Furtado e Faria, Matheus, & Kondulukov, Vitaly*
### **Date**: *November 2019*
### **Subject**: *COMP551 Applied Machine Learning - Project 03*

## **Overview**
The current project script aimed to implement a Neural Network, more specifically a Convolutional Neural Network (CNN), for the task of image classification of 60,000 images of the Modified MNIST dataset made available, from which 50,000 images composed the training set (with their respective label) and the 10,000 images composed the held-out test set (whose predictions were later submitted to the kaggle competition).

## INSTALLATION

The different functions in the code submitted were implemented using a Python 3 base IDE and having all the libraries and dependencies updated to their latest version. All functions used day-to-day Python modules such as NumPy and Pandas. The CNN implemented was entirely based on [PyTorch API](https://pytorch.org/) **with support for GPU**, thus it is required its availability since the script submitted assumes GPU aceleration.
Moreover, the Python 3 Notebook containing the CNN's training "pipeline" used functions implemented in a sepparate script, named ```Project_03_Functions.py```, to maintain good practices and cleaness of the main code. Although this script is submitted along with the main code, **its upload is not required**. The ***main code has a command to download this script directly from the authors kaggle account***.

## CODE FILES

The project's code delivered was divided into 2 different files to have a cleaner environment and facilitated their use. The files are:

1. *Project_03_Functions.py*: contains the functions designed to be used thoughout the code, includes dataset importing functions, plotting, the neural network classification models implemented, preprocessing functions and functions to structure the NN training pipeline. All the functions on it are duly commented and their input parameters are clearly explained (and their names are intuitive). The user does not need to run any of the scripts on it, since other files import its functions. As explained above, it **does not need to be included in the same directory as the other code file, since the main script downloads it**.

2. *Neural_Net_Group23.ipynb*: this file contains the code set-up used to assess the CNN implemented models, where the training dataset for the image classification is imported, splitted in two sets and the models are trained and tested to assess their individual performance and later make predictions for the held-out dataset, generating a ```.csv``` file to be submitted to the Kaggle competition. It **automatically imports the file described above**.

A third file is also submitted along with the two above, it being the trained model from the last run submitted to kaggle, named ```CNN_G23_Std_Model_best.pkl```.

## DATASETS UPLOAD

As for the ```Project_03_Functions.py``` file, the main code in the Python 3 notebook has commands to ***automatically download the dataset from the [Kaggle competition](https://www.kaggle.com/c/modified-mnist/data) page and unzip them in the directory for the project***.

## CONVOLUTIONAL NEURAL NETWORK CLASS

The CNN class implemented and used in the last submission was the one named ```CNN_G23_Std```, which requires when of its instantiation the following parameters:
```python
class ConvNN_G23_Std(nn.Module):
    """ The class constructor builds a neural network with convolutional layers based on the architecture of ConvNN_G23_Full, but with
    less layers to enable faster tests with extremelly high accuracy. This proposed class modifies the total number of layers, 
    wth 10 convolutional layers. The kernel size of 3x3 convolutions was maintained, using a stride of 1 and a unit (1) zero-padding in
    the borders. Rectified Linear Unit (ReLU) was used as activation function and the first 3 out of 4 fully connected layers have
    dropout option introduced to exclude a given % of the activation units and avoid over-fitting. The last fully connected layer (FC4) 
    has the option to also use ReLU or not (flag activated or not in the instantiation).
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
```
Such CNN described above has the following architecture [Class CNN_G23_Std](https://www.dropbox.com/s/a638uc9aet0fh7k/cnng23std.png?dl=0).

## USAGE AND REPRODUCTION

The instructions for the main code use are all commented in the ```Neural_Net_Group23.ipynb``` file, which also contains the configurations required to reproduce the best result submitted to the Kaggle competition.

## RESULTS

The implemented CNN, after trained, achieved an accuracy of *98.41%* on the undisclosed test set available for the Kaggle competition. 

## AUTHORS AND ACKNOWLEDGMENTS

### AUTHORSHIP
All the scripts presented in the files *Project_03_Functions.py* and *Neural_Net_Group23.ipynb* (with the exception of the imported modules) were coded/implementd by Luiz Resende Silva.

All classification and utility modules were imported from either PyTorch, Scikit-Learn or general libraries used in handling data and support implemented functions in Python 3, following the project's instructions.

### LICENSE
All files made available in this repository follow the licensing guidelines defined in the file **LICENSE.md**.
