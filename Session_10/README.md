
## Session 10 - Advanced Concepts in Training & Learning Rates
***

> ##### Assignment Objective
> * Pick assignment 9 code and add cutout
> * Implement LR finder and ReduceLR on Plateau
> * Implement GradCam function as module
> * Find Best LR and use SGD with momentum
> * Train for 50 epochs
> * Show training and test accuracy curves
> * Target 88% Accuracy.
> * Run Gradcam on 25 misclassified images and show ground truth and predictions for each


***
> ##### Modular Implementation
> * EVA5_Session10/models/resnet.py - Resnet18 model 
> * EVA5_Session10/config.py: Parameters needs to set to run the model
> * EVA5_Session10/gradcam/: Gradcam implementations
> * EVA5_Session10/lr_finder.py: Implementation to find best LR for the model
> * EVA5_Session10/engine.py - Train and test functions
> * EVA5_Session10/save_load.py - To load and save models
> * EVA5_Session10/preprocessing.py - Albumentation implementation 

***

> ##### Hyperparameter
> - Loss Function: Cross Entropy Loss
> - Optimizer: SGD with momentum
> - Batch Size: 32
> - Number of epochs: 50
> - Learning Rate: 2.31E-01
> - Scheduler: Reduce LR on Plateau with patience of 15

> ##### Transformations
> - Rotate
> - Horizontal Flip
> - Cutout


***
> ##### Model Summary
> * The model reaches a test accuracy of 89.53% in CIFAR-10 dataset in 50 epochs. 
> * Target accuracy is reached after 42 epochs.
> * The model has 11,173,962 parameters.


```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------
```
***

> ![My Image](https://github.com/hardayaleva5/EVA5_Phase1_Assignment/blob/master/Session_10/Validation_Loss_Graph.png)
***


> ![My Image](https://github.com/hardayaleva5/EVA5_Phase1_Assignment/blob/master/Session_10/Validation_Accuracy_Graph.png)

***

> ##### Misclassified images

> ![My Image](https://github.com/hardayaleva5/EVA5_Phase1_Assignment/blob/master/Session_10/misclassified_images.png)

***

> ##### Gradcam for some of the images 

> ![My Image](https://github.com/hardayaleva5/EVA5_Phase1_Assignment/blob/master/Session_10/GradCam.png)

***