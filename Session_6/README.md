
## Assignment 6 - Batch Normalization and Regularization
***

> ##### The aim of this assignment is to take the best code from the previous assignment, run for 25 epochs and report findings:

> * With L1 + Batch Norm
> * With L2 + Batch Norm
> * With L1 and L2 with Batch Norm
> * With Ghost Batch Norm
> * With L1 and L2 with Ghost Batch Norm



> ##### Write a single loop or iterator to iterate through these conditions
> ##### Draw 2 graphs with proper legends showing the following:
> * Validation accuracy curve for all 5 jobs above
> * Loss curves for all 5 jobs above

> ##### Find any 25 misclassified images (combined into single plot) for the GBN model. Must use the saved model from the above jobs. Must show actual and predicted class names.

***
#### Parameters
***
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 28, 28]              90
              ReLU-2           [-1, 10, 28, 28]               0
       BatchNorm2d-3           [-1, 10, 28, 28]              20
           Dropout-4           [-1, 10, 28, 28]               0
            Conv2d-5           [-1, 10, 28, 28]             900
              ReLU-6           [-1, 10, 28, 28]               0
       BatchNorm2d-7           [-1, 10, 28, 28]              20
           Dropout-8           [-1, 10, 28, 28]               0
         MaxPool2d-9           [-1, 10, 14, 14]               0
           Conv2d-10           [-1, 10, 12, 12]             900
             ReLU-11           [-1, 10, 12, 12]               0
      BatchNorm2d-12           [-1, 10, 12, 12]              20
          Dropout-13           [-1, 10, 12, 12]               0
           Conv2d-14           [-1, 12, 10, 10]           1,080
             ReLU-15           [-1, 12, 10, 10]               0
      BatchNorm2d-16           [-1, 12, 10, 10]              24
          Dropout-17           [-1, 12, 10, 10]               0
           Conv2d-18             [-1, 16, 8, 8]           1,728
             ReLU-19             [-1, 16, 8, 8]               0
      BatchNorm2d-20             [-1, 16, 8, 8]              32
          Dropout-21             [-1, 16, 8, 8]               0
           Conv2d-22             [-1, 16, 6, 6]           2,304
             ReLU-23             [-1, 16, 6, 6]               0
      BatchNorm2d-24             [-1, 16, 6, 6]              32
          Dropout-25             [-1, 16, 6, 6]               0
           Conv2d-26             [-1, 16, 4, 4]           2,304
             ReLU-27             [-1, 16, 4, 4]               0
      BatchNorm2d-28             [-1, 16, 4, 4]              32
        AvgPool2d-29             [-1, 16, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             160
================================================================
Total params: 9,646
Trainable params: 9,646
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.63
Params size (MB): 0.04
Estimated Total Size (MB): 0.67
----------------------------------------------------------------
```

***
#### Hyperparameter
***

> - Dropout: 0.04
> - Batch Size: 64
> - Learning Rate: 0.01
> - L1 Parameter: 0.0002
> - L2 Parameter: 0.0001
> - Ghost Batch Norm Splits: 4

***
#### Results

> - The loss for various scenarios are plotted below:


> ![My Image](https://github.com/hardayaleva5/EVA5_Phase1_Assignment/blob/master/Session_6/S6_Assignment_Solution_Images/Plot_Loss_Accuracy.png)
***

> - The accuracy for various scenarios are plotted below:

> ![My Image](https://github.com/hardayaleva5/EVA5_Phase1_Assignment/blob/master/Session_6/S6_Assignment_Solution_Images/Plot_Accuracy.png)
***


#### Misclassified Images

> - The misclassified images for the model with GBN are shown below:


> ![My Image](https://github.com/hardayaleva5/EVA5_Phase1_Assignment/blob/master/Session_6/S6_Assignment_Solution_Images/Misclassified_Images.png)
***

