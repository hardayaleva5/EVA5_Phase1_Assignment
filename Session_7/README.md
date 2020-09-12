
## Session 7 - Advanced convolutions
***

> ##### The aim of this assignment is to fix the network given with below objectives

> * Change the code such that it uses GPU
> * Change the architecture to C1C2C3C40 (basically 3 MPs)
> * Total RF must be more than 44
> * One of the layers must use Depthwise Separable Convolution
> * One of the layers must use Dilated Convolution
> * Use GAP (compulsory):- add FC after GAP to target #of classes (optional)
> * Achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M.

***
#### Hyperparameter
***

> - Dropout: 0.10
> - Batch Size: 256
> - Number of epochs: 25
> - Learning Rate: 0.016
> - Ghost Batch Norm Splits: 2
> ##### Model architecture is built including Depthwise seperable convolutions and Dilated Convolutions

> ##### Total Parameters : 657,146

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             896
    GhostBatchNorm-2           [-1, 32, 32, 32]              64
            Conv2d-3           [-1, 32, 32, 32]           9,248
    GhostBatchNorm-4           [-1, 32, 32, 32]              64
           Dropout-5           [-1, 32, 32, 32]               0
         MaxPool2d-6           [-1, 32, 16, 16]               0
            Conv2d-7           [-1, 16, 18, 18]             528
    GhostBatchNorm-8           [-1, 16, 18, 18]              32
           Dropout-9           [-1, 16, 18, 18]               0
           Conv2d-10           [-1, 64, 18, 20]           3,136
   GhostBatchNorm-11           [-1, 64, 18, 20]             128
          Dropout-12           [-1, 64, 18, 20]               0
           Conv2d-13           [-1, 64, 20, 20]          12,352
   GhostBatchNorm-14           [-1, 64, 20, 20]             128
          Dropout-15           [-1, 64, 20, 20]               0
        MaxPool2d-16           [-1, 64, 10, 10]               0
           Conv2d-17           [-1, 32, 12, 12]           2,080
   GhostBatchNorm-18           [-1, 32, 12, 12]              64
          Dropout-19           [-1, 32, 12, 12]               0
           Conv2d-20          [-1, 128, 12, 12]          36,992
   GhostBatchNorm-21          [-1, 128, 12, 12]             256
          Dropout-22          [-1, 128, 12, 12]               0
           Conv2d-23          [-1, 128, 10, 10]         147,584
   GhostBatchNorm-24          [-1, 128, 10, 10]             256
          Dropout-25          [-1, 128, 10, 10]               0
        MaxPool2d-26            [-1, 128, 5, 5]               0
           Conv2d-27            [-1, 256, 5, 5]         295,168
           Conv2d-28             [-1, 64, 5, 5]         147,520
        AvgPool2d-29             [-1, 64, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             650
================================================================
Total params: 657,146
Trainable params: 656,154
Non-trainable params: 992
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.50
Params size (MB): 2.51
Estimated Total Size (MB): 6.02
----------------------------------------------------------------
```
***
#### Results

Achived grater than 82% validation accurcy at 20th epoch

> ![My Image](https://github.com/hardayaleva5/EVA5_Phase1_Assignment/blob/master/Session_7/Validation_Loss_Graph.png)
***


> ![My Image](https://github.com/hardayaleva5/EVA5_Phase1_Assignment/blob/master/Session_7/Validation_Accuracy_Graph.png)

***