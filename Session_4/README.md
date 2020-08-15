
## Assignment 4- Architectural Basics
>#### 


***
**Ask of the Assignment**


1.  We have considered many many points in our last 4 lectures. Some of these we have covered directly and some indirectly. They are:
    1.  How many layers,
    2.  MaxPooling,
    3.  1x1 Convolutions,
    4.  3x3 Convolutions,
    5.  Receptive Field,
    6.  SoftMax,
    7.  Learning Rate,
    8.  Kernels and how do we decide the number of kernels?
    9.  Batch Normalization,
    10.  Image Normalization,
    11.  Position of MaxPooling,
    12.  Concept of Transition Layers,
    13.  Position of Transition Layer,
    14.  DropOut
    15.  When do we introduce DropOut, or when do we know we have some overfitting
    16.  The distance of MaxPooling from Prediction,
    17.  The distance of Batch Normalization from Prediction,
    18.  When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
    19.  How do we know our network is not going well, comparatively, very early
    20.  Batch Size, and effects of batch size
    21.  etc (you can add more if we missed it here)
2.  Refer to this code:  [COLABLINK (links to an external site)](https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx)
    -  **WRITE IT AGAIN SUCH THAT IT ACHIEVES**  
        1.  99.4% validation accuracy
        2.  Less than 20k Parameters
        3.  You can use anything from above you want.
        4.  Less than 20 Epochs
        5.  No fully connected layer


***
#### Assignment Objective
***

>The goal of this assignment is to achieve >99.40% accuracy on the test set of the MNIST dataset. The model needs to have the following restriction -
> - Less than 20K Parameters
> - Less than 20 epochs
> - No fully connected layers.
#### Methodology
> - 3x3 Convolutions - Best and Optimal Kernal, feature extractor.
> - Performed three 3X3 convolutions with 16 channels before maxpooling to achieve a receptive field of 7 since for images of size 28X28, edges and gradients are at a minimum size of 7 pixels.
> - Post maxpooling performed three more 3X3 convolution with 16 channels to reach 7X7 image size and receptive field of 22.
> - Used BatchNormalization and Dropout of 10% after every convolutional block except the last block.
> - Used GAP(Global Average Pooling) at 7X7 to convert to 1X1.
> - Performed 1X1 convolution to reduce the number of channels to 10.
> - Used softmax activation function at the end to get the probability for each class.
> - Loss Function- Negative Log Probability
> - Optimizer- SGD with momentum of 0.1.
> 
#### Parameters
```
 ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
       BatchNorm2d-2           [-1, 16, 26, 26]              32
         Dropout2d-3           [-1, 16, 26, 26]               0
            Conv2d-4           [-1, 16, 24, 24]           2,304
       BatchNorm2d-5           [-1, 16, 24, 24]              32
         Dropout2d-6           [-1, 16, 24, 24]               0
            Conv2d-7           [-1, 16, 22, 22]           2,320
       BatchNorm2d-8           [-1, 16, 22, 22]              32
         Dropout2d-9           [-1, 16, 22, 22]               0
        MaxPool2d-10           [-1, 16, 11, 11]               0
           Conv2d-11             [-1, 16, 9, 9]           2,304
      BatchNorm2d-12             [-1, 16, 9, 9]              32
        Dropout2d-13             [-1, 16, 9, 9]               0
           Conv2d-14             [-1, 16, 7, 7]           2,304
      BatchNorm2d-15             [-1, 16, 7, 7]              32
        Dropout2d-16             [-1, 16, 7, 7]               0
           Conv2d-17             [-1, 16, 7, 7]           2,304
      BatchNorm2d-18             [-1, 16, 7, 7]              32
        Dropout2d-19             [-1, 16, 7, 7]               0
        AvgPool2d-20             [-1, 16, 1, 1]               0
           Conv2d-21             [-1, 10, 1, 1]             160
================================================================
Total params: 12,032
Trainable params: 12,032
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.72
Params size (MB): 0.05
Estimated Total Size (MB): 0.77
----------------------------------------------------------------
```
- Total number of parameters: 12,032.
#### Hyperparameter
> - Learning Rate : 0.01
> - Batch Size : 32
> - Dropout : 0.1


#### Final Outcome
> - Achieved accuracy of 99.43% at 18th epoch 