
## Assignment 5 - Coding Drill Down


***
#### Assignment Objective


>The goal of this assignment is to achieve >99.40% accuracy on the test set of the **MNIST handwritten dataset**. 

> The objective needs to be achieved using following conditions :
> * Consistently reach greater than or equal to 99.4% accuracy in the last few epochs
> * Less than 10K Parameters
> * Less than or equal to 15 epochs
> * The objective is to be reached in exactly 4 steps
> * Each step needs to have a "target, result, analysis" TEXT block (either at the start or the end)
> * Each step must be convincing as to why it was  decided that the target should be what it was decided to be, and the analysis must be correct


***
#### Step1 - Setup
***

##### Target

> * Get the setup right
> * Read MNIST dataset, set train test split and create Data Loader
> * Get the summary statistics for the data
> * Set initial transforms and apply transformation to the train and test set separately
> * Get the basic neural net architecture skeleton right. We will try and avoid changing the skeleton later
> * Set basic training and test loop

##### Results

> * Parameters: 992,800
> * Best Training Accuracy: 99.88
> * Best Test Accuracy: 99.14

##### Analysis

> * Very heavy model for such an easy problem. Lots of parameters. Have to reduce the number of parameters in the next step
> * Test accuracy is way below the target accuracy
> * Model is overfitting


***
#### Step2 - Reducing the number of parameters
***

##### Target

> * Make the model lighter by reducing the number of parameters
>    * Reduce the number of channels
>    * Use Global Average Pooling instead of a large 7X7 kernel in the output block
> * Have more number of channels before maxpooling
> * Transition layer after reaching a receptive field of 5 instead of 7 since in smaller images, edges and gradients starts getting detected at a receptive field of 5.

##### Results

> * Parameters: 9,712
> * Best Training Accuracy: 98.44
> * Best Test Accuracy: 98.53

##### Analysis

> * Good model
> * Model is not overfitting
> * There is room for improvement in terms of accuracy. Can push the model further.


***
#### Step3 - Improving model performance by adding Batch Norm and regularization to prevent overfitting
***

##### Target

> * Apply Batch Norm after every convolution layer except the last layer before output to improve accuracy 
> * Apply dropouts if there is overfitting after applying batch norm

##### Results

> * Parameters: 9,904
> * Best Training Accuracy: 98.75
> * Best Test Accuracy: 99.24

##### Analysis

> * Good model but still we haven't been able to reach our target accuracy
> * Applied low dropout to prevent overfitting and for model stability
> * We can still increase the accuracy further
> * We can't increase the model capacity any further. We have to rely on some other method to increase the accuracy further
> * We can decrease batch size and try image augmentations


***
#### Step4 - Improving model performance by adding Image Augmentation and LR Scheduler
***


   ##### Target

> * Apply Image Augmentation techniques like Random Rotation since there are some images which are slightly rotated
> * Apply LR scheduler for finding good Learning rate for better convergence
> * Reduce batch size for more parameter updates

##### Results

> * Parameters: 9,904
> * Best Training Accuracy: 99.08
> * Best Test Accuracy: 99.44

##### Analysis

> * Very good model. Able to achieve the target accuracy with required parameters
> * Adding Image Augmentation and LR scheduler have made the model more robust. There is less variance in the validation accuracy

***
#### Parameters
***

> - Total number of parameters in the network is 9,904.

***
#### Hyperparameter
***

> - Learning Rate : 0.01
> - Batch Size : 32
> - Dropout : 0.1
> - LR Step Size : 5

***
#### Final Outcome
***
> - Achieved accuracy of 99.44% at 15th epoch