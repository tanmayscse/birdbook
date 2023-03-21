# Bird Image Classifier: Finding an Economic Solution  

For the final project of CSE 455, I decided to do the kaggle bird classification. 


### Project Scope and motivation

For the kaggle competition, I was limited by the processing power 
available for training. Due to this, I decided to  explore all the 
less intensive models available on the `torchvision` library. To make 
the most economical use of the computing resources available to me, I decided to explore,  
train and run the less intensive models. The models which I tested are as follows: 

- Mobilenet with weights MobileNet_V3_Large_Weights.IMAGENET1K_V2
- Efficientnet with weights EfficientNet_B3_Weights.IMAGENET1K_V1
- Resnet101 with weights ResNeXt101_64X4D_Weights.IMAGENET1K_V1
- Resnet50 with weights ResNet50_Weights.IMAGENET1K_V2

Resnet101 model is intuitively not less intensive to train. However, I decided to use it  
as a comparison factor, and also because resnet 50 is the default model we used in the class. 


### Method

For each model, I divided the training process of every model in 4 steps: 

##### Step 1 
Create the model and train it for 4 to 5 epochs to see the convergence with different 
learning rates. Terminate if significant convergence does not occur by the third epoch. 
Add the schedule of higher learning rate for the first one to two epochs and lower it for fine tuning. 


##### Step 2
Once a satisfactory convergence is observed in step 1, train the model for 2 more epochs, this time
to look for a good learning rate to find a good minima. There is a chance of the optimizer getting 
stuck in the local minima. To mitigate that, add some momentum to explore more regions. 

#### Step 3
Continue training the model for sets of 2 epochs with varying learning rate and optimizer
until a good minima is reached. 

#### Step4 
Use the model to generate predictions and evaluate on kaggle


Normally, the training approach would involve options like k-cross validation. But owing to the limited 
compute resources for the task, my focus was to find the model which could be fast and reliably trained.
The assumption being that in a time constrained situation, validation is generally a secondary option as long as a good amount of 
regularization is applied. 
_______

### Observations: 


#### Mobilenet with weights MobileNet_V3_Large_Weights.IMAGENET1K_V2

Mobilenet appeared to be a promising option






















