# Target achieve more than 50% Test accuracy on Tiny ImageNet dataset 

### 1. Model to be used : [custom Resnet18](https://github.com/sumitsarkar1/sumitEVA7/tree/main/models) ###
### 2. Number of epochs to be used: 50 ###
### 3. Learning Rate (LR) Policy : One Cycle LR with max LR at 15th epoch ###
### 4. Data Augmentation used : Padding, Random Crop, Horizontal Flip, Rotate , RGB Shift 
### 4. Test Accuracy to achieve : 50% ###

## Data Set used : [TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
## Split Data : 
```
print("train data set : ",len(trainloader)*100/(len(trainloader) + len(testloader)),"%")
print("test data set : ",len(testloader)*100/(len(trainloader) + len(testloader)),"%")
train data set :  70.0 %
test data set :  30.0 %
```

## Training/Testing loss and accuracy ##


