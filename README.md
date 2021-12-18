# Target : achieve more than 50% Test accuracy on TinyImageNet dataset 

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
## Last 2 epoch results ##
```
EPOCH: 49 LR Value:  0.00016065038254939196
Loss=0.3622516691684723 Batch_id=1203 Train Accuracy=99.68: 100%|█| 1204/12

Test set: Average loss: 0.0384, Test Accuracy: 19126/33000 (57.96%)

EPOCH: 50 LR Value:  4.0270099675616e-05
Loss=0.38646507263183594 Batch_id=1203 Train Accuracy=99.66: 100%|█| 1204/1

Test set: Average loss: 0.0384, Test Accuracy: 19084/33000 (57.83%)
```


