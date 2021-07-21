'''Digit recognition with pyTorch'''

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision #contatins utilities for working with image data
from torchvision.datasets import MNIST #MNIST has 24 by 24 px images of digits 
import matplotlib.pyplot as plt #for plotting the images from dataset
import torchvision.transforms as transforms #transforms image to tensor
from torch.utils.data.sampler import SubsetRandomSampler #samples elements randomly from a given list, while creating batches of data
from torch.utils.data.dataloader import DataLoader #to split data into batches of predefined sizeimport torch.nn.functional as F

'''download training datset that contains 60000 training-images & 10000 test-images in Folder 'data' '''
dataset = MNIST(root='data/',download=True)
#print(len(dataset))
#print(dataset[0])

#image,label = dataset[0]
#plt.imshow(image,cmap='gray') #display the image
#print('Label:',label) #print label of image, dataset[0] is 5.

'''since pyTorch require tensors, we need to convert these images to tensors'''
dataset = MNIST(root='data/',train=True,transform=transforms.ToTensor())
#img_tensor,label = dataset[0]
#print(img_tensor.shape,label) 
'''image is converted into a 1*28*28 tensor, where first dimension(1) is for color, and 28*28 is width and height'''

'''
While building real world ML models, it is common to split the dataset into 3 parts
1. Training set - used to train the model, compute loss, adjust the weights and biases using GD
2. Validation set - used to evaluate the model, adjust lerning rate, decide how long to train so as to pick the best model
3. Test set - used to compare different models, or modeling approaches
'''

'''Since there is no predefined validation set, we need to manually split the dataset into training set and validation set'''
def split_dataset(numof_images, split_percent):
    #determine size of validation set
    validationset_size = int((split_percent/100)*numof_images)
    #create a random permutation of 0 to numof_images-1
    rand = np.random.permutation(numof_images) #creates a np array with random numbers from 0 - 59999
    #get the first N values from rand that becomes the training_set, and rest becomes validation_set, where N = validationset_size
    return rand[validationset_size:], rand[:validationset_size]

training_set, validation_set = split_dataset(len(dataset),20)
#print(len(training_set),len(validation_set))

'''randomly shuffle the training_set and validation_set and create batches of data'''
training_sampler = SubsetRandomSampler(training_set)
training_dl = DataLoader(dataset,batch_size=100,sampler=training_sampler) #create randomized batches from the training_set

validation_sampler = SubsetRandomSampler(validation_set)
validation_dl = DataLoader(dataset,batch_size=100,sampler=validation_sampler)

'''create our ML model'''
input_size = 28*28
output_size = 10 #output is vector of size 10
#model = nn.Linear(input_size,output_size)
#print(model.weight.shape)

'''need to reshape the size of input images from x*1*28*28 to x*784, where x is the batch_size so that we will get x no. of image vectors with 784 elements each '''
class MNISTmodel(nn.Module):
    #constructor
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)
    
    def forward(self,xb): #grgs = self(instance of current object), batch of inputs
        xb = xb.reshape(-1,784) #weights.transpose() = 784*10. So, matric mul. of xb*weight.t() = -1*10 matrix, -1 is for generic batch-size x
        out = self.linear(xb)
        return out
    
model = MNISTmodel()

for images,labels in training_dl:
    prediction_outputs = model(images)
    break

'''
apply softmax for each output row
softmax takes prediction_outputs and converts them into probabilities(0-1)
this is done because we want the 10 elements of output row to be probabilities,
with each element representing probability that the model thinks the image is a digit from 0-10'''
probability = F.softmax(prediction_outputs,dim=1)
#print((probability[:2].data))
#print("Sum:",torch.sum(probability[0]).item())

#pick the max/best probability from the row
max_prob, prediction = torch.max(probability,dim=1)
#print(max_prob)

'''perform element-wise comparison of labels and our prediction and give accuracy percentage of the model'''
'''def accuracy(labels,predictions):
    _, preds = torch.max(predictions,dim=1)
    return torch.sum(labels==preds).item() / len(labels)

print(accuracy(labels,prediction))
'''
'''
What we have till now: Model, dataset, prediciton, target(labels).
Now we need to define the loss functon. 
A commonly used loss function in classification problem is "cross entropy"
'''

'''
Cross entropy: 
    1. we take the each label from the label and create a 10-element vector like below
    eg: if the label is of an image of 2, we create [0,0,1,0,0,0,0,0,0,0]
    2. take the log of the probability predicted by our model and do a dot product
    eg:[0.1014,0.0993,0.1122,0.7373,0.3213,0.8373,0.0993,0.0012,0.1342,0.9983]
    output = 1*log(0.1122)  
    3. Finally, we take the average of the cross entropy of all output rows to get the overall loss for the given batch
'''

#calculate loss
loss_fn = F.cross_entropy
loss = loss_fn(prediction_outputs,labels)
#print(loss)

'''Define the optimizer and train the model'''
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

#calculate loss for a batch of data, apply gradient descent
def loss_batch(model,loss_fn,xb,yb,opt=None,metric=None):
    #step 1: generate predictions
    pred = model(xb)
    #step 2: calculate loss
    loss = loss_fn(pred,yb)
    
    if opt is not None: #if an optimizer is provided
        #step 3:compute gradient
        loss.backward()
        #step 4: adjust the weights and biases using gradient
        opt.step()
        #step 5: reset the gradients to 0
        opt.zero_grad()
    
    metric_result=None
    if metric is not None: #if metric like calculating the accuracy is provided
        #compute the metric
        metric_result = metric(yb,pred)

    return loss.item(), len(xb), metric_result


#calculate overall loss for validation set, so that we can evaluate our model
def evaluate(model,loss_fn,validation_dl,metric=None):
    with torch.no_grad(): #we do not compute gradient for the validation set, we only compute it for thr training set
        #pass each batch of validations set into the model
        results = (loss_batch(model,loss_fn,xb,yb,metric=metric) for xb,yb, in validation_dl)
        
        losses,size,metric_result = zip(*results)
        
        totalsizeof_dataset = np.sum(size)
        
        avg_loss = np.sum(np.multiply(losses,size))/totalsizeof_dataset
        
        avg_metric = None
        
        if metric is not None:
            avg_metric = np.sum(np.multiply(metric_result,size))/totalsizeof_dataset
            
        return avg_loss, totalsizeof_dataset, avg_metric

#redefine accuracy function to operate on entire batch of outputs
def accuracy(labels,predictions):
    _, preds = torch.max(predictions,dim=1)
    return torch.sum(labels==preds).item() / len(labels)

'''Initially,the loss and accuracy for validation set should be similar to the values we got in the training set'''        
#val_loss,tottal,val_accuracy = evaluate(model,loss_fn,validation_dl,metric=accuracy)
#print("Loss: {:.4f}, Accuracy: {:.4f}".format(val_loss,val_accuracy)) 

def train_model(numof_epochs, model, loss_fn, opt, training_dl,validation_dl,metric=None):
    #repert of given num of epochs
    for epoch in range(numof_epochs):
         #train with batches of data
         #xb = input data
         #yb = target data
         #TRAINING
         for xb,yb in training_dl:             
             # calculate loss
             loss,_,_ = loss_batch(model,loss_fn,xb,yb,opt)
             
         #EVALUATION
         result = evaluate(model,loss_fn,validation_dl,metric)   
         val_loss,total,val_metric = result
          
         #PRINT PROGRESS
         if metric is None:
             print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1,numof_epochs,val_loss))
             
         else:
             print("Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}".format(epoch+1, numof_epochs, val_loss, metric.__name__,val_metric))
             
train_model(5,model,loss_fn,optimizer,training_dl,validation_dl,metric=accuracy)

'''Testing our model with individual images'''  
#get the test dataset
test_dataset = MNIST(root='data/',train=False,transform=transforms.ToTensor())

#select a test-image from the dataset           
img,label = test_dataset[17]
plt.imshow(img[0],cmap='gray')

#generate prediction through our model based on the test image
def predict_image(img,model):
    xb=img.unsqueeze(0)
    yb=model(xb) #yb=output of our model
    _,preds = torch.max(yb,dim=1)
    return preds[0].item()

print('Label:',label,',Predicted:',predict_image(img,model))