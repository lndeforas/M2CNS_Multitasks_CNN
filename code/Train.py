
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from MultiTaskCNN import MultiTaskCNN
from MultiTaskLoss import MultiTaskLoss
import torch.nn as nn
import  torch.optim
from visualization import vis
import itertools

class ExtractFirstChannel(object):
    def __call__(self, img):
        return img[0, :, :].reshape(1,64,64)
    
batchsize = 128
transform = transforms.Compose([
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    ExtractFirstChannel(),     # the image is in black & white so we extract only first channel
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image
])
# loading the training dataset
traindataset = datasets.ImageFolder(root='C:/Helene/School/CNS/Masteriales/MultiDigitMNIST-master/dataset/double_mnist/train', transform=transform)
trainloader = DataLoader(traindataset,batch_size=batchsize,shuffle=True)
# loading the validation dataset
valdataset = datasets.ImageFolder(root='C:/Helene/School/CNS/Masteriales/MultiDigitMNIST-master/dataset/double_mnist/val', transform=transform)
valloader = DataLoader(valdataset,batch_size=batchsize)
# loading the test dataset
testdataset = datasets.ImageFolder(root='C:/Helene/School/CNS/Masteriales/MultiDigitMNIST-master/dataset/double_mnist/test', transform=transform)
testloader = DataLoader(testdataset,batch_size=batchsize)

number_of_batches = len(trainloader)

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using GPU")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

def proba_labels(i, j):
    tab = [[0 for k in range(10)], [0 for k in range(10)]]
    for k in range(10):
        if (i == k):
            tab[0][k] = 1
        if (j == k):
            tab[1][k] = 1
    return tab

def train(model,epochs):
    model.train()
    losses_history = [ ]
    for epoch in range(epochs) :
        run_loss=0
    
        for inputs,labels in trainloader:
            
            # split the labels ex: "90" => [9,0]
            labels = torch.tensor([ proba_labels(int(traindataset.classes[e][0]) , int(traindataset.classes[e][1])) for e in labels ])
            inputs = inputs.to(device)
            labels = labels.float()
            labels = labels.to(device)
    
            # make predictions
            outputs=model(inputs)
            #initialize gradients
            optimizer.zero_grad()
            #compute loss
            loss = loss_fn(outputs,labels)
            #backpropagation
            loss.backward()
            #update weights
            optimizer.step()
            run_loss+=loss.item()
        
        run_loss = run_loss / number_of_batches
        print(f"Epoch: [{epoch + 1}/{epochs}], Loss: {run_loss}")
        losses_history.append(run_loss)
    ## vis(losses_history)

def evaluate(model,dataset,loader):
    
    # evaluation step
    number_of_batches = len(loader)
    model.eval()
    correct_predictions=0
    total=0
    for inputs, labels in loader:
        labels = torch.tensor([ proba_labels(int(dataset.classes[e][0]) , int(dataset.classes[e][1])) for e in labels ])
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
    
        for i in range(len(outputs)):
            total+=1
            if( (outputs[i,0,:].argmax()==labels[i,0,:].argmax()) and (outputs[i,1,:].argmax()==labels[i,1,:].argmax()) ):
                correct_predictions+=1
    accuracy = correct_predictions/total
    return accuracy

params = {
    "optimizers" : [torch.optim.RMSprop],
    "epochs": [20,25],
    "filters":[32,64]
 }
combinations = list(itertools.product(*params.values()))
best_acc = 0
# initialize the model
print(" --- training step ---")
for i,comb in enumerate(combinations): #for each combination
    print(f"Model {i+1}/{len(combinations)}")
    model = MultiTaskCNN(comb[2]) # passing the number of filters paramaeter to the model
    model=model.to(device) # using the model with GPU
    optimizer = comb[0](model.parameters(), lr=0.001) 
    loss_fn= MultiTaskLoss()
    epochs=comb[1]
    train(model,epochs) # training the model
    acc = evaluate(model,valdataset,valloader) # evaluating the model on the validation dataset
    # selecting the best model and combination
    if(acc> best_acc):
        best_acc = acc
        best_model = model
        best_comb=comb

best_model_parameters=dict(zip(params.keys(),best_comb))
print(f"best model : {best_model_parameters}")
acc=evaluate(best_model,traindataset, trainloader)
print(f'Train Accuracy: {acc * 100:.2f}%')
acc=evaluate(best_model,valdataset, valloader)
print(f'Validation Accuracy: {acc * 100:.2f}%')
acc=evaluate(best_model,testdataset, testloader)
print(f'Test Accuracy: {acc * 100:.2f}%')

