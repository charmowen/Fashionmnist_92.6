import torch
from visualize import vis_datasets
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
from lenet import LeNet

trainsforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
train_dataset = torchvision.datasets.FashionMNIST(root = './fashionMNIST/data',train = True, download = False, transform =trainsforms)
test_dataset = torchvision.datasets.FashionMNIST(root = './fashionMNIST/data',train = False, download = False, transform =trainsforms)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 100, shuffle=True, num_workers = 4)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 100, shuffle = False, num_workers = 4)

text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#vis_datasets(train_loader,text_labels)

epoch_size = 20
lr = 0.001
device = torch.device('cuda')

net = LeNet(1).to(device)

optimizer = optim.Adam(net.parameters(),lr = lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [14,19], gamma = 0.1,last_epoch =-1)
criterion = nn.CrossEntropyLoss()
loss_total = 0

for epoch in range(epoch_size):


    for index,(imgs, labels) in enumerate(train_loader):
        imgs,labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = net(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        if (index+1) % 400 == 0:
            print(epoch,index+1,loss_total/2000.0)
            loss_total = 0
    scheduler.step()
    print(epoch, 'lr:', scheduler.get_lr())

total = 0
correct = 0
with torch.no_grad():
    for img_test,label_test in test_loader:
        batch_label = label_test.size(0)
        img_test,label_test = img_test.to(device),label_test.to(device)
        test_out = net(img_test)
        predicted = torch.argmax(test_out,dim = 1)
        total += batch_label
        correct += predicted.eq(label_test).sum().item()

print('accuracy:',correct / total*100)
