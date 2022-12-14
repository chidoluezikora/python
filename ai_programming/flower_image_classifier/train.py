import torch, json, argparse
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def parse_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'data_dir',
        action = 'store',
        type = str,
        help = 'directory containing flower images'
    )
    parser.add_argument(
        '--save_dir',
        type = str,
        default = 'checkpoint.pth',
        help = 'path to save the checkpoint'
    )
    parser.add_argument(
        '--arch',
        type = str,
        default = 'vgg19',
        help = 'architecture to be used'
    )
    parser.add_argument(
        '--gpu',
        action = 'store_true',
        default = False,
        help = 'pass argument to use gpu'
    )
    parser.add_argument(
        '--epochs',
        type = int,
        default = 30,
        help = 'number of epochs'
    )
    parser.add_argument(
        '--hidden_units',
        type = int,
        default = 4096,
        help = 'hidden units for first hidden layer'
    )
    parser.add_argument(
        '--learnrate',
        type = int,
        default = 0.003,
        help = 'learn rate'
    )

    args = parser.parse_args()
    
    return args


in_arg = parse_input_args()
data_dir  = in_arg.data_dir
save_dir = in_arg.save_dir
arch = in_arg.arch
gpu = in_arg.gpu
epochs = in_arg.epochs
hidden_units = in_arg.hidden_units
learn_rate = in_arg.learnrate

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])


valid_transforms = transforms.Compose([transforms.Resize(270),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])


train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=108, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=108, shuffle=True)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=108, shuffle=True)


model = models.vgg19(pretrained=True)
model.name = 'vgg19'
if arch == 'alexnet':
    model = models.vgg19(pretrained=True)
    model.name = 'vgg19'
elif arch == 'resnet152':
    model = models.resnet152(pretrained=True)
    model.name = 'resnet152'
elif arch == 'densenet161':
    model = models.densenet161(pretrained=True)
    model.name = 'densenet161'
else:
    model = models.vgg19(pretrained=True)
    model.name = 'vgg19'


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(25088, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 2048)
        self.fc3 = nn.Linear(2048, 102)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x

classifier = Classifier()
model.classifier = classifier


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
model.to(device)


def train():

    train_losses, valid_losses = [], []

    for e in range(epochs):
        running_loss = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0

            with torch.no_grad():
                model.eval()

                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    valid_loss += criterion(outputs, labels).item()
                    ps = torch.exp(outputs)
                    equals = labels.data == ps.max(1)[1]
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

                model.train()    
                    
                train_losses.append(running_loss/len(trainloader))
                valid_losses.append(valid_loss/len(validloader))

                print("Epoch: {}/{} | ".format(e+1, epochs),
                "Training Loss: {:.3f} | ".format(train_losses[-1]),
                "Validation/Test Loss: {:.3f} | ".format(valid_losses[-1]),
                "Validaton/Test Accuracy: {:.3f}%".format(accuracy/len(validloader)*100))


def test_data():
    equal = 0
    total = 0

    with torch.no_grad():
        model.eval()
        
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pd = torch.max(outputs.data, 1)
            total += labels.size(0)
            equal += (pd == labels).sum().item()

    print("Test accuracy on test data: {:.3f}%".format(equal/total*100))


train()
test_data()


model.class_to_idx = train_datasets.class_to_idx

checkpoint = {
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'classifier': model.classifier,
            'class_to_dict': model.class_to_idx,
            }

torch.save(checkpoint, save_dir)
