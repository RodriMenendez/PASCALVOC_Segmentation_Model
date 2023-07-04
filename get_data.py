import torch
import torchvision
import torchvision.transforms as transforms
from subprocess import run

# extracting data from tar file
cmd = 'tar xf drive/MyDrive/SURF_2023/Practice/Colabs/pascal/VOCtrainval_11-May-2012.tar -C /content/'
run(cmd)

class ValueReplace():
    def __init__(self, old_val, new_val):
        self.old_val = old_val
        self.new_val = new_val

    def __call__(self, sample):

        sample[sample==self.old_val] = self.new_val

        return sample

class Squeeze():
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, sample):
        sample = sample.squeeze(self.dim)

        return sample

class SetCuda():
    # putting all data on GPU at once takes up too much RAM
    def __init__(self):
        self.available = torch.cuda.is_available()
    
    def __call__(self, sample):
        if self.available:
            sample = sample.cuda()

        return sample

def get_PASCAL(path='/content/', batch_size=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    target_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.CenterCrop((224, 224)),
        ValueReplace(255, 21),
        Squeeze(),
    ])

    train_data = torchvision.datasets.VOCSegmentation(root=path, year='2012', image_set='trainval', \
                    transform=transform, target_transform=target_transform, download=False)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True)

    val_data = torchvision.datasets.VOCSegmentation(root=path, year='2012', \
                    image_set='val', transform=transform, target_transform=target_transform, download=False)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_data, trainloader, val_data, valloader
