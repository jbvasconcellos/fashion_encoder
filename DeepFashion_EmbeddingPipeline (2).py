import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import DeepFashion_Toolbox as TB


class dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform_img, transform_bbox):
        """
        Args:
            data: images dataframes
            transforms: preprocess transforms
        """
        self.data = data
        self.transform_img = transform_img
        self.transform_bbox = transform_bbox
    
    def __getitem__(self, index): # Extracted from Gabriele's notebook

        curr  = self.data.iloc[index]
#         img   = Image.open(curr['img_path']).convert('RGB')
        img =  Image.open(curr['img_path'])#(curr['img_path'], drawn=True) # <------ 
#         import pdb; pdb.set_trace()
        bbox = curr['bounding_boxxes']
        polygon = curr['segmentation']
        sliced_img = TB.bbox_BG_remover(img, bbox, polygon) # list of images
#         sliced_img = bbox_to_tensors(image_tensor, bbox)
        
        image_tensor = self.transform_img(sliced_img)
#         import pdb; pdb.set_trace()
#         target = transforms.Resize((256,256))(image_tensor)
#         target = transforms.Compose([transforms.ToTensor(), transforms.Resize((256,256))])(image_tensor)
        
        label = curr['category_id'] 
        return self.transform_bbox(image_tensor), label

    
    def __len__(self):
        return len(self.data) # <----
    

    
# Encoder

class Encoder(nn.Module):

    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()

        self.encoder_cnn =  models.resnet18(pretrained=True)
        self.encoder_cnn.fc = torch.nn.Identity(512, 512)

        self.encoder_lin = nn.Sequential(
                                            nn.Linear(512, 256),
                                            nn.ReLU(True),
                                            nn.Linear(256, 128)
                                        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_lin(x)
        return x

# Decoder blocks
class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode) # <------- !!!
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Decoder(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=128, nc=1):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec,  64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec,  64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, 32, kernel_size=3, scale_factor=2)
        self.conv2 = ResizeConv2d(32, 16, kernel_size=3, scale_factor=2)
        self.conv3 = ResizeConv2d(16,  3, kernel_size=3, scale_factor=2)


    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sigmoid(self.conv3(x))
        return x # x.squeeze()

# Encoder 50

# Encoder

class Encoder50(nn.Module):

    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()

        self.encoder_cnn =  models.resnet50(pretrained=True)
        self.encoder_cnn.fc = torch.nn.Identity(2048, 2048)

        self.encoder_lin = nn.Sequential(nn.Linear(2048, 512))
                                        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_lin(x)
        return x


class Decoder50(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=512, nc=1):
        super().__init__()
        self.in_planes = 1024 #2048

        self.linear = nn.Linear(z_dim, 1024) #

#         self.layer4 = self._make_layer(BasicBlockDec, 1024, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 512, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec,  256, num_Blocks[1], stride=2)
#         self.layerA = self._make_layer(BasicBlockDec,  128, num_Blocks[1], stride=2)
        self.layerB = self._make_layer(BasicBlockDec,  256, num_Blocks[0], stride=1)
#         self.layer1 = self._make_layer(BasicBlockDec,  64, num_Blocks[0], stride=1)
        self.conv0 = ResizeConv2d(256, 128, kernel_size=3, scale_factor=4) # 256, 128
        self.conv1 = ResizeConv2d(128, 64, kernel_size=3, scale_factor=2)
        self.conv2 = ResizeConv2d(64, 3, kernel_size=3, scale_factor=2)
#         self.conv3 = ResizeConv2d(16,  3, kernel_size=3, scale_factor=2)


    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
#         import pdb; pdb.set_trace()
        x = self.linear(z)
        x = x.view(z.size(0), 1024, 1, 1) # 2048
        x = F.interpolate(x, scale_factor=4)
#         x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
#         x = self.layerA(x)
        x = self.layerB(x)
#         x = self.layer1(x)
        x = self.conv0(x)
        x = self.conv1(x)
#         x = self.conv2(x)
        x = torch.sigmoid(self.conv2(x))
        return x # x.squeeze()

def train(device, encoder_model, decoder_model, loader, criterion, optimizer):
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)
    
    encoder_model.train()
    decoder_model.train()
    
    loss_list = []
    for batch_idx, (bbox_tensor, label) in enumerate(loader, 0):
        
        bbox_tensor = bbox_tensor.to(device) 
        
        # zero grad
        optimizer.zero_grad()
        
        # embed
        enc = encoder_model(bbox_tensor)
        dec = decoder_model(enc)
    
        # calculate loss
#         import pdb; pdb.set_trace()
        loss = criterion(dec, torch.sigmoid(bbox_tensor) )
    
        # back prop
        loss.backward()
        optimizer.step()
        
        # statistics
        template = "Iteration {} ({:3.1f}%): Loss = {:.8f}\r"
        loss_list.append(loss.item()) 
        percentage = 100*batch_idx/len(loader)
        
        loss_mean = np.mean(loss_list)
        
        print(template.format(batch_idx, percentage, loss_mean ), end='')
    
    print('')
    return encoder_model, decoder_model, loss_list

def val(device, encoder_model, decoder_model, loader, criterion, optimizer):
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)
    
    encoder_model.eval()
    decoder_model.eval()
    
    loss_list = []
    for batch_idx, (bbox_tensor, label) in enumerate(loader, 0):
        
        bbox_tensor = bbox_tensor.to(device)
        
        # zero grad
        optimizer.zero_grad()
        
        # embed
        enc = encoder_model(bbox_tensor)
        dec = decoder_model(enc)
    
        # calculate loss
        loss = criterion(dec, torch.sigmoid(bbox_tensor))
        
        # statistics
        template = "Val iteration {} ({:3.1f}%): Loss = {:.8f}\r"
        loss_list.append(loss.item()) 
        percentage = 100*batch_idx/len(loader)
        
        loss_mean = np.mean(loss_list)

        
        print(template.format(batch_idx, percentage, loss_mean ), end='')
    
    print('')
    return encoder_model, loss_list, loss_mean

# =============================================================+
# deprecated classes or functions

class eval_dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform_img, transform_bbox):
        """
        Args:
            data: images dataframes
            transforms: preprocess transforms
        """
        self.data = data
        self.transform_img = transform_img
        self.transform_bbox = transform_bbox
    
    def __getitem__(self, index): 

        curr  = self.data.iloc[index]
        img =  Image.open(curr['img_path'])
        bbox = curr['bbox']
        sliced_img = TB.bbox_croper(img, bbox) # list of images
        
        try:
            image_tensor = self.transform_img(sliced_img)
        except: import pdb; pdb.set_trace()
#         target = transforms.Resize((256,256))(image_tensor)
        
        label = curr['category_id'] 
        return self.transform_bbox(image_tensor), label

    
    def __len__(self):
        return len(self.data) 
    