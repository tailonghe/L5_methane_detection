import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from time import sleep
import numpy as np
from torchvision.models import resnext50_32x4d
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def zstandard(arr):
    '''
    h, w
    '''
    _mu = np.nanmean(arr)
    _std = np.nanstd(arr)
    
    return (arr - _mu)/_std

def dice_coef_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union


def dice_coef_loss(inputs, target):
    smooth = 1.0
    intersection = 2.0 * ((target * inputs).sum()) + smooth
    union = target.sum() + inputs.sum() + smooth

    return 1 - (intersection / union)


def bce_dice_loss(inputs, target):
    dicescore = dice_coef_loss(inputs, target)
    bcescore = nn.BCELoss()
    bceloss = bcescore(inputs, target)

    return bceloss + dicescore


def train_model(model_name, model, train_loader, val_loader, train_loss, optimizer, num_epochs):  
    
    print(model_name)
    loss_history = []
    train_history = []
    val_history = []
    # best_perform = 0
    
    for epoch in range(num_epochs):
        model.train() # Enter train mode
        
        losses = []
        train_iou = []

        
        # with tqdm(train_loader, unit="batch") as tepoch:
            # for data, target in tepoch:
        for i_step, (data, target) in enumerate(train_loader):

            # tepoch.set_description(f"Epoch {epoch}")

            data = data.to(device)
            target = target.to(device)
            # print(data.shape, target.shape)
            outputs = model(data)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.33)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.33)] = 1.0

            train_dice = dice_coef_metric(out_cut, target.data.cpu().numpy())

            loss = train_loss(outputs, target)

            losses.append(loss.item())
            train_iou.append(train_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if lr_scheduler:
            #     lr_scheduler.step()

            # tepoch.set_postfix(loss=loss.item(), accuracy=train_dice)
            # sleep(0.1)

        val_mean_iou = compute_iou(model, val_loader)

        # if val_mean_iou > best_perform:
        #     score_str = '%.2f'%(val_mean_iou)
        #     best_perform = val_mean_iou

        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_iou)

        print("Epoch [%d]" % (epoch))
        print("Mean loss on train:", np.array(losses).mean(), 
              "\nMean DICE on train:", np.array(train_iou).mean(), 
              "\nMean DICE on validation:", val_mean_iou)
        
    return loss_history, train_history, val_history


def compute_iou(model, loader, threshold=0.33):
    """
    Computes accuracy on the dataset wrapped in a loader
    
    Returns: accuracy as a float value between 0 and 1
    """
    #model.eval()
    valloss = 0
    
    with torch.no_grad():

        for i_step, (data, target) in enumerate(loader):
            
            data = data.to(device)
            target = target.to(device)
            #prediction = model(x_gpu)
            
            outputs = model(data)
           # print("val_output:", outputs.shape)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            picloss = dice_coef_metric(out_cut, target.data.cpu().numpy())
            valloss += picloss

        #print("Threshold:  " + str(threshold) + "  Validation DICE score:", valloss / i_step)

    return valloss / i_step

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding):
        super().__init__()

        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convrelu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,
                                          stride=2)
        self.conv1 = ConvRelu(out_channels, out_channels, 3, padding='same')
        self.conv2 = ConvRelu(out_channels, out_channels, 3, padding='same')

    def forward(self, x):
        x = self.deconv(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x
    
class PlumeNet(nn.Module):

    def __init__(self, in_channels, out_channels, n_classes):
        super().__init__()
        
        self.base_model = resnext50_32x4d(pretrained=True)
        self.base_layers = [nn.Conv2d(in_channels, 64, (7, 7), stride=(2, 2), padding=(3, 3), bias=True)] + list(self.base_model.children())[1:]
        # self.base_layers = list(self.base_model.children())
        filters = [64, 128, 256, 512]
        
        # Down
        self.encoder0 = nn.Sequential(*self.base_layers[:3])
        self.encoder1 = nn.Sequential(*self.base_layers[4])
        self.encoder2 = nn.Sequential(*self.base_layers[5])
        self.encoder3 = nn.Sequential(*self.base_layers[6])
        self.encoder4 = nn.Sequential(*self.base_layers[7])

        self.neck1 = ConvRelu(2048, 1024, 3, padding='same')
        self.neck2 = ConvRelu(1024, 512, 3, padding='same')
        # Up
        self.decoder4 = DecoderBlock(2560, 512)
        self.decoder3 = DecoderBlock(1536, 256)
        self.decoder2 = DecoderBlock(768, 128)
        self.decoder1 = DecoderBlock(384, 64)

        # Final Classifier
        self.last_conv0 = ConvRelu(64, 64, 3, 1)
        self.last_conv1 = nn.Conv2d(64, n_classes, 3, padding=1)
                       
        
    def forward(self, x):
        # Down
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        n1 = self.neck1(e4)
        n2 = self.neck2(n1)
        # Up + sc
        # print(e4.shape, e3.shape, e2.shape, e1.shape)
        d4 = self.decoder4(torch.cat((n2, e4), 1)) # 3072, 1024
        # print(torch.cat((d4, e3), 1).shape)
        d3 = self.decoder3(torch.cat((d4, e3), 1))
        # print(torch.cat((d3, e2), 1).shape)
        d2 = self.decoder2(torch.cat((d3, e2), 1))
        d1 = self.decoder1(torch.cat((d2, e1), 1))
        # print(d1.shape)

        # final classifier
        out = self.last_conv0(d1)
        out = self.last_conv1(out)
        out = torch.sigmoid(out)
        
        return out
    
    
    
def transform(image):
    '''
    0.dr, 1.red, 2.green, 
    3.blue, 4.nir, 5.swir1, 
    6.swir2, 7.cloudscore, 8.bg, 
    9.gs, 10.ndvi, 11.zdrmbg, 12.zdrmbg2
    (H, W, 9)
    '''
    # dR, red, green, blue, nir, swir1, swir2, cloudscore, bg, gs, ndvi, zdrmbg, zdrmbg2
#         model = HuberRegressor().fit(self.dr.flatten().reshape(-1, 1), self.bg.flatten().reshape(-1, 1))
#         b0 = model.coef_[0] #This slope is dR/bg        
#         dRmbg = zstandard( (self.dr - self.bg*b0) )
#         dRmbg = np.clip(dRmbg, -4, 4)
#         self.zdrmbg = dRmbg # zstandard(self.dr) - zstandard(self.bg)
    
#         self.zdrmbg2 = zstandard(self.dr - self.bg)

    dr = image[:, :, 0]
    dr[dr < -1] = -1
    dr = zstandard(dr)
    # red = image[:, :, 1]
    # green = image[:, :, 2]
    # blue = image[:, :, 3]
    # nir = image[:, :, 4]
    bg = image[:, :, 8]
    bg[bg<-1] = -1
    bg = -zstandard(bg)
    
    zdrmbg1 = image[:, :, 11]
    zdrmbg2 = image[:, :, 12]
    gs = zstandard(image[:, :, 9])
    ndvi = -zstandard(image[:, :, 10])
    
    
    # ndvi = (nir - red)/(nir + red)
    # ndvi = zstandard(ndvi)
    
#     model = HuberRegressor().fit(dr.flatten().reshape(-1, 1), bg.flatten().reshape(-1, 1))
#     b0 = model.coef_[0] #This slope is dR/bg        
#     dRmbg = zstandard( (dr - bg*b0) )
#     dRmbg = np.clip(dRmbg, -4, 4)
    
#     zdrmbg2 = zstandard(dr) - zstandard(bg)
    
#     gs = 0.3*red + 0.59*green + 0.11*blue
#     gs = zstandard(gs)
    
    return np.stack([dr, bg, zdrmbg1, zdrmbg2, ndvi, gs])
    
class PlumeDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # dr, drmbg, ndvi, gs, bg
        image = np.load(self.df.path.iloc[idx])['pred']  # H, W, F  # x.dr, x.bg, x.zdrmbg, x.ndvi, x.red, x.green, x.blue
        mask = np.load(self.df.path.iloc[idx])['mask']   # H, W, 1
        
        _image = transform(image)
        _image[np.where(np.isnan(_image))] = 0
        
        mask = np.squeeze(mask)
        mask = mask[np.newaxis, :, :]
        mask[np.where(np.isnan(mask))] = 0
        
        
        return _image.astype(np.float32), mask.astype(np.float32)
    