# SAMPLE GENERATOR PYRAMID 2D PERIODIC
#
# Code for the texture synthesis method in:
# Ulyanov et al. Texture Networks: Feed-forward Synthesis of Textures and Stylized Images
# https://arxiv.org/abs/1603.03417
# Generator architecture fixed to 6 scales!
#
# Author: Jorge Gutierrez
# Creation:  22 Jan 2019
# Last modified: 22 Jan 2019
# Based on https://github.com/leongatys/PytorchNeuralStyleTransfer
import math
import numpy as np
import scipy.misc

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from helpers_said import *
from matplotlib import animation as animation
import matplotlib.pyplot as plt
 #%% quelques fonctions d'aide  

def egalhisto(r):
    shapeo=r.shape
    r2=r.reshape(-1)
    idxs=np.argsort(r2)
    r2[idxs]=np.arange(len(r2))/len(r2)
    return np.float32(r2.reshape(shapeo))

def redresse(t,trans=False,lam=0.5):
    if lam>1-lam:
        lam=1-lam
    if (not trans) or (lam<0.01):
        return t
    else:
        c=1/2*(lam)/(1-lam)
        mask0=t>1/2
        t[mask0]=1-t[mask0] 
        mask=(t<=lam)
        t[mask]=1/2*(1/(1-lam))*(1/lam)*t[mask]**2
        mask=(t>lam)*(t<1/2)
        t[mask]=c+(1/(1-lam))*(t[mask]-lam)
        t[mask0]=1-t[mask0]
        return t
#%creer une animation avec un tableau d'image

def creeanimation(imgs,nom='demo.mp4',retour=False,fps=30):
    dpi=100
    ims = []
    fig = plt.figure()
    for k in range(imgs.shape[0]):
        im = plt.imshow(imgs[k], animated=True)
        ims.append([im])
    if retour:
        for k in range(imgs.shape[0]-1,0,-1):
            im = plt.imshow(imgs[k], animated=True)
            ims.append([im])
     

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
    writer = animation.writers['ffmpeg'](fps=fps)

    ani.save(nom,writer=writer,dpi=dpi)


#%% DEFINITIONS DU RESEAU 

#generator's convolutional blocks 2D
class Conv_block2D(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, m=0.1):
        super(Conv_block2D, self).__init__()

        self.conv1 = nn.Conv2d(n_ch_in, n_ch_out, 3, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(n_ch_out, momentum=m)
        self.conv2 = nn.Conv2d(n_ch_out, n_ch_out, 3, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(n_ch_out, momentum=m)
        self.conv3 = nn.Conv2d(n_ch_out, n_ch_out, 1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(n_ch_out, momentum=m)

    def forward(self, x):
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        return x

#Up-sampling + batch normalization block
class Up_Bn2D(nn.Module):
    def __init__(self, n_ch):
        super(Up_Bn2D, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn = nn.BatchNorm2d(n_ch)

    def forward(self, x):
        x = self.bn(self.up(x))
        return x

class Pyramid2D(nn.Module):
    def __init__(self, ch_in=3, ch_step=8):
        super(Pyramid2D, self).__init__()

        self.cb1_1 = Conv_block2D(ch_in,ch_step)
        self.up1 = Up_Bn2D(ch_step)

        self.cb2_1 = Conv_block2D(ch_in,ch_step)
        self.cb2_2 = Conv_block2D(2*ch_step,2*ch_step)
        self.up2 = Up_Bn2D(2*ch_step)

        self.cb3_1 = Conv_block2D(ch_in,ch_step)
        self.cb3_2 = Conv_block2D(3*ch_step,3*ch_step)
        self.up3 = Up_Bn2D(3*ch_step)

        self.cb4_1 = Conv_block2D(ch_in,ch_step)
        self.cb4_2 = Conv_block2D(4*ch_step,4*ch_step)
        self.up4 = Up_Bn2D(4*ch_step)

        self.cb5_1 = Conv_block2D(ch_in,ch_step)
        self.cb5_2 = Conv_block2D(5*ch_step,5*ch_step)
        self.up5 = Up_Bn2D(5*ch_step)

        self.cb6_1 = Conv_block2D(ch_in,ch_step)
        self.cb6_2 = Conv_block2D(6*ch_step,6*ch_step)
        self.last_conv = nn.Conv2d(6*ch_step, 3, 1, padding=0, bias=True)

    def forward(self, z):

        y = self.cb1_1(z[5])
        y = self.up1(y)
        y = torch.cat((y,self.cb2_1(z[4])),1)
        y = self.cb2_2(y)
        y = self.up2(y)
        y = torch.cat((y,self.cb3_1(z[3])),1)
        y = self.cb3_2(y)
        y = self.up3(y)
        y = torch.cat((y,self.cb4_1(z[2])),1)
        y = self.cb4_2(y)
        y = self.up4(y)
        y = torch.cat((y,self.cb5_1(z[1])),1)
        y = self.cb5_2(y)
        y = self.up5(y)
        y = torch.cat((y,self.cb6_1(z[0])),1)
        y = self.cb6_2(y)
        y = self.last_conv(y)
        return y

# post processing for images
postpa = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1./255)),
        #add imagenet mean
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                            std=[1,1,1]),
        #turn to RGB
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        ])

postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    
    return t.numpy().transpose((1,2,0)) # CANAUX A LA FIN (torh les met au debut)

#%% DEFINIR UNE IMAGE EXEMPLE ET CHARGER LE RESEAU PRE-ENTRAINE
## exemple d'image
#image_exemple='BrickRound0122_1_seamless_S'
# image_exemple='BubbleMarbel'
# image_exemple='CRW_3241_1024'
#image_exemple='CRW_3444_1024'
# image_exemple='Pierzga_2006_1024'
#image_exemple='Scrapyard0093_1_seamless_S'
#image_exemple='TexturesCom_BrickSmallBrown0473_1_M_1024'
image_exemple='TexturesCom_FloorsCheckerboard0046_4_seamless_S_1024'
# image_exemple='TexturesCom_TilesOrnate0085_1_seamless_S'
#image_exemple='TexturesCom_TilesOrnate0158_1_seamless_S'
# image_exemple='bubble_1024'
#image_exemple='fabric_white_blue_1024'
# image_exemple='glass_1024'
# image_exemple='lego_1024'
# image_exemple='marbre_1024'
#image_exemple='metal_ground_1024'
# image_exemple='rouille_1024'

### SUR MON MAC
##models_folder='/Users/said/Nextcloud_maison/Boulot/TP3_MVA_DELIRES/Texturenet_Trained_models/'
## PC LINUX
#models_folder='/home/said/CODE/TP3_MVA_DELIRES/Texturenet_Trained_models/'
### sur la DSI 
models_folder='/media/lawliet/Unlimited Burst/Study/DELIRES/TD/TP3_DELIRES/Texturenet_Trained_models/'
model_folder=models_folder+image_exemple

image_originale_file='./images/'+image_exemple+'.png'
image_originale=np.float32(skio.imread(image_originale_file))
#viewimage(image_originale)

#load model
generator = Pyramid2D(ch_step=8)
generator.load_state_dict(torch.load(model_folder + '/params.pytorch',map_location='cpu'))
generator.cuda()

generator.eval()
for param in generator.parameters():
    param.requires_grad = False

#%% GENERATION D"UN SEUL EXEMPLE
#draw sample
n_input_ch = 3
sample_size = 256
n_samples = 1

sz = [sample_size /1,sample_size /2,sample_size /4,sample_size /8,sample_size /16,sample_size /32]
zk = [torch.rand(n_samples,n_input_ch,int(szk),int(szk)) for szk in sz]


        

z_samples = [Variable(z).cuda() for z in zk ]
sample = generator(z_samples)
#%% extraction de torch vers un tableau numpy
out_imgs=np.zeros([n_samples,sample_size,sample_size,n_input_ch])
for n in range(n_samples):
    single_sample = sample[n,:,:,:]
    out_imgs[n,:,:,:] = postp(single_sample.data.squeeze().cpu())

# scipy.misc.imsave(image_exemple + '.jpg', out_imgs[0])
# import sys
# sys.exit()
#############################################################
#%% Visualisation par GIMP   
for n in range(n_samples):
    viewimage(out_imgs[n],titre=('echan_%d'%n))


#%%
#% GENERATION de VIDEOS 
    
#draw sample
n_input_ch = 3
sample_size = 256
n_samples = 30

sz = [sample_size /1,sample_size /2,sample_size /4,sample_size /8,sample_size /16,sample_size /32]
zk = [torch.rand(n_samples,n_input_ch,int(szk),int(szk)) for szk in sz]


for k in zk:
    for t in range(1,n_samples-1):
        lam=t/(n_samples-1)
        k[t,:,:,:]= (1 - lam) * k[0,:,:,:] + lam * k[-1,:,:,:]
        

# pour reduire la consommation memoire on fait par packets de N images
Nimages=10 # pour simplifier le code prendre Nimages diviseur de n_samples
assert n_samples % Nimages ==0 , "Nimages pas diviseur de n_samples"
#%% boucle de génération
ideb=0
out_imgs=np.zeros([n_samples,sample_size,sample_size,n_input_ch])
while ideb<n_samples:
    print(ideb)
    z_samples = [Variable(z[ideb:ideb+Nimages]).cuda() for z in zk ]
    #Lancer le reseau
    sample = generator(z_samples)
    #% extraction de torch vers un tableau numpy
    
    for n in range(Nimages):
        single_sample = sample[n,:,:,:].cpu()
        out_imgs[ideb+n,:,:,:] = postp(single_sample.data.squeeze())
    ideb+=Nimages
#%% Visualisation par GIMP   
for n in range(n_samples):
    viewimage(out_imgs[n],titre=('echan_%d'%n))
    #input()

#%% CREER UNE VIDEO (Attention au nom de fichier!)
creeanimation(out_imgs,nom='/media/lawliet/Unlimited Burst/Study/DELIRES/TD/TP3_DELIRES/demo5.mp4',retour=True,fps=3)

#%%%
#  VIDEO avec deplacement
    
#draw sample
n_input_ch = 3
sample_size = 512
n_samples = 60

#sz = [sample_size /1,sample_size /2,sample_size /4,sample_size /8,sample_size /16,sample_size /32]
zk = [torch.rand(n_samples,n_input_ch,sample_size,sample_size)]


k=zk[0]

for t in range(1,n_samples-1):
    lam=t/(n_samples-1)
    posx=int(lam*400)
    posy=int(lam*400)
    k[t,:,:,:]=k[-1,:,:,:]
    k[t,:,posy:posy+100,posx:posx+100]=k[0,:,:100,:100]
    
zk=[zk[0]]
for k in range(1,6):
    pui=2**k
    zk.append(zk[0][:,:,::pui,::pui])
     
# pour reduire la consommation memoire on fait par packets de N images
Nimages=10 # pour simplifier le code prendre Nimages diviseur de n_samples
assert n_samples % Nimages ==0 , "Nimages pas diviseur de n_samples"
#%%
ideb=0
out_imgs=np.zeros([n_samples,sample_size,sample_size,n_input_ch])
while ideb<n_samples:
    print(ideb)
    z_samples = [Variable(z[ideb:ideb+Nimages]).cuda() for z in zk ]
    #Lancer le reseau
    sample = generator(z_samples)
    #% extraction de torch vers un tableau numpy
    
    for n in range(Nimages):
        single_sample = sample[n,:,:,:].cpu()
        out_imgs[ideb+n,:,:,:] = postp(single_sample.data.squeeze())
    ideb+=Nimages
#%%    

#%% Visualisation par GIMP   
for n in range(n_samples):
    viewimage(out_imgs[n],titre=('echan_%d'%n))
    #input()

#%% CREER UNE VIDEO (Attention au nom de fichier!)
creeanimation(out_imgs,nom='/media/lawliet/Unlimited Burst/Study/DELIRES/TD/TP3_DELIRES/demo5.mp4',retour=True,fps=3)
