!pip install timm scikit-learn opencv-python kaggle matplotlib seaborn albumentations -q

import os, shutil, random, time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import timm
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# DRIVE
from google.colab import drive
drive.mount('/content/drive')

BASE = "/content/drive/MyDrive/capstone"
os.makedirs(BASE, exist_ok=True)

# KAGGLE
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# DATASET
RAW = "/content/dataset_raw"

!kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p /content -q
!unzip -q /content/cifake-real-and-ai-generated-synthetic-images.zip -d $RAW

print("Dataset loaded")

# BALANCED SPLIT
SPLIT = "/content/dataset_split"
if os.path.exists(SPLIT):
    shutil.rmtree(SPLIT)

classes = ["REAL", "FAKE"]
splits = ["train", "val", "test"]
ratios = (0.7, 0.85)

for s in splits:
    for c in classes:
        os.makedirs(f"{SPLIT}/{s}/{c}", exist_ok=True)

def get_images(cls):
    imgs = []
    for r,_,f in os.walk(RAW):
        if cls.lower() in r.lower():
            for x in f:
                if x.lower().endswith((".jpg",".png",".jpeg",".webp")):
                    imgs.append(os.path.join(r,x))
    return imgs

def split(cls):
    imgs = get_images(cls)
    random.shuffle(imgs)

    n = len(imgs)
    t1 = int(n*ratios[0])
    t2 = int(n*ratios[1])

    data = {
        "train": imgs[:t1],
        "val": imgs[t1:t2],
        "test": imgs[t2:]
    }

    for s,items in data.items():
        for i,src in enumerate(tqdm(items)):
            dst = f"{SPLIT}/{s}/{cls}/{i}.jpg"
            shutil.copy2(src,dst)

for c in classes:
    split(c)

print("Split done")

# AUGMENTATION ENGINE
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_tf = A.Compose([
    A.Resize(224,224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.CoarseDropout(p=0.3),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2()
])

val_tf = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2()
])

class AlbDataset(torch.utils.data.Dataset):
    def __init__(self, root, tf):
        self.ds = torchvision.datasets.ImageFolder(root)
        self.tf = tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        path, label = self.ds.samples[i]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tf(image=img)["image"]
        return img, label

train_ds = AlbDataset(SPLIT+"/train", train_tf)
val_ds   = AlbDataset(SPLIT+"/val", val_tf)
test_ds  = AlbDataset(SPLIT+"/test", val_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)

classes = ["REAL","FAKE"]

#  MODELS

cnn = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
cnn.classifier[1] = nn.Linear(cnn.classifier[1].in_features, 2)
cnn = cnn.to(device)

vit = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=2).to(device)
swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=2).to(device)

# TRAINING ENGINE 

loss_fn = nn.CrossEntropyLoss()

def train(model, loader, epochs=3, name="model"):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best = 0

    for ep in range(epochs):

        start_time = time.time()   #  START TIMER

        model.train()
        total = 0

        loop = tqdm(loader, desc=f"{name} Epoch {ep+1}/{epochs}")

        for x,y in loop:
            x,y = x.to(device), y.to(device)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out,y)
            loss.backward()
            opt.step()

            total += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        correct=tot=0

        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device)
                pred=model(x).argmax(1)
                correct+=(pred==y).sum().item()
                tot+=y.size(0)

        acc = correct/tot

        end_time = time.time()   # ⭐ END TIMER
        epoch_time = end_time - start_time

        print(f"\n{name} ep {ep+1} acc {acc:.3f} | ⏱ {epoch_time:.2f} sec")

        if acc>best:
            best=acc
            torch.save(model.state_dict(), f"{BASE}/{name}.pth")

    print("BEST:", best)

train(cnn, train_loader, 5, "cnn")
train(vit, train_loader, 3, "vit")
train(swin, train_loader, 3, "swin")

# 🔥 SIMCLR 

class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model("resnet18", pretrained=True, num_classes=0)
        self.projector = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )

    def forward(self,x):
        return self.projector(self.encoder(x))

simclr = SimCLR().to(device)

def nt_xent(z1, z2, temp=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.mm(z1, z2.T) / temp
    labels = torch.arange(z1.size(0)).to(device)

    return F.cross_entropy(logits, labels)

def train_simclr(model, epochs=2):
    opt = torch.optim.Adam(model.parameters(),1e-4)

    for ep in range(epochs):
        for x,_ in train_loader:

            x = x.to(device)

            x1 = x + 0.1*torch.randn_like(x)
            x2 = x + 0.1*torch.randn_like(x)

            z1 = model(x1)
            z2 = model(x2)

            loss = nt_xent(z1,z2)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print("SimCLR ep",ep+1)

train_simclr(simclr)

#  FFT + HYBRID 

def fft(x):
    feats=[]
    for i in x:
        i=i.permute(1,2,0).cpu().numpy()
        i=(i*0.5+0.5)*255
        g=cv2.cvtColor(i.astype(np.uint8),cv2.COLOR_RGB2GRAY)
        g=cv2.resize(g,(64,64))
        f=np.fft.fft2(g)
        m=np.abs(f)
        feats.append([m.mean(),m.std(),m.max(),np.log1p(m).mean()])
    return torch.tensor(feats,dtype=torch.float32).to(device)

class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
        self.fc = nn.Sequential(
            nn.Linear(384+4,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )

    def forward(self,x,f):
        v=self.vit(x)
        return self.fc(torch.cat([v,f],dim=1))

hybrid = Hybrid().to(device)

def train_hybrid(model, epochs=3):
    opt = torch.optim.AdamW(model.parameters(),1e-4)

    for ep in range(epochs):
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            f=fft(x)

            opt.zero_grad()
            out=model(x,f)
            loss=loss_fn(out,y)
            loss.backward()
            opt.step()

        print("Hybrid ep",ep+1)

train_hybrid(hybrid)

#  ENSEMBLE + EVAL (UNCHANGED)

def ensemble_predict(x):
    with torch.no_grad():
        c = torch.softmax(cnn(x),1)
        v = torch.softmax(vit(x),1)
        s = torch.softmax(swin(x),1)
        return (c+v+s)/3

def eval_model(model,is_hybrid=False):
    model.eval()
    yt,yp=[],[]

    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device)

            if is_hybrid:
                f=fft(x)
                o=model(x,f)
            else:
                o=model(x)

            yp.extend(o.argmax(1).cpu().numpy())
            yt.extend(y.numpy())

    print(classification_report(yt,yp,target_names=classes))
    print(confusion_matrix(yt,yp))

eval_model(cnn)
eval_model(vit)
eval_model(swin)
eval_model(hybrid,True)

print("DONE")
 
