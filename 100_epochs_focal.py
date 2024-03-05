# %%
import torch
from torch import nn
import numpy as np

# %%
#you can create a Layernorm function or class here in future
class LayerNorm2d(nn.LayerNorm):
  def forward(self,x):
    x = x.permute(0, 2, 3, 1)
    x = super().forward(x)
    x = x.permute(0, 3, 1, 2)

    return x

# %%
class OverlapPatchMerging(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False
            ),
            LayerNorm2d(out_channels)
        )


# %%
#mix-ffn layer
class Mix_FFN(nn.Module):
  '''
  dense layer followed by convolution layer with gelu activation then another
  dense layer
  '''
  def __init__(self, channels, expansion=4):
    super().__init__()
    self.channels = channels
    self.expansion = expansion
    self.dense1 = nn.Conv2d(channels, channels, kernel_size=1)
    self.conv = nn.Conv2d(channels,
                              channels*expansion,
                              kernel_size=3,
                              groups= channels,
                              padding=1)
    self.gelu= nn.GELU()
    self.dense2 = nn.Conv2d(channels*expansion, channels, kernel_size=1)

  def forward(self,x):
    x=self.dense1(x)
    x=self.conv(x)
    x=self.gelu(x)
    x=self.dense2(x)

    return x

# %%
#multi headed attention block
class Rearrange(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)

class RearrangeBack(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class EfficientMultiHeadedAttention(nn.Module):
  def __init__(self, channels, reduction_ratio, num_heads):
    super().__init__()
    self.reducer = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=reduction_ratio, stride=reduction_ratio),
        Rearrange(),  # Transpose tensor
        nn.LayerNorm(channels),  # Use nn.LayerNorm directly
        RearrangeBack(),  # Transpose tensor back
    )
    self.att = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

  def forward(self, x):
    batch_size, chnls, h, w = x.shape
    reduced_x = self.reducer(x)
    # attention needs tensor of shape (batch, sequence_length, channels)
    reduced_x = reduced_x.reshape(reduced_x.size(0), -1, reduced_x.size(1))
    # print(reduced_x.shape)
    x = x.reshape(x.size(0), -1, x.size(1))
    # print(x.shape)
    out = self.att(x, reduced_x, reduced_x)[0]
    # print(out.shape)
    # reshape it back to (batch, channels, height, width)
    out = out.reshape(out.size(0), out.size(2), h, w)
    out = out.reshape(out.size(0), out.size(1), h, w)

    # print(out.shape)
    return out



# x = torch.randn((1, 8, 64, 64))
# block = EfficientMultiHeadedAttention(8, 4, 8)
# block(x).shape

# %%
#helper classes
from torchvision.ops import StochasticDepth

class ResidualAdd(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x, **kwargs):
    out = self.fn(x, **kwargs)
    x=x+out
    return x



# %%
#segfromer encoder class
# encoder block for the seg former

class SegFormerEncoderBlock(nn.Sequential):
  def __init__(
      self,
      channels: int,
      reduction_ratio: int = 1,
      num_heads: int = 8,
      mlp_expansion: int = 4,
      drop_path_prob: float = 0.0,
  ):

    super().__init__(
        ResidualAdd(
            nn.Sequential(
                LayerNorm2d(channels),
                EfficientMultiHeadedAttention(channels, reduction_ratio, num_heads),
            )
        ),
        ResidualAdd(
            nn.Sequential(
                LayerNorm2d(channels),
                Mix_FFN(channels, expansion = mlp_expansion),
                StochasticDepth(p=drop_path_prob, mode="batch")
            )
        ),
    )

# x=torch.randn((1,8,64,64))
# block = SegFormerEncoderBlock(8, reduction_ratio=4)
# block(x).shape





# %%
from typing import List

class SegFormerEncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)

# %%
#another helper function
from typing import Iterable

def chunks(data: Iterable, sizes):
  curr=0
  for size in sizes:
    chunk = data[curr: curr+size]
    curr+=size
    yield chunk

# %%
class SegFormerEncoder(nn.Module):
  def __init__(
      self,
      in_channels,
      widths,
      depths,
      all_num_heads,
      patch_sizes,
      overlap_sizes,
      reduction_ratios,
      mlp_expansions,
      drop_prob = 0.0,
  ):
    super().__init__()

    drop_probs = [x.item() for x in torch.linspace(0,drop_prob,sum(depths))]
    self.stages = nn.ModuleList(
        [
            SegFormerEncoderStage(*args)
            for args in zip(
                [in_channels, *widths],
                widths,
                patch_sizes,
                overlap_sizes,
                chunks(drop_probs, sizes=depths),
                depths,
                reduction_ratios,
                all_num_heads,
                mlp_expansions
            )
        ]
    )

  def forward(self,x):
    features = []
    for stage in self.stages:
      x=stage(x)
      features.append(x)
    return features

# %%

class DecoderBlock(nn.Sequential):
  def __init__(self, in_channels, out_channels, scale_factor: int =2):
    super().__init__(
    nn.UpsamplingBilinear2d(scale_factor = scale_factor),
    nn.Conv2d(in_channels, out_channels, kernel_size = 1),
    )

# %%
class Decoder(nn.Module):
  def __init__(self, out_channels:int , widths: List[int], scale_factors: List[int]):
    super().__init__()
    self.stages = nn.ModuleList(
        [
            DecoderBlock(in_channels, out_channels, scale_factor)
            for in_channels, scale_factor in zip(widths, scale_factors)
        ]
    )

  def forward(self, features):
    new_features = []
    for feature, stage in zip(features, self.stages):
      x=stage(feature)
      new_features.append(x)

    return new_features

# %%
class SegmentationHead(nn.Module):
  def __init__(self, channels, num_classes, num_features = 4):
    super().__init__()
    self.channels = channels
    self.num_classes = num_classes
    self.num_features = num_features
    self.dense1 = nn.Conv2d(channels*num_features, channels, kernel_size=1, bias=False)
    self.relu = nn.ReLU()
    self.bn = nn.BatchNorm2d(channels)
    self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)
    self.upscale = nn.UpsamplingBilinear2d(scale_factor=4)

  def forward(self,x):
    x=torch.cat(x, dim=1)
    x=self.dense1(x)
    x=self.relu(x)
    x=self.bn(x)
    x=self.predict(x)
    x=self.upscale(x)

    return x


# %%
class SegFormer(nn.Module):
    def __init__(
        self,
        in_channels,
        widths,
        depths,
        all_num_heads,
        patch_sizes,
        overlap_sizes,
        reduction_ratios,
        mlp_expansions,
        decoder_channels,
        scale_factors,
        num_classes,
        drop_prob : float = 0.0,
    ):

        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = Decoder(decoder_channels, widths[::-1], scale_factors)
        self.seghead = SegmentationHead(
            decoder_channels, num_classes, num_features=len(widths)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.seghead(features)
        return segmentation

# %%
# test_segformer = SegFormer(
#     in_channels=3,
#     widths=[64, 128, 256, 512],
#     depths=[3, 4, 6, 3],
#     all_num_heads=[1, 2, 4, 8],
#     patch_sizes=[7, 3, 3, 3],
#     overlap_sizes=[4, 2, 2, 2],
#     reduction_ratios=[8, 4, 2, 1],
#     mlp_expansions=[4, 4, 4, 4],
#     decoder_channels=256,
#     scale_factors=[8, 4, 2, 1],
#     num_classes=20,
# )



# %%
# test_data = torch.randn((10, 3, 224, 224))
# # print(test_data.shape)

# features=test_segformer(test_data)
# print(features.shape)

# %%
# from torchsummary import summary
# summary(test_segformer, (3,224,224))

# %%
from torchvision.datasets import Cityscapes #premade dataloader for cityscapes
import matplotlib.pyplot as plt
from PIL import Image

# %%
# dataset  = Cityscapes('../cityscapes/', split='train', mode='fine', target_type='semantic')

# %%
# fig, ax = plt.subplots(ncols=2, figsize=(12,8))
# ax[0].imshow(dataset[0][0])
# ax[1].imshow(dataset[0][1], cmap='gray')

# %%
ignore_index = 255
void_classes= [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30,-1]
valid_classes= [ignore_index, 7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
class_names = ['unlabeled',
               'road',
               'sidewalk',
               'building',
               'wall',
               'fence',
               'pole',
               'traffic light',
               'traffic sign',
               'vegetation',
               'terrain',
               'sky',
               'person',
               'rider',
               'car',
               'truck',
               'bus',
               'train',
               'motorcycle',
               'bicycle']
class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes = len(valid_classes)
class_map

# %%
colors =[
    [  0,   0,   0],
    [128,  64, 128],
    [244,  35, 232],
    [ 70,  70,  70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170,  30],
    [220, 220,   0],
    [107, 142,  35],
    [152, 251, 152],
    [ 70, 130, 180],
    [220,  20,  60],
    [255,   0,   0],
    [  0,   0, 142],
    [  0,   0,  70],
    [  0,  60, 100],
    [  0,  80, 100],
    [  0,   0, 230],
    [119,  11,  32],
]

label_colors = dict(zip(range(n_classes), colors))

# %%
def encode_segmap(mask):
    '''
    online mila tha 
    remove unwanted classes and rectify the labels of wanted classes
    '''
    for void_c in void_classes:
        mask[mask == void_c] = ignore_index
    for valid_c in valid_classes:
        mask[mask == valid_c] = class_map[valid_c]
    
    return mask

# %%
def decode_segmap(temp):
    '''
    ye bhi online mila tha
    convert greyscale to color
    '''
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colors[l][0] 
        g[temp == l] = label_colors[l][1] 
        b[temp == l] = label_colors[l][2]
    
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = r/255.0 
    rgb[:,:,1] = g/255.0 
    rgb[:,:,2] = b/255.0 
    
    return rgb

# %%
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision

class AdjustGamma:
    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain
    
    def __call__(self, image, mask):
        img = np.transpose(image,(2,0,1))
        gamma_tensor = torchvision.transforms.functional.adjust_gamma(torch.from_numpy(img), self.gamma, self.gain)
        img = np.transpose(gamma_tensor.numpy(), (1,2,0))
        return {'image': img, 'mask': mask}

transform = A.Compose(
    [
        A.Resize(224,224),
        # A.HorizontalFlip(),
        AdjustGamma(gamma=0.65),
        A.Normalize(mean = (0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225), max_pixel_value = float(225)),
        ToTensorV2(),
    ]
)

# %%
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets import Cityscapes

class data_transform(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        
        targets: Any = []
        for i,t in enumerate(self.target_type):
            if t == 'polygon':
                target = self.load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]
        

        if self.transforms is not None :
            transformed=transform(image=np.array(image), mask=np.array(target))
            return transformed['image'], transformed['mask']
        return image, target

# %%
dataset = data_transform('./cityscapes/', split='val', mode='fine', target_type='semantic', transforms=transform)
img, seg = dataset[20]
print(img.shape, seg.shape)

# %%
# fig, ax = plt.subplots(ncols=2, figsize=(12,8))
# ax[0].imshow(img.permute(1,2,0))
# ax[1].imshow(seg, cmap='gray')

# %%
# import torch
# print(torch.unique(seg))
# print(len(torch.unique(seg)))

# %%
# res = encode_segmap(seg.clone())
# print(res.shape)
# print(torch.unique(res))
# print(len(torch.unique(res)))

# %%
# res1= decode_segmap(res.clone())

# %%
# import cv2

# fig, ax = plt.subplots(ncols=2, figsize=(12,8))
# ax[0].imshow(res1)
# ax[1].imshow(res, cmap='gray')

# %%
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


class SegFormerDataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = data_transform(root='./cityscapes/', split='train', mode='fine', target_type='semantic', transforms=transform)
        self.val_dataset = data_transform(root='./cityscapes/', split='val', mode='fine', target_type='semantic', transforms=transform)
        # print(self.train_dataset.shape)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)



# %%

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SegFormerModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SegFormer(
            in_channels=3,
            widths=[64, 128, 256, 512],
            depths=[3, 4, 6, 3],
            all_num_heads=[1, 2, 4, 8],
            patch_sizes=[7, 3, 3, 3],
            overlap_sizes=[4, 2, 2, 2],
            reduction_ratios=[8, 4, 2, 1],
            mlp_expansions=[4, 4, 4, 4],
            decoder_channels=256,
            scale_factors=[8, 4, 2, 1],
            num_classes=20,
        )
        # self.criterion = smp.losses.DiceLoss(mode='multiclass')
        self.criterion = smp.losses.FocalLoss(mode='multiclass')

        self.metrics = torchmetrics.JaccardIndex(num_classes=n_classes, task='multiclass')
    
    def forward(self,x):
        return self.model(x)
    
    def process(self, image, segment):
        out=self(image)
        segment = encode_segmap(segment)
        loss= self.criterion(out, segment.long())
        iou = self.metrics(out, segment)
        return loss, iou
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True, threshold=0.001)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor' : 'val_loss'}
    
    def training_step(self, batch, batch_idx):
        image, segment = batch
        loss, iou = self.process(image, segment)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, segment = batch
        loss, iou = self.process(image, segment)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    


# %%
model = SegFormerModel()
datamodule = SegFormerDataModule(batch_size=32)

checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', filename='file', save_last = True)
# Define the early stopping callback

from pytorch_lightning.callbacks import EarlyStopping
early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')


# %%
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

tb_logger = TensorBoardLogger("logs/", name = "SegFormer_v2_epoch501_FocalLoss")


trainer = Trainer(max_epochs=150,
                  accelerator="cuda" if torch.cuda.is_available() else "cpu",
                  callbacks=[checkpoint_callback, early_stop_callback],
                  num_sanity_val_steps=0,
                  logger = tb_logger
                  )

# %%
trainer.fit(model, datamodule=datamodule)
# Loading the best model from checkpoint
best_model = SegFormerModel.load_from_checkpoint(checkpoint_callback.best_model_path)

# Assuming you have trained your model and it's stored in the variable `best_model`

# Define the file path where you want to save the model weights
weights_path = "segformer_100epochs_model_weightsFocalLoss.pth"

# Save the model weights
# torch.save(best_model.state_dict(), weights_path)

# Optionally, you can also save the entire model
# torch.save(best_model, 'entire_model.pth')


# %% [markdown]
# RUN THE CELLS AFTER THIS ONLY AFTER COMPLETION OF TRAINING  (ESTIMATED TRAINING COMPLETION TIME 9:00 am MORNING)

# %%
# %reload_ext tensorboard
# %tensorboard --logdir=logs/tensorboard/ 

# %%
# Evaluate the model on validation dataset
datamodule.setup()
val_loader = datamodule.val_dataloader()

pred_masks = []
gt_masks = []
val_images = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for val_batch in val_loader:
    val_image, val_segment = val_batch
    val_image = val_image.to(device)
    pred_mask = best_model.model(val_image).argmax(dim=1).cpu().numpy()
    pred_masks.append(pred_mask)
    gt_masks.append(val_segment)
    val_images.append(val_image.to(device))

# %%
print(gt_masks[0].shape)
print(pred_masks[0].shape)
print(val_images[0].shape)

import os
temp_folder = 'FocalLoss'
os.makedirs(temp_folder, exist_ok=True)

# Visualizing predicted masks
num_samples_to_visualize = 5  # Choose the number of samples to visualize
for i in range(num_samples_to_visualize):
    plt.figure(figsize=(12, 8))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    original_img = val_images[i][0].cpu().permute(1, 2, 0).numpy()
    plt.imshow(original_img)
    plt.title("original_img")

    # Plot ground truth
    plt.subplot(1, 3, 2)
    ground_truth = gt_masks[i][0].cpu().numpy()  # Assuming gt_masks is a tensor
    plt.imshow(ground_truth)
    plt.title('ground truth')

    # Plot predicted mask
    plt.subplot(1, 3, 3)
    predicted_mask = torch.tensor(pred_masks[i][0]).cpu().numpy()  # Assuming pred_masks is a numpy array
    plt.imshow(predicted_mask)
    plt.title('Predicted Mask')

    # Save the figure
    plt.savefig(os.path.join(temp_folder, f'image_{i + 1}.png'))
    
    
#image still not encoded so segment caolors will not match i guess


# %%



