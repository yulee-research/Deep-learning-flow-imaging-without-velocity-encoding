import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange

import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,sampler,Dataset
import torchvision.transforms as T
import numpy as np
from torchvision import models
from Swin_Transformer_3D import PatchEmbed3D, BasicLayer, PatchMerging
from Downsample_Upsample import DoubleConv_3d, Down_3d, Up_3d, OutConv_3d

class Multiscale_perception_block(nn.Module):
    def __init__(self,i_depth,in_channel,out_channel,
                 window_size=(5,5,5),
                 patch_size=(1,2,2),
                 in_chans=1,
                 embed_dim=128,
                 depths=[2, 4, 6],
                 num_heads=[4, 8, 16],                 
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False
                ):
        super().__init__()
        
        self.in_channel = in_channel // 2
        self.out_channel = out_channel // 2
        self.depth = i_depth
        
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.swin = BasicLayer(
                dim=int(embed_dim * 2**i_depth),
                depth=depths[i_depth],
                num_heads=num_heads[i_depth],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_depth]):sum(depths[:i_depth + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_depth>0 else None,
                use_checkpoint=use_checkpoint)
        
        if i_depth == 0:
            self.conv = nn.Sequential(
                            DoubleConv_3d(1, 32),
                            Down_3d(in_channel // 2, out_channel // 2)
                        )
            
        else:
            self.conv = Down_3d(in_channel // 2, out_channel // 2)
        
        if i_depth == 2:
            self.conv1 = nn.Conv3d(out_channel,out_channel // 2,1)
        else:
            self.conv1 = None
            
        self.dense = nn.Linear(out_channel,out_channel)                
        
    def forward(self,x):
        if self.depth == 0:
            up = x
            down = self.pos_drop(self.patch_embed(x))        
        else:
            up = x[:,0:self.in_channel,:,:,:]
            down = x[:,self.in_channel:self.in_channel * 2,:,:,:]            
        y_up = self.conv(up)
        y_down = self.swin(down)
        
        if self.conv1:
            y_down = self.conv1(y_down)
        y = torch.cat([y_up,y_down],1)        
        y = rearrange(y, 'b c d h w -> b d h w c')
        y = self.dense(y)
        y = rearrange(y, 'b d h w c -> b c d h w')
        return y



class SwinTransformer3D_UNet(nn.Module):
    def __init__(self,
                 patch_size=(2,2,2),
                 in_chans=1,
                 embed_dim=128,
                 depths=[2, 4, 6],
                 num_heads=[4, 8, 16],
                 window_size=(3,3,3),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,                
                downsample=PatchMerging if i_layer>0 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)
        
        factor = 2
        bilinear = True
        self.inc = DoubleConv_3d(1, 128)
        self.down1 = Multiscale_perception_block(0,64,256)
        self.down2 = Multiscale_perception_block(1,256,512)
        self.down3 = Multiscale_perception_block(2,512,512)        
                
        self.up2 = Up_3d(1024, 512 // factor, bilinear)
        self.up3 = Up_3d(512, 256 // factor, bilinear)
        self.up4 = Up_3d(256, 64, bilinear)
        self.outc = OutConv_3d(64, 1)                       

    def init_weights(self):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        
        raw = x
        x1 = self.inc(raw)        
        x2 = self.down1(raw)        
        x3 = self.down2(x2)
        x4 = self.down3(x3)        
        
        x = self.up2(x4, x3)            
        x = self.up3(x, x2)        
        x = self.up4(x, x1)        
        recon = self.outc(x)
        
        return recon