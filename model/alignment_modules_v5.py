import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from basicsr.archs.arch_util import DCNv2Pack
from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.arch_util import flow_warp
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import constant_init


class AffineAlign(nn.Module):
    def __init__(self,dim,emb_dim=32,window_size_multiply_scale=128,scale=16,enlarge_rate=2):#尝试window12x16;8x16
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.scale = scale
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.window_size = window_size_multiply_scale//scale
        self.enlarge_rate = enlarge_rate
        self.window_shift = self.window_size*(self.enlarge_rate-1)//2
        self.localization = nn.Sequential(
            nn.Conv2d(2*dim, emb_dim, 1),
            #nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(emb_dim, emb_dim, 3,2,1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(emb_dim, emb_dim, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(self.window_size * self.window_size * emb_dim//16*self.enlarge_rate*self.enlarge_rate, 32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(32, 6)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def window_partition_overlap(self,x):
        N, C, H, W = x.shape
        windows = x.unfold(2, self.window_size*self.enlarge_rate, self.window_size).unfold(3, self.window_size*self.enlarge_rate, self.window_size)#B,C,H//size,W//size,h_size,w_size
        windows = windows.permute(0,2,3,1,4,5).contiguous().view(-1,C,self.window_size*self.enlarge_rate,self.window_size*self.enlarge_rate)
        return windows



    def forward(self, ref, supp, affine_return_flag=False):
        # ref: frame to be propagated.n,c,h,w
        # supp: frame propagated to.n,c,h,w
        assert ref.size() == supp.size()
        n, c, h, w = ref.size()
        # affine metrix calculation
        ref_supp = torch.cat([ref,supp],dim=1)
        ref_supp = F.pad(ref_supp,(self.window_shift,self.window_shift,self.window_shift,self.window_shift))
        ref_supp = self.window_partition_overlap(ref_supp)
        ref_supp = self.localization(ref_supp)
        _, c1, h1, w1 = ref_supp.size()
        ref_supp = ref_supp.view(-1,c1*h1*w1)
        theta_reverse = self.fc_loc(ref_supp)
        if affine_return_flag == True:
            theta_back = theta_reverse.view(-1, 2, 3).clone() # (n*num_win)*2*3
        theta_reverse = theta_reverse.view(-1, 2, 3).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,self.window_size,self.window_size,1,1,1)
        n2 = theta_reverse.shape[0]


        device = ref.device
        grid_h_win, grid_w_win, grid_ones_win = torch.meshgrid(
            torch.arange(0, self.window_size, device=device, dtype=ref.dtype),
            torch.arange(0, self.window_size, device=device, dtype=ref.dtype),
            torch.ones(1, device=device, dtype=ref.dtype))
        grid = torch.stack((grid_w_win, grid_h_win,grid_ones_win), 3).unsqueeze(0).unsqueeze(5).repeat(n2, 1, 1, 1, 1,1) # n*num_win,h_w, w_w, 1,3(wh1),注意顺序要为(wh1)
        grid.requires_grad = False
        grid_sample = torch.matmul(theta_reverse, grid).squeeze(5).squeeze(3).clamp(-self.window_shift,self.window_size + self.window_shift-1)#n*num_win,h_w, w_w, 2(wh)
        grid_sample = (grid_sample + self.window_shift)*2/(self.window_size*self.enlarge_rate-1)-1.0

        supp = F.pad(supp, (self.window_shift, self.window_shift, self.window_shift, self.window_shift))
        supp_affine = self.window_partition_overlap(supp) # n*num_win,c,h_w_enlarge, w_w_enlarge
        supp_affine = F.grid_sample(supp_affine, grid_sample,align_corners=True).view(n,h//self.window_size,w//self.window_size,c,self.window_size,self.window_size).permute(0,3,1,4,2,5).contiguous().view(n,c,h,w)
        if affine_return_flag == True:
            return supp_affine, theta_back
        else:
            return supp_affine




class ADMA(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.L1_down_1 = nn.Conv2d(out_channels,out_channels,3,2,1)#共用
        self.L1_down_2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)  # 共用
        self.L2_down_1 = nn.Conv2d(out_channels,out_channels,3,2,1)#共用
        self.L2_down_2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)  # 共用
        self.L2_fused_conv = nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1)

        self.L1_fused_conv = nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.deformable_align_module_affine = DeformableAlignModulewoFlow(self.out_channels, self.out_channels, 3,
                                                                      padding=1,
                                                                      deform_groups=8, max_residue_magnitude=10)
        self.deformable_align_module_L2 = DeformableAlignModulewoFlow(self.out_channels, self.out_channels, 3,
                                                                   padding=1,
                                                                   deform_groups=8, max_residue_magnitude=10)
        self.deformable_align_module_L1 = DeformableAlignModulewoFlow(self.out_channels, self.out_channels, 3,
                                                                      padding=1,
                                                                      deform_groups=8, max_residue_magnitude=10)

        self.affine_align = AffineAlign(dim=out_channels,scale=16,window_size_multiply_scale=64)



    def forward(self, ref, supp):
        L2_ref = self.lrelu(self.L1_down_2(self.lrelu(self.L1_down_1(ref))))
        L3_ref = self.lrelu(self.L2_down_2(self.lrelu(self.L2_down_1(L2_ref))))
        L2_supp = self.lrelu(self.L1_down_2(self.lrelu(self.L1_down_1(supp))))
        L3_supp = self.lrelu(self.L2_down_2(self.lrelu(self.L2_down_1(L2_supp))))
        L3_align = self.deformable_align_module_affine(L3_ref,self.affine_align(L3_ref,L3_supp))
        L3_align = F.interpolate(L3_align, scale_factor=2, mode='bilinear', align_corners=False)
        L2_align = self.deformable_align_module_L2(L2_ref, L2_supp)
        L2_align = F.interpolate(self.L2_fused_conv(torch.cat([L3_align, L2_align], dim=1)), scale_factor=2,mode='bilinear', align_corners=False)

        L1_align_dcn = self.deformable_align_module_L1(ref, supp)
        L1_fused = self.L1_fused_conv(torch.cat([L2_align,L1_align_dcn],dim=1))


        return L1_fused




class DeformableAlignModule(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(DeformableAlignModule, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, ref, supp, flow):
        supp_flow = flow_warp(supp, flow.permute(0, 2, 3, 1))
        cond = torch.cat((ref, supp_flow, flow), dim=1)
        cond = self.conv_offset(cond)
        offset1, offset2, mask = torch.chunk(cond, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((offset1, offset2), dim=1))
        offset = offset + flow.flip(1).repeat(1,offset.size(1) // 2, 1,1)
        mask = torch.sigmoid(mask)
        supp_align = modulated_deform_conv2d(supp, offset, mask, self.weight, self.bias,
                                             self.stride, self.padding,
                                             self.dilation, self.groups,
                                             self.deform_groups)
        return supp_align

class DeformableAlignModulewoFlow(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(DeformableAlignModulewoFlow, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, ref, supp):
        cond = torch.cat((ref, supp), dim=1)
        cond = self.conv_offset(cond)
        offset1, offset2, mask = torch.chunk(cond, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((offset1, offset2), dim=1))
        mask = torch.sigmoid(mask)
        supp_align = modulated_deform_conv2d(supp, offset, mask, self.weight, self.bias,
                                             self.stride, self.padding,
                                             self.dilation, self.groups,
                                             self.deform_groups)
        return supp_align
