from torch import nn as nn
from .psrt_sliding_arch import SwinIRFM
from .alignment_modules_v5 import ADMA
import torch



class ZoomSEM(nn.Module):
    def __init__(self,
                 in_channels=1,
                 mid_channels=64,
                 embed_dim=120,
                 depths=(6, 6, 6),
                 num_heads=(6, 6, 6),
                 window_size=(3, 8, 8),
                 num_frames=3,
                 img_size = 32,
                 patch_size=1,
                 cpu_cache_length=100,
                 is_low_res_input=True,
                 upscale=16):

        super().__init__()
        self.scale = upscale
        self.align_module = ADMA(out_channels=embed_dim)

        self.mid_channels = mid_channels
        self.embed_dim = embed_dim
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.conv_before_upsample = nn.Conv2d(embed_dim, mid_channels, 3, 1, 1)
        self.img_size = img_size
        self.patch_size = patch_size

        # feature extraction module
        if is_low_res_input:
            #self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
            self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # propagation branches
        self.patch_align = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.patch_align[module] = SwinIRFM(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_channels,
                    embed_dim=embed_dim,
                    depths=depths,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=2.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm,
                    ape=False,
                    patch_norm=True,
                    use_checkpoint=False,
                    upscale=16,
                    img_range=1.,
                    upsampler='pixelshuffle',
                    resi_connection='1conv',
                    num_frames=num_frames)


        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv4 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def propagate(self, feats, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propgated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        t = len(feats['spatial']) - 1

        frame_idx = range(0, t + 1)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]#[0,1,..,t-1,t-1,...0]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]

        #feat_prop = flows.new_zeros(n, self.embed_dim, h, w)#n,c,h,w;zeors
        for i, idx in enumerate(frame_idx):
            if module_name == 'backward_1':
                feat_current = feats['spatial'][mapping_idx[idx]]#t-1,..,0
            if module_name == 'forward_1':
                feat_current = feats['backward_1'][mapping_idx[idx]]
            if module_name == 'backward_2':
                feat_current = feats['forward_1'][mapping_idx[idx]]
            if module_name == 'forward_2':
                feat_current = feats['backward_2'][mapping_idx[idx]]
            # second-order deformable alignment
            if i > 0:
                cond_n1 = self.align_module(feat_current, feat_prop)

                # initialize second-order features
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    cond_n2 = self.align_module(feat_current, feat_n2)


                cond = torch.stack([cond_n1, feat_current, cond_n2], dim=1)
                # patch alignment
                feat_prop = self.patch_align[module_name](cond)

            if i == 0:
                cond = torch.stack([feat_current, feat_current, feat_current], dim=1)
                feat_prop = self.patch_align[module_name](cond)#独立的SwinIRFM

            feats[module_name].append(feat_prop)#t list:b,c,h,w

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = feats['forward_2'][i]
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.conv_before_upsample(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv3(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv4(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)



    def forward(self, lqs):
        n, t, c, h, w = lqs.size()

        self.cpu_cache = False

        feats = {}
        # compute spatial features

        feats_ = self.conv_first(lqs.view(-1, c, h, w))
        h, w = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h, w)
        feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs


        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []

                feats = self.propagate(feats, module)
                if self.cpu_cache:
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)

