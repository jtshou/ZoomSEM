import os
import torch.nn as nn
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util.util as utils
import random
import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import yaml
from model.ZoomSEM_x8 import ZoomSEM
from util.util import OrderedYaml
Loader, Dumper = OrderedYaml()

class RealEMSR_recurrent_MultiSR_Val(data.Dataset):

    def __init__(self, opt):
        super(RealEMSR_recurrent_MultiSR_Val, self).__init__()
        self.opt = opt
        self.GT_dir = opt['GT_dir']
        self.LR_dir = opt['LR_dir']
        self.LR_input = True  # low resolution inputs
        self.GT_list = []
        self.LR_list = []
        self.Z_images = opt['Z_images']
        mod = 100 % self.Z_images
        for z in range(100, 128):
            for row in range(1):
                for column in range(1):
                    if (z - mod) % self.Z_images == 0:
                        image_name = 'z' + str(z + 1).zfill(3) + '_r{}_c{}.tif'.format(row + 1, column + 1)
                        GT_path = os.path.join(self.GT_dir, image_name)
                        if self.opt['phase'] == 'val':
                            self.GT_list.append(GT_path)


    def __getitem__(self, index):
        im_path_l = []
        image_begin_name = self.GT_list[index].split('/')[-1]
        z_num_begin = int(image_begin_name[1:4])
        z_num_end = z_num_begin + self.Z_images - 1
        if z_num_end > 128:
            z_num_end = 128
        img_LQ_l = []
        img_GT_l = []
        for z in range(z_num_begin,z_num_end+1):
            image_name = 'z' + str(z).zfill(3) + image_begin_name[4:]
            LQ_path = os.path.join(self.LR_dir, image_name)
            GT_path = os.path.join(self.GT_dir, image_name)
            img_LQ = util.read_img(None, LQ_path)
            img_GT = util.read_img(None, GT_path)
            img_LQ_l.append(img_LQ)
            img_GT_l.append(img_GT)
            im_path_l.append(GT_path.split('/')[-1])
        img_LQs = np.stack(img_LQ_l, axis=0)
        img_GTs = np.stack(img_GT_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2))).copy()).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2))).copy()).float()

        return {'LQ': img_LQs, 'GT': img_GTs, 'im_name':im_path_l}

    def __len__(self):
        count = len(self.GT_list)
        return count

def main():
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    im_path_SR = os.path.join(root_path, 'output_data', 'ZoomSEM_Acini_x8')
    assert not os.path.exists(im_path_SR)
    scale = 8
    pretrain_path = os.path.join(root_path, 'pretrained', 'ZoomSEM_FT_x8.pth')
    utils.mkdirs([im_path_SR])
    opt_path = os.path.join(root_path, 'option', 'train_ZoomSEM_x8_Acini.yml')



    #### Create test dataset and dataloader
    with open(opt_path, mode='r',encoding='utf-8') as f:
        opt = yaml.load(f, Loader=Loader)
    opt = utils.dict_to_nonedict(opt)
    dataset_dict = opt['datasets']
    dataset_dict['val']['scale'] = scale
    dataset_dict['val']['phase'] = 'val'
    for phase, dataset_opt in dataset_dict.items():
        if phase == 'val':
            test_set = RealEMSR_recurrent_MultiSR_Val(dataset_opt)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=False)
    model = ZoomSEM()


    device = torch.device("cuda")
    model.eval()
    model.load_state_dict(torch.load(pretrain_path), strict=True)
    model = model.to(device)
    with torch.no_grad():
        for data in test_loader:
            LQ, GT, im_path_l = data['LQ'], data['GT'], data['im_name']
            B, N, C, H, W = LQ.size()
            LQ = LQ.to(device)
            SR = model(LQ)
            SR = SR.cpu()
            for i in range(N):
                SR1 = SR[:,i,:,:,:]
                SR1 = utils.tensor2img(SR1)
                im_path_final = os.path.join(im_path_SR,im_path_l[i][0])
                utils.save_img(SR1, im_path_final)

if __name__=='__main__':
    main()
