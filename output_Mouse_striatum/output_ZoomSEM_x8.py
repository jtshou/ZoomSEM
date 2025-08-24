import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util.util as utils
import numpy as np
import torch
import torch.utils.data as data
import data.util as util
from model.ZoomSEM_x8 import ZoomSEM
import argparse
from tqdm import tqdm


class RealEMSR_recurrent_MultiSR_Val(data.Dataset):

    def __init__(self,start_idx=0, end_idx=None):
        super(RealEMSR_recurrent_MultiSR_Val, self).__init__()
        self.LR_dir = '../Mouse_striatum'
        self.LR_input = True  # low resolution inputs
        self.LR_list = []
        self.Z_images = 7
        z_delete = [27, 168, 950]
        self.z_delete = z_delete
        mod = 1
        for z in range(1, z_delete[0]):
            #for row in range(1,17):
            for row in range(5, 6):
                #for column in range(1,17):
                for column in range(5, 6):
                    if (z - mod) % self.Z_images == 0:
                        image_name = 'z' + str(z).zfill(3) + '_r{}_c{}.tif'.format(row, column)
                        LR_path = os.path.join(self.LR_dir, image_name)
                        self.LR_list.append(LR_path)
        mod = z_delete[0]+1
        for z in range(z_delete[0]+1,z_delete[1]):
            # for row in range(1,17):
            for row in range(5, 6):
                # for column in range(1,17):
                for column in range(5, 6):
                    if (z - mod) % self.Z_images == 0:
                        image_name = 'z' + str(z).zfill(3) + '_r{}_c{}.tif'.format(row, column)
                        LR_path = os.path.join(self.LR_dir, image_name)
                        self.LR_list.append(LR_path)
        mod = z_delete[1] + 1
        for z in range(z_delete[1] + 1, z_delete[2]):
            # for row in range(1,17):
            for row in range(5, 6):
                # for column in range(1,17):
                for column in range(5, 6):
                    if (z - mod) % self.Z_images == 0:
                        image_name = 'z' + str(z).zfill(3) + '_r{}_c{}.tif'.format(row, column)
                        LR_path = os.path.join(self.LR_dir, image_name)
                        self.LR_list.append(LR_path)
        mod = z_delete[2] + 1
        for z in range(z_delete[2] + 1, 971):
            # for row in range(1,17):
            for row in range(5, 6):
                # for column in range(1,17):
                for column in range(5, 6):
                    if (z - mod) % self.Z_images == 0:
                        image_name = 'z' + str(z).zfill(3) + '_r{}_c{}.tif'.format(row, column)
                        LR_path = os.path.join(self.LR_dir, image_name)
                        self.LR_list.append(LR_path)

        self.LR_list = self.LR_list[start_idx:end_idx] if end_idx is not None else self.LR_list[start_idx:]


    def __getitem__(self, index):
        im_path_l = []
        image_begin_name = self.LR_list[index].split('/')[-1]
        z_num_begin = int(image_begin_name[1:4])
        z_num_end = z_num_begin + self.Z_images - 1
        if z_num_begin < self.z_delete[0] and z_num_end>=self.z_delete[0]:
            z_num_end = self.z_delete[0]-1
        if z_num_begin < self.z_delete[1] and z_num_end>=self.z_delete[1]:
            z_num_end = self.z_delete[1]-1
        if z_num_begin < self.z_delete[2] and z_num_end>=self.z_delete[2]:
            z_num_end = self.z_delete[2]-1
        if z_num_end > 970:
            z_num_end = 970
        img_LQ_l = []
        for z in range(z_num_begin,z_num_end+1):
            image_name = 'z' + str(z).zfill(3) + image_begin_name[4:]
            LQ_path = os.path.join(self.LR_dir, image_name)
            img_LQ = util.read_img(None, LQ_path)
            img_LQ_l.append(img_LQ)
            im_path_l.append(LQ_path.split('/')[-1])
        ls_len = len(img_LQ_l)
        if ls_len<self.Z_images:
            for z in range(self.Z_images-ls_len):
                img_LQ_l.append(img_LQ.copy())
                im_path_l.append(LQ_path.split('/')[-1])
        num_valid = torch.tensor(ls_len)
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2))).copy()).float()

        return {'LQ': img_LQs, 'im_name':im_path_l,'num_valid':num_valid}

    def __len__(self):
        count = len(self.LR_list)
        return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=4, help='End index (exclusive)')
    args = parser.parse_args()
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    im_path_SR = os.path.join(root_path, 'output_data', 'ZoomSEM_striatum_x8')
    assert not os.path.exists(im_path_SR)
    pretrain_path = os.path.join(root_path,'pretrained','ZoomSEM_x8.pth')
    utils.mkdirs([im_path_SR])


    test_set = RealEMSR_recurrent_MultiSR_Val(args.start, args.end)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=9, shuffle=False, num_workers=8,
                                           pin_memory=True)
    model = ZoomSEM()


    device = torch.device("cuda")
    model.eval()
    model.load_state_dict(torch.load(pretrain_path), strict=True)
    model = model.to(device)


    with torch.no_grad():
        for data in tqdm(test_loader, desc='Inference'):
            LQ, im_path_l,num_valid = data['LQ'], data['im_name'],data['num_valid']
            B, N, C, H, W = LQ.size()
            LQ = LQ.to(device)
            SR = model(LQ)
            SR = SR.cpu()
            for b in range(B):
                valid = num_valid[b].item()
                for i in range(valid):
                    SR1 = SR[b,i,:,:,:]
                    im_path_cur = im_path_l[i][b]
                    SR1 = utils.tensor2img(SR1)
                    im_path_final = os.path.join(im_path_SR,im_path_cur)
                    utils.save_img(SR1, im_path_final)

if __name__=='__main__':
    main()
