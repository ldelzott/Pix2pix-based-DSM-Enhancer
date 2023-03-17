import os
from base_dataset import BaseDataset, get_params, get_transform
from image_folder import make_dataset
from PIL import Image, ImageStat
import random
import numpy as np
import torch.utils.data
#import matplotlib.pyplot as plt


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = opt.dataroot #os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        #print("dataset [%s] was created" % type(self.dataset).__name__)
        
    


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        
        # split AB image into A and B
        w, h = AB.size
        
        w2 = int(w / 2)
        #w3 = int(w / 3) * 2
        A = AB.crop((0, 0, w2, h)) # The LIDAR tile
        B = AB.crop((0, 0, w2, h)) # The stereo tile
        C = AB.crop((w2, 0, w, h))
        
        
        #plt.imshow(np.array(A), cmap="gray");plt.title("A"); plt.colorbar(); plt.show()
        #plt.imshow(np.array(B), cmap="gray");plt.title("B"); plt.colorbar(); plt.show()
        #plt.imshow(np.array(C), cmap="gray");plt.title("C"); plt.colorbar(); plt.show()

        
        w, h = A.size
        new_h = h
        new_w = w

        x = random.randint(0, np.maximum(0, new_w - self.opt.crop_size)) # opt.crop_size default is 256; thus x and y are in [0, 286-256]
        y = random.randint(0, np.maximum(0, new_h - self.opt.crop_size))
        flip = random.random() > 0.5
        transform_params = {'crop_pos': (x, y), 'flip': flip}
        
        
        
        # apply the same transform to both A and B
        A_transform = get_transform(self.opt, transform_params, grayscale=True)
        B_transform = get_transform(self.opt, transform_params, grayscale=True)
        C_transform = get_transform(self.opt, transform_params, grayscale=True, normalize_value=0.7)


        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)

        return {'A': A, 'B': B, 'C': C, 'A_paths': AB_path, 'B_paths': AB_path, 'C_paths': AB_path}
    
    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.AB_paths), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

