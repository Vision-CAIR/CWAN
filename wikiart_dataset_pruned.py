from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import scipy.io
import numpy as np
import random
import skimage
from skimage import filters
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

## Noise Augmentation
class AddNoise(object):
    """Rotate by one of the given angles."""
    def __init__(self, noise_type, **kwargs):
        self.noise_type = noise_type
        self.kwargs = kwargs

    def __call__(self, x):
        x = x/255
        return (skimage.util.random_noise(x, mode=self.noise_type, **self.kwargs)*255).astype(np.uint8)
        
class GaussFilt(object):
    def __call__(self, x):
        x = x/255
        return (filters.gaussian(x, multichannel=True)*255).astype(np.uint8) 

gauss_tfm = transforms.Compose([AddNoise('gaussian', mean=0, var=0.008),
                                  transforms.ToPILImage()
                                 ])

speckle_tfm = transforms.Compose([AddNoise('speckle', mean=0, var=0.008),
                                  transforms.ToPILImage()
                                 ])

poisson_tfm = transforms.Compose([AddNoise('poisson'),
                                  transforms.ToPILImage()
                                 ]) 
                                 
gauss_filt = transforms.Compose([GaussFilt(),
                                  transforms.ToPILImage()
                                 ])     

class wikiart_dataset_HR(Dataset):
    def __init__(self, opt, transform=None, root='./wikiart/images'):

        self.root = root #wikiartimages.zip unzipped
        self.images = []
        self.class_count = {}
        self.root = self.root
        self.classes  = [
            art_style_name
            for art_style_name in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, art_style_name))
        ]

        self.classes_to_num = {}
        for (i,x) in enumerate(self.classes):
            self.classes_to_num[x] = i 

        self.pruned_style_class = []
        for art_style in self.classes:
            
            count = 0
            for img_name in os.listdir(os.path.join(self.root, art_style)):
                if is_img(img_name):
                    fullpath = os.path.join(self.root, art_style, img_name)
                    self.images.append(WikiartImage(img_name, art_style, fullpath, self.classes_to_num[art_style]))
                    self.pruned_style_class.append([self.classes_to_num[art_style]])
                    count += 1
            self.class_count[art_style] = count
            print(art_style, len(os.listdir(os.path.join(self.root, art_style))))            

        opt.n_styles = len(self.classes)
        # No genre or style info is present, but setting to 1
        # so that it doesn't error out
        opt.n_genres = 1
        opt.n_artists = 1
        # opt.n_genres = len(np.unique(self.pruned_genre))
        # opt.n_artists = len(np.unique(self.pruned_artist))

        print('wikiart dataset contains %s art_styles'%(len(self.classes)))

        if opt.data_aug == 'matlab':
            print("Data Augmentation Used: ", opt.data_aug)
            # 50% time no aug, and 50% time one of the first four.
            self.additional_tfms = [gauss_tfm, speckle_tfm, poisson_tfm, gauss_filt, None, None, None, None]       

        self.opt = opt

    def __getitem__(self, index):
        
        self.transform = transforms.Compose([
            transforms.Resize((self.resolution,self.resolution), Image.BICUBIC),
            # transforms.CenterCrop(self.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        art_image = self.images[index]
        wikiartimage_path = art_image.path #os.path.join(self.root, self.pruned_filenames[index][0][0])
        wp = Path(wikiartimage_path)
        wikiartimage_path = os.path.join(self.root, wp.parents[0].name, wp.name)

        
        inputs = {}
        img = Image.open(wikiartimage_path).convert('RGB')

        if self.opt.data_aug == 'matlab':
            tfm_id = random.randint(0, len(self.additional_tfms)-1)
            if self.additional_tfms[tfm_id] is not None: #tfm_id != len(self.additional_tfms) - 1:
                # print('applying' , tfm_id, self.additrional_tfms[tfm_id])
                img = self.additional_tfms[tfm_id](np.array(img)) 

        img = self.transform(img)

        inputs['img'] = img
        inputs['art_style'] = art_image.art_style_num 
        inputs['genre'] = 0
        inputs['artist'] = 0

        return inputs

    def __len__(self):
        return len(self.images)

    def name(self):
        return 'wikiart_dataset_HR'

class WikiartImage:
    def __init__(self, image_name, art_style, path, art_style_num):
        self.image_name = image_name
        self.art_style = art_style
        self.path = path
        self.art_style_num = art_style_num

def is_img(filename):
    exts = {".jpg", "jpeg", ".png"}
    return any(filename.endswith(ext) for ext in exts)
