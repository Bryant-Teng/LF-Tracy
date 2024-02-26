import os
import random
import numpy
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import numpy as np
import scipy.io as sio
import cv2
from PIL import ImageEnhance


def cv_random_flip(img, label, focal):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(focal.shape[2]):
            focal[:,:,i] = numpy.flip(focal[:,:,i],1)
    return img, label, focal

def randomCrop(image, label, focal):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    W1 = (image_width - crop_win_width) >> 1
    H1 = (image_height - crop_win_height) >> 1
    W2 = (image_width + crop_win_width) >> 1
    H2 = (image_height + crop_win_height) >> 1
    random_region = (W1, H1, W2, H2)  
    focal_crop = focal[H1:H2, W1:W2, :]
    image = image.crop(random_region)
    label = label.crop(random_region)
    return image, label, focal_crop

def randomRotation(image, label, focal):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        angle = [90,180,270]
        random_angle = random.choice(angle)
        if random_angle == 90:
            m = 1
        elif random_angle == 180:
            m = 2
        else:
            m = 3
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        for i in range(focal.shape[2]):
            focal[:, :, i] = np.rot90(focal[:, :, i],m)
    return image, label, focal

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)

class SalObjDataset(data.Dataset):
    mean_rgb = np.array([0.485, 0.456, 0.406])
    std_rgb = np.array([0.229, 0.224, 0.225])
    mean_focal = np.tile(mean_rgb, 12)
    std_focal = np.tile(std_rgb, 12)
    def __init__(self, image_root, gt_root, focal_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.focals = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.mat')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.focals = sorted(self.focals) 
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        focal = self.focal_loader(self.focals[index])


        image, gt, focal = cv_random_flip(image, gt, focal)
        image, gt, focal = randomRotation(image, gt, focal)
        image, gt, focal = randomCrop(image, gt, focal)
        image = colorEnhance(image)
        gt = randomPeper(gt)

        image = self.img_transform(image) 
        gt = self.gt_transform(gt) 

        focal = np.array(focal, dtype=np.int32)
        if focal.shape[0] != 256:
            new_focal = []
            focal_num = focal.shape[2] // 3
            for i in range(focal_num):
                a = focal[:, :, i * 3:i * 3 + 3].astype(np.uint8)
                a = cv2.resize(a, (256, 256))
                new_focal.append(a)
            focal = np.concatenate(new_focal, axis=2)  

        focal = focal.astype(np.float64)/255.0
        focal -= self.mean_focal
        focal /= self.std_focal
        focal = focal.transpose(2, 0, 1)
        focal = torch.from_numpy(focal).float() 
        return image, gt, focal
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def focal_loader(self, path):
        with open(path, 'rb') as f:
            focal = sio.loadmat(f)
            focal = focal['img'] 
            return focal

    def resize(self, img, gt, focal):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), focal
        else:
            return img, gt, focal

    def __len__(self):
        return self.size

class test_dataset:
    mean_rgb = np.array([0.485, 0.456, 0.406])
    std_rgb = np.array([0.229, 0.224, 0.225])
    mean_focal = np.tile(mean_rgb, 12)
    std_focal = np.tile(std_rgb, 12)
    def __init__(self, image_root, gt_root, focal_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.focals = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.mat')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.focals = sorted(self.focals)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0
    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)  
        gt = self.binary_loader(self.gts[self.index]) 
        focal = self.focal_loader(self.focals[self.index]) 
        focal = np.array(focal, dtype=np.int32) 
        if focal.shape[0] != 256:
            new_focal = []
            focal_num = focal.shape[2] // 3
            for i in range(focal_num):
                a = focal[:, :, i * 3:i * 3 + 3].astype(np.uint8)
                a = cv2.resize(a, (256, 256))
                new_focal.append(a)
            focal = np.concatenate(new_focal, axis=2)

        focal = focal.astype(np.float64)/255.0
        focal -= self.mean_focal
        focal /= self.std_focal
        focal = focal.transpose(2, 0, 1)
        focal = torch.from_numpy(focal).float()

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, focal, gt, name
    def focal_loader(self, path):
        with open(path, 'rb') as f:
            focal = sio.loadmat(f)
            focal = focal['img']
            return focal

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


