import os
import matplotlib.pyplot as plt
import tifffile
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
# from utils.load_label import color2label
import csv
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = '5'


def transform_train():
    transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
    ]
    return transforms.Compose(transform_list)


def transform():
    transform_list = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transform_list)


class LEVID_CDset(Dataset):
    def __init__(self, mode="train"):
        self.data_path = "/data/sdu08_lyk/data/LEVIR-CD_256x256/"
        self.mode = mode
        self.path = os.listdir(self.data_path + self.mode + '/A/')
        self.transform = transform()

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        img_A = Image.open(self.data_path + self.mode + '/A/' + path).convert('RGB')
        img_B = Image.open(self.data_path + self.mode + '/B/' + path).convert('RGB')
        lbl = Image.fromarray(
            np.array(Image.open(self.data_path + self.mode + '/label/' + path), dtype=np.uint32) / 255)
        A = self.transform(img_A)
        B = self.transform(img_B)
        L = self.transform(lbl)

        return {'A': A,
                'B': B,
                'L': L, 'path': path}


class SYSU_CDset(Dataset):
    def __init__(self, mode="train"):
        self.data_path = "/data/sdu08_lyk/data/SYSU-CD/"
        self.mode = mode
        self.path = os.listdir(self.data_path + self.mode + '/time1/')
        self.transform = transform()

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        img_A = Image.open(self.data_path + self.mode + '/time1/' + path).convert('RGB')
        img_B = Image.open(self.data_path + self.mode + '/time2/' + path).convert('RGB')
        lbl = Image.fromarray(
            np.array(Image.open(self.data_path + self.mode + '/label/' + path), dtype=np.uint32) / 255)
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        lbl = self.transform(lbl)

        return {'A': img_A,
                'B': img_B,
                'L': lbl, 'path': path}


class DSIFN_set(Dataset):
    def __init__(self, mode="train"):
        self.data_path = "/data/sdu08_lyk/data/DSIFN_256x256/"
        self.mode = mode
        self.path = os.listdir(self.data_path + self.mode + '/t1/')
        self.transform = transform()

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        path_lbl = path
        if self.mode == 'test':
            path_lbl = path_lbl.replace('jpg', 'tif')
        img_A = Image.open(self.data_path + self.mode + '/t1/' + path).convert('RGB')
        img_B = Image.open(self.data_path + self.mode + '/t2/' + path).convert('RGB')
        lbl = Image.fromarray(
            np.array(Image.open(self.data_path + self.mode + '/mask/' + path_lbl), dtype=np.uint32) / 255)
        A = self.transform(img_A)
        B = self.transform(img_B)
        L = self.transform(lbl)

        return {'A': A,
                'B': B,
                'L': L, 'path': path}


class CDD_set(Dataset):
    def __init__(self, mode="train"):
        self.data_path = "/data/sdu08_lyk/data/CDD/"
        self.mode = mode
        self.path = os.listdir(self.data_path + self.mode + '/A/')
        self.transform = transform()

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        # path_lbl = self.lbl[idx]
        img_A = Image.open(self.data_path + self.mode + '/A/' + path).convert('RGB')
        img_B = Image.open(self.data_path + self.mode + '/B/' + path).convert('RGB')
        lbl = Image.fromarray(
            np.array(Image.open(self.data_path + self.mode + '/OUT/' + path), dtype=np.uint32) / 255)

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        lbl = self.transform(lbl)

        return {'A': img_A,
                'B': img_B,
                'L': lbl, 'path': path}


class WHU_CDset(Dataset):
    def __init__(self, mode="train"):
        self.data_path = "/data/sdu08_lyk/data/Building_change_detection_dataset_add_256x256/"
        self.mode = mode
        self.path = os.listdir(self.data_path + '/2012/whole_image/{}/image/'.format(self.mode))
        self.transform = transform()

    def __len__(self):
        return len(self.lbl)

    def __getitem__(self, idx):
        path_2012 = self.path[idx]
        path_2016 = self.path[idx].replace('2012', '2016')
        path_change = self.path[idx].replace('2012_{}'.format(self.mode), 'change_label')
        img_A = Image.open(self.data_path + '/2012/whole_image/{}/image/'.format(self.mode) + path_2012).convert('RGB')
        img_B = Image.open(self.data_path + '/2016/whole_image/{}/image/'.format(self.mode) + path_2016).convert('RGB')
        lbl_A = tifffile.imread(self.data_path + '/2012/whole_image/{}/label/'.format(self.mode) + path_2012)
        lbl_B = tifffile.imread(self.data_path + '/2016/whole_image/{}/label/'.format(self.mode) + path_2016)
        lbl = tifffile.imread((self.data_path + '/change_label/{}/'.format(self.mode) + path_change))
        lbl_A = Image.fromarray(lbl_A / 255)
        lbl_B = Image.fromarray(lbl_B / 255)
        lbl = Image.fromarray(lbl / 255)

        A = self.transform(img_A)
        B = self.transform(img_B)

        L_A = self.transform(lbl_A)
        L_B = self.transform(lbl_B)
        L = self.transform(lbl)

        return {'A': A,
                'B': B,
                'L_A': L_A, 'L_B': L_B, 'L': L}


class SECONDset(Dataset):  # 标签部分不对
    def __init__(self, mode="train", model_name='fcn'):
        self.data_path = "/data/sdu08_lyk/data/SECOND/"
        self.mode = mode
        self.preprocess = 'resize_and_crop'
        self.load_size = 512
        self.angle = 15
        self.crop_size = 512

        # self.images = os.listdir(self.data_path + 'im' + self.date)
        # self.labels = os.listdir(self.data_path + 'label' + self.date)
        # self.change_label = os.listdir(self.data_path + 'change_label/')
        # self.conf_file = [i.replace(".JPG", "_conf_up.npy") for i in self.images]

        if self.mode == 'train':
            filename_csv = open("/data/sdu08_lyk/data/SECOND/train.csv", "r")
        else:
            filename_csv = open("/data/sdu08_lyk/data/SECOND/val.csv", "r")
        reader = csv.reader(filename_csv)
        self.filename = []
        for item in reader:
            if reader.line_num == 1:
                continue
            self.filename.append(item[1])
        filename_csv.close()

    def __getitem__(self, idx):
        img_name = self.filename[idx].split('/')[-1]
        lbl_name = self.filename[idx].split('/')[-1]
        img_A = Image.open(self.data_path + 'im1/' + self.filename[idx]).convert('RGB')
        img_B = Image.open(self.data_path + 'im2/' + self.filename[idx]).convert('RGB')
        lbl_A = np.array(Image.open(self.data_path + 'label1/' + self.filename[idx]), dtype=np.uint32)
        lbl_B = np.array(Image.open(self.data_path + 'label2/' + self.filename[idx]), dtype=np.uint32)
        lbl_A = np.array(color2label(lbl_A, dataset='SECOND'), dtype=np.uint32)
        lbl_B = np.array(color2label(lbl_B, dataset='SECOND'), dtype=np.uint32)
        change_map = Image.fromarray(np.array(np.where(lbl_A != lbl_B, 1, 0), dtype=np.uint32))
        lbl_A = Image.fromarray(lbl_A)
        lbl_B = Image.fromarray(lbl_B)
        transform_params = get_params(self.preprocess, self.angle, self.crop_size, self.load_size, img_A.size,
                                      mode=self.mode)
        transform = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                  mode=self.mode)

        A = transform(img_A)
        B = transform(img_B)

        transform_L = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                    method=InterpolationMode.NEAREST, normalize=False,
                                    mode=self.mode)

        L_A = transform_L(lbl_A)
        L_B = transform_L(lbl_B)
        change_map = transform_L(change_map)

        return {'A': A, 'paths_img': img_name,
                'B': B, 'paths_L': lbl_name,
                'L_A': L_A, 'L_B': L_B, 'change_map': change_map}

    def __len__(self):
        return len(self.filename)


if __name__ == '__main__':
    set = SYSU_CDset('test')
    loader = DataLoader(set, batch_size=1, num_workers=4, shuffle=True,
                        pin_memory=True)
    for idx, data in enumerate(loader):
        A = data['A'].squeeze().cpu().numpy().transpose(1, 2, 0)
        B = data['B'].squeeze().cpu().numpy().transpose(1, 2, 0)
        # L_A = data['L_A'].squeeze().cpu().numpy() * 255
        # L_B = data['L_B'].squeeze().cpu().numpy() * 255
        path = data['path']
        change_map = data['L'].squeeze().cpu().numpy() * 255
        # change_map_pred = np.where(L_A != L_B, 1, 0)
        plt.subplot(131)
        plt.axis('off')
        plt.imshow(A)
        plt.subplot(132)
        plt.axis('off')
        plt.imshow(B)
        # plt.subplot(233)
        # plt.axis('off')
        # plt.imshow(L_A)
        # plt.subplot(234)
        # plt.axis('off')
        # plt.imshow(L_B)
        plt.subplot(133)
        plt.axis('off')
        plt.imshow(change_map)
        # plt.subplot(144)
        # plt.axis('off')
        # plt.imshow(change_map_pred)
        plt.title(path)
        plt.show()
