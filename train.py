import argparse
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from utils.copy import deepcopy
from utils.helpers import get_logger, get_criterion, get_scheduler, EarlyStopping, find_gpu, \
    seed_torch
from torch.utils.data import DataLoader
from utils.metrics import averageMeter, RunningMetrics_CD
from Dataloader.CD_dataset import LEVID_CDset, WHU_CDset, SYSU_CDset, CDD_set
import tqdm
from model.net import CDNet
from tqdm import tqdm

# CUDA_VISIBLE_DEVICES= python -m torch.distributed.launch --nproc_per_node=4 train_CD.py
# ssh -L 12580:127.0.0.1:12580 liuyikun@211.87.232.115


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='LEVIR-CD')
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_name', default='linear')
parser.add_argument('--lr_lrf', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--optimizer', default="SGD")
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--loss_function', default='hybrid')
parser.add_argument('--resume_model', default='')
args = parser.parse_args()

seed_torch(args.seed)

Note = 'l2'
gpu_id = find_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

logdir = 'logs/{}/{}'.format(args.dataset, Note)
if not os.path.exists(logdir):
    os.makedirs(logdir)
logger = get_logger(logdir, Note)

# writer = SummaryWriter(logdir)

# print_info = "abstract:\n\t classifier: {}\n\t dataset: {}\n\t size: {}\n\t batch_size: {}\n\t lr: {}\n\t gpu_id: {
# }".format( classifier, dataset, size, batch_size, lr, gpu_id)
print("----Loading Datasets----")
if args.dataset == 'LEVIR-CD':
    train_set = LEVID_CDset(mode='train')
    print("get {} images from LEVIR-CD train set".format(len(train_set)))
    test_set = LEVID_CDset(mode='test')
    print("get {} images from LEVIR-CD test set".format(len(test_set)))

elif args.dataset == 'WHU-CD':
    train_set = WHU_CDset(mode='train')
    print("get {} images from WHU_CD train set".format(len(train_set)))
    test_set = WHU_CDset(mode='test')
    print("get {} images from WHU_CD test set".format(len(test_set)))

elif args.dataset == 'CDD':
    train_set = CDD_set(mode='train')
    print("get {} images from CDD train set".format(len(train_set)))
    test_set = CDD_set(mode='test')
    print("get {} images from CDD test set".format(len(test_set)))

elif args.dataset == 'SYSU-CD':
    train_set = SYSU_CDset(mode='train')
    print("get {} images from SYSU-CD train set".format(len(train_set)))
    test_set = SYSU_CDset(mode='test')
    print("get {} images from SYSU-CD test set".format(len(test_set)))
train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=2, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size * 6, num_workers=2, shuffle=False)
train_iters = len(train_loader) * args.epochs

# model initialization
print("----Model Initialization----")

model = CDNet(args)

if args.resume_model:
    model = torch.load(args.resume_model)
    model = model.module
    print(f'Resume model:{args.resume_model}')
    model_name = 'resume'
model.to(device)

optimizer = torch.optim.SGD(model.decoder.parameters(), lr=args.lr,
                            momentum=0.9,
                            weight_decay=5e-4)
scheduler = get_scheduler(optimizer, args)
criterion = get_criterion(args.loss_function)

running_metrics_val = RunningMetrics_CD()
time_meter = averageMeter()
print("Start Training!")

best = -1
for epoch in range(args.start_epoch, args.epochs):

    model.train()
    mean_loss = torch.zeros(1).to(device)
    pbar = tqdm(train_loader, unit='pair')
    for step, data in enumerate(pbar):
        optimizer.zero_grad()

        image_A = data['A'].cuda()
        image_B = data['B'].cuda()
        lbl = data['L'].long().cuda()

        loss_c, loss_s = model(image_A, image_B, lbl)

        # cd_loss = criterion(pred, lbl)
        loss = loss_c + loss_s
        loss.backward()
        optimizer.step()
        pbar.desc = "[epoch {}/{}] loss_c {} loss_s {}".format(epoch, args.epochs, round(loss_c.item(), 3), round(loss_s.item(), 3))
    scheduler.step()

    # evaluation
    # with torch.no_grad():
    #     model.eval()
    #     torch.cuda.empty_cache()
    #     pbar = tqdm(test_loader, unit='pair')
    #     for idx, val_data, in enumerate(tqdm(pbar)):
    #         image_A = val_data['A'].cuda()
    #         image_B = val_data['B'].cuda()
    #         label = val_data['L'].cuda()
    #
    #         pred = model(image_A, image_B)
    #         out = pred.max(1)[1].detach().cpu().numpy()
    #         lbl = label.data.detach().cpu().numpy()
    #         running_metrics_val.update(lbl, out)
    #
    #     score = running_metrics_val.get_scores()
    #     for k, v in score.items():
    #         logger.info('{}: {}'.format(k, v))
    #
    #     if score['F1'] >= best:
    #         best = score['F1']
    checkpoint = {
        "epoch": epoch,
        "state_dict_backbone": deepcopy(model).state_dict(),
        "state_optimizer": optimizer.state_dict(),
        "state_lr_scheduler": scheduler.state_dict()
    }
    save_path = os.path.join(logdir,
                             "best_model.pkl")
    torch.save(checkpoint, save_path)
    #         running_metrics_val.reset()
    # torch.cuda.empty_cache()

print(Note)
