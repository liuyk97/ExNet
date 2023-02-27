import argparse
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchinfo import summary
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.utils.data import ConcatDataset
import logging
from utils.helpers import get_logger, get_criterion, EarlyStopping, find_gpu, seed_torch, save_images
from torch.utils.data import DataLoader
from utils.metrics import averageMeter, RunningMetrics_CD
from Dataloader.CD_dataset import LEVID_CDset, WHU_CDset, SYSU_CDset, CDD_set
import tqdm
from model.Net import Style_Net, style_transfer, CD_Net
from tqdm import tqdm
from utils.losses import hybrid_loss

# CUDA_VISIBLE_DEVICES= python -m torch.distributed.launch --nproc_per_node=4 train_CD.py
# ssh -L 12580:127.0.0.1:12580 liuyikun@211.87.232.115


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='LEVIR-CD')
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lrS', type=float, default=1e-4)
parser.add_argument('--max_iterS', type=int, default=200000)
parser.add_argument('--epochC', type=int, default=200)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--batch_sizeS', type=int, default=8)
parser.add_argument('--resume_model', default='')
parser.add_argument('--resume_decoder', default='')
parser.add_argument('--style_trans', action='store_true')
parser.add_argument('--generate_images', default=False)
parser.add_argument('--print_interval', default=100)
parser.add_argument('--style_path', default='/data/sdu08_lyk/data/style_cd/output/')
parser.add_argument('--loss_function', default='dice')
args = parser.parse_args()

seed_torch(args.seed)

Note = 'CFCF'
print(Note)
gpu_id = find_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

logdir = 'logs/{}/{}'.format(args.dataset, Note)
if not os.path.exists(logdir):
    os.makedirs(logdir)
if args.style_path and not os.path.exists(args.style_path):
    os.makedirs(args.style_path)
logger = get_logger(logdir)
logger.info(args.__dict__)
# writer = SummaryWriter(logdir)

print("----Loading Datasets----")
if args.dataset == 'LEVIR-CD':
    train_set = LEVID_CDset(mode='train')
    print("get {} images from LEVIR-CD train set".format(len(train_set)))
    val_set = LEVID_CDset(mode='val')
    print("get {} images from LEVIR-CD val set".format(len(val_set)))
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
val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=2, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2, shuffle=False)

# model initialization
print("----Model Initialization----")
Style_Net = CD_Net().to(device)
logger.info(summary(Style_Net, [(args.batch_size, 3, 256, 256), (args.batch_size, 3, 256, 256)]))
optimizer = torch.optim.SGD(Style_Net.parameters(), lr=args.lr, weight_decay=2e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)
criterion = get_criterion(args.loss_function)
running_metrics_val = RunningMetrics_CD()
best = -1
best_epoch = -1
# Style_Net = Style_Net().to(device)
# logger.info(summary(Style_Net,
#                     [(args.batch_size, 3, 256, 256), (args.batch_size, 3, 256, 256), (args.batch_size, 1, 256, 256)], dtypes=[torch.float, torch.float, torch.long]))

if args.resume_model:
    CD_Net.load_state_dict(torch.load(args.resume_model))
    print(f'Resume model:{args.resume_model}')

if args.style_trans:
    concat_set = ConcatDataset([train_set, val_set, test_set])
    concat_loader = iter(DataLoader(concat_set, batch_size=args.batch_sizeS, num_workers=2, shuffle=False))

    if args.resume_decoder:
        Style_Net.decoder.load_state_dict(torch.load(args.resume_decoder))
        print(f'Resume style decoder:{args.resume_decoder}')

    optimizerS = torch.optim.Adam(Style_Net.decoder.parameters(), lr=args.lrS)
    # schedulerS = torch.optim.lr_scheduler.StepLR(optimizerS, step_size=20, gamma=0.5)
    print("Start Style transform!")
    Style_Net.train()
    mean_loss = torch.zeros(1).to(device)
    pbar = tqdm(concat_loader, unit='pair')
    for step in range(args.max_iterS):
        pbar.desc = "[Iter {}/{}]".format(step, args.max_iterS)
        data = next(concat_loader)
        image_A = data['A'].cuda()
        image_B = data['B'].cuda()
        optimizerS.zero_grad()

        loss_c, loss_s = Style_Net(image_A, image_B)

        loss = loss_c + loss_s
        loss.backward()
        optimizerS.step()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
        pbar.postfix = "loss_c {}, loss_s {}, mean_loss {}".format(round(loss_c.item(), 3), round(loss_s.item(), 3),
                                                                   round(mean_loss.item(), 3))
        if (step + 1) % args.print_interval == 0:
            print_str = "[iter {}/{}] loss_c {}, loss_s {}, mean_loss {}".format(step + 1, args.max_iterS,
                                                                                 round(loss_c.item(), 3),
                                                                                 round(loss_s.item(), 3),
                                                                                 round(mean_loss.item(), 3))
            logger.info(print_str)
    checkpoint = Style_Net.decoder.state_dict()

    save_path = os.path.join(logdir, "best_decoder.pth")
    torch.save(checkpoint, save_path)
# generate and cd
tqdm.write("Start Change detection!")
for epoch in range(args.epochC):
    Style_Net.train()
    mean_loss = torch.zeros(1).to(device)
    pbar = tqdm(train_loader, unit='pair')
    pbar.desc = "[Epoch {}/{}]".format(epoch, args.epochC)
    for step, data in enumerate(pbar):
        image_A = data['A'].cuda()
        image_B = data['B'].cuda()
        lbl = data['L'].long().cuda()

        optimizer.zero_grad()

        loss_cd, loss_c = Style_Net(image_A, image_B, lbl)
        # loss = criterion(pred, lbl)
        loss = loss_cd + loss_c
        loss.backward()
        optimizer.step()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
        pbar.postfix = "loss_cd {}, loss_c {}, mean_loss {}".format(round(loss_cd.item(), 3), round(loss_c.item(), 3), round(mean_loss.item(), 3))
        if (step + 1) % args.print_interval == 0:
            print_str = "[epoch {}/{} iter:{}] loss_cd {}, loss_c {}, mean_loss {}".format(epoch, args.epochC, step + 1, round(loss_cd.item(), 3), round(loss_c.item(), 3), round(mean_loss.item(), 3))
            logger.info(print_str)
    scheduler.step(mean_loss)

    # evaluation
    with torch.no_grad():
        Style_Net.eval()
        torch.cuda.empty_cache()
        ebar = tqdm(test_loader, unit='pair')
        ebar.desc = "[Evaluation]".format(epoch, args.epochC)
        for idx, val_data, in enumerate(ebar):
            image_A = val_data['A'].cuda()
            image_B = val_data['B'].cuda()
            label = val_data['L'].long().cuda()
            path = val_data['path']
            if args.generate_images:
                # tqdm.write("Generate Styled Images!")
                with torch.no_grad():
                    image_A = style_transfer(Style_Net.encode, Style_Net.decoder, image_A, image_B)

                save_images(image_A.cpu(), args.style_path, path)

            pred = Style_Net(image_A, image_B)
            out = pred.max(1)[1].detach().cpu().numpy()
            lbl = label.data.detach().cpu().numpy()
            running_metrics_val.update(lbl, out)
        score = running_metrics_val.get_scores()

        for k, v in score.items():
            logger.info('{}: {}'.format(k, round(v, 4)))
        if score['F1'] >= best:
            best = score['F1']
            best_epoch = epoch
            checkpoint = Style_Net.state_dict()
            save_path = os.path.join(logdir, "best_model.pth")
            torch.save(checkpoint, save_path)
            running_metrics_val.reset()
        print_str = "F1: {}({}) / Best F1: {}({})".format(round(score['F1'], 4), epoch, round(best, 4), best_epoch)
        tqdm.write(print_str)
        logger.info("Best F1: {}(epoch:{})".format(best, epoch))
        time.sleep(1)
    torch.cuda.empty_cache()
