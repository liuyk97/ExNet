import argparse
import os
import time
import torch
from torchinfo import summary
from tensorboardX import SummaryWriter
from utils.helpers import get_logger, EarlyStopping, find_gpu, seed_torch, PolynomialLR
from torch.utils.data import DataLoader
from utils.metrics import RunningMetrics_CD
from Dataloader.CD_dataset import LEVID_CDset, WHU_CDset, SYSU_CDset, CDD_set
import tqdm
from model.Net import CD_Net
from tqdm import tqdm

from colorama import Fore
# ssh -L 12580:127.0.0.1:12580 liuyikun@211.87.232.115

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='LEVIR-CD')
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochC', type=int, default=200)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--gpu_id', default='6')
parser.add_argument('--resume_model', default='')
parser.add_argument('--print_interval', default=100)
parser.add_argument('--loss_function', default='dice')
args = parser.parse_args()

seed_torch(args.seed)

Note = 'spex'
print(Note)
if not args.gpu_id:
    args.gpu_id = find_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

logdir = '/data/sdu08_lyk/data/style/logs/{}/{}'.format(args.dataset, Note)
if not os.path.exists(logdir):
    os.makedirs(logdir)
logger = get_logger(logdir)
logger.info(args.__dict__)
logger.info(Note)
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
train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
# val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)

# model initialization
print("----Model Initialization----")
CD_Net = CD_Net().to(device)
logger.info(summary(CD_Net, [(args.batch_size, 3, 256, 256), (args.batch_size, 3, 256, 256)], row_settings=["var_names"], col_names=["kernel_size", "output_size", "num_params", "mult_adds"],))
optimizer = torch.optim.SGD(CD_Net.parameters(), lr=args.lr, weight_decay=2e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=30, factor=0.05)
# scheduler = PolynomialLR(optimizer, max_iter=len(train_loader) * args.epochC, gamma=0.9)
running_metrics_val = RunningMetrics_CD()
best = -1
best_epoch = -1
if args.resume_model:
    CD_Net.load_state_dict(torch.load(args.resume_model))
    print(f'Resume model:{args.resume_model}')

# training
tqdm.write("Start Training!")
for epoch in range(args.epochC):
    CD_Net.train()
    mean_loss = torch.zeros(1).to(device)
    pbar = tqdm(train_loader, unit='pair')
    pbar.desc = "[Epoch {}/{}]".format(epoch + 1, args.epochC)
    for step, data in enumerate(pbar):
        image_A = data['A'].cuda()
        image_B = data['B'].cuda()
        label = data['L'].long().cuda()

        optimizer.zero_grad()

        loss_cd = CD_Net(image_A, image_B, label)
        loss = loss_cd
        loss.backward()
        optimizer.step()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
        cur_lr = optimizer.param_groups[-1]['lr']
        pbar.postfix = "loss {}, mean_loss {}, cur_lr {}".format(round(loss.item(), 3), round(mean_loss.item(), 3), cur_lr)
        if (step + 1) % args.print_interval == 0:
            print_str = "[epoch {}/{} iter:{}] loss_cd {}, mean_loss {}".format(epoch + 1, args.epochC, step + 1, round(loss.item(), 3), round(mean_loss.item(), 3))
            logger.info(print_str)
        # scheduler.step()

    # evaluation
    with torch.no_grad():
        CD_Net.eval()
        torch.cuda.empty_cache()
        ebar = tqdm(test_loader, unit='pair')
        ebar.desc = "[Evaluation]".format(epoch + 1, args.epochC)
        for idx, val_data, in enumerate(ebar):
            image_A = val_data['A'].cuda()
            image_B = val_data['B'].cuda()
            label = val_data['L'].long().cuda()

            pred = CD_Net(image_A, image_B)
            out = pred.max(1)[1].detach().cpu().numpy()
            lbl = label.data.detach().cpu().numpy()
            running_metrics_val.update(lbl, out)
        score = running_metrics_val.get_scores()

        for k, v in score.items():
            logger.info('{}: {}'.format(k, round(v, 4)))
        if score['F1'] > best:
            best = score['F1']
            best_epoch = epoch + 1
            checkpoint = CD_Net.state_dict()
            save_path = os.path.join(logdir, "best_model.pth")
            torch.save(checkpoint, save_path)
            running_metrics_val.reset()
        print_str = "F1: {}({}) / Best F1: {}({})".format(round(score['F1'], 4), epoch + 1, round(best, 4), best_epoch)
        tqdm.write(print_str)
        logger.info("Best F1: {}(epoch:{})".format(best, best_epoch))
        time.sleep(1)
    torch.cuda.empty_cache()
    scheduler.step(score['F1'])
