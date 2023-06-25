import sys

from model.Net import CD_Net

sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler
from torchinfo import summary
import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms
from utilis.metric_tool import ConfuseMatrixMeter
from utilis.torchutils import get_logger, make_numpy_grid, de_norm
import matplotlib.pyplot as plt
# from utils.metrics import RunningMetrics_CD
import os, time
import numpy as np
from argparse import ArgumentParser
import random


def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice


def BCE(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    return bce


@torch.no_grad()
def val(args, val_loader, model, cur_iter, epoch):
    model.eval()

    salEvalVal = ConfuseMatrixMeter(n_class=2)

    epoch_loss = []

    total_batches = len(val_loader)
    print(len(val_loader))
    for iter, batched_inputs in enumerate(val_loader):

        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        start_time = time.time()

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the mdoel
        output1, output2, output3 = model(pre_img_var, post_img_var, iter + cur_iter)
        loss = BCEDiceLoss(output1, target_var) + BCEDiceLoss(output2, target_var) + BCEDiceLoss(output3, target_var)

        pred = torch.where(output1 > 0.5, torch.ones_like(output1), torch.zeros_like(output1)).long()

        # torch.cuda.synchronize()
        time_taken = time.time() - start_time

        epoch_loss.append(loss.data.item())

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)
        # salEvalVal.addBatch(pred, target_var)
        f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())
        if iter % 5 == 0:
            print('\r[%d/%d] F1: %3f loss: %.3f time: %.3f' % (iter, total_batches, f1, loss.data.item(), time_taken),
                  end='')

        if np.mod(iter, 200) == 1:
            vis_input = make_numpy_grid(de_norm(pre_img_var[0:8]))
            vis_input2 = make_numpy_grid(de_norm(post_img_var[0:8]))
            vis_pred = make_numpy_grid(pred[0:8])
            vis_gt = make_numpy_grid(target_var[0:8])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                args.vis_dir, 'val_' + str(epoch) + '_' + str(iter) + '.jpg')
            plt.imsave(file_name, vis)

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_val, scores


def train(args, train_loader, model, optimizer, epoch, max_batches, cur_iter=0, lr_factor=1.):
    # switch to train mode
    model.train()

    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    total_batches = len(train_loader)

    for iter, batched_inputs in enumerate(train_loader):

        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        start_time = time.time()

        # adjust the learning rate
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches, lr_factor=lr_factor)

        if args.onGPU:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the model
        output1, output2, output3 = model(pre_img_var, post_img_var, iter + cur_iter)
        loss = BCEDiceLoss(output1, target_var) + BCEDiceLoss(output2, target_var) + BCEDiceLoss(output3, target_var)

        pred = torch.where(output1 > 0.5, torch.ones_like(output1), torch.zeros_like(output1)).long()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter - cur_iter) * time_taken / 3600

        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)

        # Computing F-measure and IoU on GPU
        with torch.no_grad():
            f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())

        if iter % 5 == 0:
            print('\riteration: [%d/%d] f1: %.3f lr: %.7f loss: %.3f time:%.3f h' % (
                iter + cur_iter, max_batches * args.max_epochs, f1, lr, loss.data.item(),
                res_time),
                  end='')
        if epoch in [0, 50, 100, 150]:
            model_file_name = args.savedir + '{}_model.pth'.format(cur_iter)
            torch.save({'cur_iter': cur_iter, 'state_dict': model.state_dict()}, model_file_name)

        if np.mod(iter, 200) == 1:
            vis_input = make_numpy_grid(de_norm(pre_img_var[0:8]))
            vis_input2 = make_numpy_grid(de_norm(post_img_var[0:8]))
            vis_pred = make_numpy_grid(pred[0:8])
            vis_gt = make_numpy_grid(target_var[0:8])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                args.vis_dir, 'train_' + str(epoch) + '_' + str(iter) + '.jpg')
            plt.imsave(file_name, vis)

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_train, scores, lr


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        cur_iter = iter
        max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr  # warm_up
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def trainValidateSegmentation(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID

    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    args.savedir = args.savedir + '/' + args.file_root + '/' + args.note + '/'
    args.vis_dir = args.savedir + '/Vis/'

    if args.file_root == 'LEVIR':
        args.file_root = '/data/sdu08_lyk/data/LEVIR-CD_256x256'
        # args.file_root = '/data2/wangyuting/lyk/LEVIR-CD'
        # args.file_root = '/home/wangyt/rs/LEVIR-CD'
    elif args.file_root == 'LEVIR+':
        args.file_root = '/data/sdu08_lyk/data/LEVIR-CD+_256x256'
        # args.file_root = '/home/wangyt/rs/LEVIR-CD+_256x256'
    elif args.file_root == 'BCDD':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/BCDD'
    elif args.file_root == 'SYSU':
        args.file_root = '/data/sdu08_lyk/data/SYSU-CD'
    elif args.file_root == 'CDD':
        args.file_root = '/data/sdu08_lyk/data/CDD'
    elif args.file_root == 'DSIFN':
        args.file_root = '/data/sdu08_lyk/data/DSIFN_256x256'
    elif args.file_root == 'CLCD':
        args.file_root = '/data/sdu08_lyk/data/CLCD_256x256'
    elif args.file_root == 'S2Looking':
        args.file_root = '/data/sdu08_lyk/data/S2Looking_256x256'
    elif args.file_root == 'quick_start':
        args.file_root = './samples'
    else:
        raise TypeError('%s has not defined' % args.file_root)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    # mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        # myTransforms.GaussianNoise(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    train_data = myDataLoader.Dataset("train", file_root=args.file_root, transform=trainDataset_main)

    trainLoader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    val_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    valLoader = torch.utils.data.DataLoader(
        val_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # whether use multi-scale training

    max_batches = len(trainLoader)

    print('For each epoch, we have {} batches'.format(max_batches))

    if args.onGPU:
        cudnn.benchmark = True

    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    start_epoch = 0
    cur_iter = 0
    max_F1_val = 0

    model = CD_Net(args.max_steps)
    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))
    if args.resume:
        args.resume = args.savedir + '/checkpoint.pth.tar'
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            cur_iter = start_epoch * len(trainLoader)
            # args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # if os.path.isfile(logFileLoc):
    #     logger = open(logFileLoc, 'a')
    # else:
    logger = get_logger(args.savedir)
    logger.info(args.__dict__)
    logger.info(
        summary(model, [(args.batch_size, 3, 256, 256), (args.batch_size, 3, 256, 256)], row_settings=["var_names"],
                col_names=["kernel_size", "output_size", "num_params", "mult_adds"], ))
    logger.info("Parameters: %s" % (str(total_params)))
    logger.info(
        "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa (val)', 'IoU (val)', 'F1 (val)', 'R (val)', 'P (val)'))
    params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    for epoch in range(start_epoch, args.max_epochs):

        lossTr, score_tr, lr = \
            train(args, trainLoader, model, optimizer, epoch, max_batches, cur_iter)

        torch.cuda.empty_cache()

        # evaluate on validation set
        if epoch == 0:
            continue

        lossVal, score_val = val(args, valLoader, model, cur_iter, epoch)

        torch.cuda.empty_cache()
        logger.info("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (epoch, score_val['Kappa'], score_val['IoU'],
                                                                      score_val['F1'], score_val['recall'],
                                                                      score_val['precision']))

        torch.save({
            'epoch': epoch + 1,
            'cur_iter': cur_iter,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossVal,
            'F_Tr': score_tr['F1'],
            'F_val': score_val['F1'],
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        cur_iter += len(trainLoader)

        # save the model also
        model_file_name = args.savedir + 'best_model.pth'
        if epoch % 1 == 0 and max_F1_val <= score_val['F1']:
            max_F1_val = score_val['F1']
            torch.save({'cur_iter': cur_iter, 'state_dict': model.state_dict()}, model_file_name)

        print("Epoch " + str(epoch) + ': Details')
        print("Epoch No. %d:\tTrain Loss = %.4f\tVal Loss = %.4f\t F1(tr) = %.4f\t F1(val) = %.4f\n" \
              % (epoch, lossTr, lossVal, score_tr['F1'], score_val['F1']))
        torch.cuda.empty_cache()
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict['state_dict'])

    loss_test, score_test = val(args, testLoader, model, state_dict['cur_iter'], 0)
    print("\nTest :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
          % (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision']))
    logger.info("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('Test', score_test['Kappa'], score_test['IoU'],
                                                                  score_test['F1'], score_test['recall'],
                                                                  score_test['precision']))
    # logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="LEVIR", help='Data directory | LEVIR | BCDD | SYSU | DSIFN ')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=40000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='/data/sdu08_lyk/data/logs', help='Directory to save the results')
    # parser.add_argument('--savedir', default='./logs', help='Directory to save the results')
    parser.add_argument('--resume', default=False, help='Use this checkpoint to continue training | '
                                                        './results_ep100/checkpoint.pth.tar')
    parser.add_argument('--logFile', default='trainValLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')
    parser.add_argument('--GPU_ID', type=str, default='6', help='GPU ID')
    parser.add_argument('--note', type=str, default='+1-', help='training note')

    args = parser.parse_args()
    print(args.note)
    print('Called with args:')
    print(args)
    trainValidateSegmentation(args)
    print(args.note)


