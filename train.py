import os
import sys

import h5py
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.utils import make_grid

from utils import ramps, losses, metrics, test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
import torch.distributed as dist

from utils.lr_policy import WarmUpPolyLR

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_linear_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.linear_rampup(epoch, args.consistency_rampup)

def get_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.rampup(epoch, args.consistency_rampup)

def get_atten_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.atten_sigmoid_rampup(epoch, args.consistency_rampup)

def get_cos_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.cosine_rampdown(epoch, args.consistency_rampup)

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen


def get_kernel():
    weight = torch.zeros(8, 1, 3, 3)
    weight[0, 0, 0, 0] = 1
    weight[1, 0, 0, 1] = 1
    weight[2, 0, 0, 2] = 1

    weight[3, 0, 1, 0] = 1
    weight[4, 0, 1, 2] = 1

    weight[5, 0, 2, 0] = 1
    weight[6, 0, 2, 1] = 1
    weight[7, 0, 2, 2] = 1

    return weight

kernel = get_kernel()
# dilations = [1, 2, 4, 8, 12, 24]
dilations = [24]

num_iter = 20

# 膨胀 获得邻居
def get_dilated_neighbors(x):
    b, c, h, w = x.shape
    x_aff = []
    for d in dilations:
        _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
        _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
        # print(_x_pad.shape)
        _x = F.conv2d(_x_pad, kernel.cuda(), dilation=d).view(b * c, -1, h, w)
        x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

def get_pos():
    pos_xy = []

    ker = torch.ones(1, 1, 8, 1, 1)
    ker[0, 0, 0, 0, 0] = np.sqrt(2)
    ker[0, 0, 2, 0, 0] = np.sqrt(2)
    ker[0, 0, 5, 0, 0] = np.sqrt(2)
    ker[0, 0, 7, 0, 0] = np.sqrt(2)

    for d in dilations:
        pos_xy.append(ker * d)
    return torch.cat(pos_xy, dim=2)

def get_mask(imgs, masks):
    masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
    print(masks.shape)
    pos = get_pos()
    print(pos.shape)
    dim = 2
    w1 = 0.3
    w2 = 0.01

    b, c, h, w = imgs.shape
    _imgs = get_dilated_neighbors(imgs)
    print(_imgs.shape)
    _pos = pos.to(_imgs.device)
    print(_pos.shape)

    _pos_rep = _pos.repeat(b, 1, 1, h, w)
    print(_pos_rep.shape)

    _pos_std = torch.std(_pos_rep, dim=dim, keepdim=True)
    print(_pos_std.shape)

    pos_aff = -(_pos_rep / (_pos_std + 1e-8) / w1) ** 2
    pos_aff = pos_aff.mean(dim=1, keepdim=True)

    aff = w2 * F.softmax(pos_aff, dim=2).squeeze(1)
    print(aff.shape)

    # for _ in range(num_iter):
    #     _masks = get_dilated_neighbors(masks)
    #     print(_masks.shape)
    masks = (masks * aff).sum(2)

    return masks

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--model', type=str,  default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int,  default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='trained samples')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
parser.add_argument('--percent', type=float, default=90, help='weight to balance all losses')

args = parser.parse_args()

snapshot_path = args.root_path + "model/{}_{}_labeled/{}".format(args.dataset_name, args.labelnum, args.model)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path+'data/LA'
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path+'data/Pancreas'
    args.max_samples = 62
train_data_path = args.root_path

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model_1 = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    model_2 = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")

    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                        split='train',
                        transform = transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                            ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    # 标签数据
    labeled_idxs = list(range(labelnum))
    # 无标签数据
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-labeled_bs)

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed+worker_id)

    # trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer_1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    # 对比loss
    consistency_criterion = losses.mse_loss
    # dice loss
    # dice_loss = losses.Binary_dice_loss
    dice_loss = losses.dice_loss
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    cps_weight = 1.5

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    # tqdm：迭代进度条
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()


            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            noisy_input = volume_batch + noise
            volume_batch_l, label_batch_l = volume_batch[:labeled_bs], label_batch[:labeled_bs]
            volume_batch_un = volume_batch[labeled_bs:]
            noisy_input_l = noisy_input[:labeled_bs]
            noisy_input_un = noisy_input[labeled_bs:]

            model_1.train()
            model_2.train()

            out_l, l = model_1(volume_batch_l)
            out_un, un = model_1(volume_batch_un)
            out_noisy_l, noisy_l = model_2(noisy_input_l)
            out_noisy_un, noisy_un = model_2(noisy_input_un)

            out = torch.cat([out_l, out_un], dim=0)
            out_noisy = torch.cat([out_noisy_l, out_noisy_un], dim=0)

            pred = torch.cat([l, un], dim=0)
            pred_noisy = torch.cat([noisy_l, noisy_un], dim=0)

            # cps loss
            consistency_weight = get_current_consistency_weight(iter_num//100)
            # consistency_weight_att = get_current_consistency_weight(iter_num//150)
            # linear_weight = get_linear_consistency_weight(iter_num//150)
            # one_weight = get_consistency_weight(iter_num//150)
            # att_weight = get_atten_consistency_weight(iter_num//150)
            # cos_weight = get_cos_consistency_weight(iter_num//150)
            _, max_out = torch.max(out, dim=1)
            _, max_out_noisy = torch.max(out_noisy, dim=1)
            max_out = max_out.long()
            max_out_noisy = max_out_noisy.long()

            # _, max_pred = torch.max(pred, dim=1)
            # _, max_pred_noisy = torch.max(pred_noisy, dim=1)
            # max_pred = max_pred.long()
            # max_pred_noisy = max_pred_noisy.long()

            # cps_loss1 = (criterion(out, max_pred_noisy) + criterion(out_noisy, max_pred))
            cps_loss1 = (criterion(out, max_out_noisy) + criterion(out_noisy, max_out))
            cps_loss2 = losses.mse_loss(pred, pred_noisy)
            # cps_loss2 = torch.mean((pred - pred_noisy) ** 2)
            # cps_loss2 = (criterion(pred, max_pred_noisy) + criterion(pred_noisy, max_pred))

            cps_loss1 = consistency_weight * cps_loss1
            cps_loss2 = consistency_weight * cps_loss2
            cps_loss = cps_loss1 + cps_loss2
            # dist.all_reduce(cps_loss, dist .ReduceOp.SUM)
            # cps_loss = cps_loss / world_size
            # cps_loss = cps_loss * cps_weight


            # c_loss = criterion(max_out, max_out_noisy)
            # max_out_soft = F.softmax(max_out, dim=1)
            # uncertainty = -1.0*torch.sum(max_out_soft*torch.log(max_out_soft + 1e-6), dim=1, keepdim=True)
            # threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            # mask = (uncertainty < threshold).float()
            # c_loss = torch.sum(mask * c_loss) / (2 * torch.sum(mask) + 1e-16)

            # out_soft = F.softmax(out, dim=1)
            # uncertainty = -1.0*torch.sum(out_soft*torch.log(out_soft + 1e-6), dim=1, keepdim=True)
            # # threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            # threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            # mask = (uncertainty < threshold).float()
            # cps_loss =torch.sum(mask * cps_loss) / (2 * torch.sum(mask) + 1e-16)

            # atten_soft = F.softmax(atten, dim=1)
            # print(atten_soft)
            # # uncertainty = -1.0 * torch.sum(atten_soft * torch.log(atten_soft + 1e-6), dim=1, keepdim=True)
            # # print(uncertainty)
            # # threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            # threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            # print(threshold)
            # mask = (atten_soft < threshold).float()
            # print(mask)
            # cps_loss = torch.sum(mask * cps_loss) / (2 * torch.sum(mask) + 1e-16)
            #

            # cps_loss = torch.sum(mask_1 * cps_loss) / (2 * torch.sum(mask_1) + 1e-16)

            # cps_loss = cps_loss * consistency_weight

            ### standard cross entropy loss ###
            out_l_soft = F.softmax(out_l, dim=1)
            loss_dice = dice_loss(out_l_soft, label_batch_l)
            focal = losses.FocalLoss()
            # loss_hd = consistency_weight * hd_loss(max_out_l, label_batch_l)
            # loss_f_turkey = losses.focal_tversky(label_batch_l, max_out_l)
            # exp = losses.ExpLog_loss()
            # loss_exp = exp(out_l, label_batch_l)
            loss_seg = criterion(out_l, label_batch_l)
            # loss_f = focal(out_l_soft, label_batch_l)
            # print(loss_exp)
            loss_sup_1 = 0.5 * loss_seg + 0.5 * loss_dice
            # dist.all_reduce(loss_sup_1, dist.ReduceOp.SUM)
            # loss_sup_1 = loss_sup_1 / world_size

            out_l_noisy_soft = F.softmax(out_noisy_l, dim=1)
            # loss_dice_noisy = dice_loss(out_l_noisy_soft, label_batch_l)
            loss_sup_2 = 0.5 * criterion(out_noisy_l, label_batch_l)
            # dist.all_reduce(loss_sup_2, dist.ReduceOp.SUM)
            # loss_sup_2 = loss_sup_2 / world_size

            # _, max_atten = torch.max(atten_un, dim=1)
            # _, max_atten_noisy = torch.max(atten_noisy_un, dim=1)
            # loss_unsup = losses.mse_loss(atten_un, atten_noisy_un)

            iter_num = iter_num + 1
            loss = cps_loss + loss_sup_1 + loss_sup_2
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            loss.backward()
            optimizer_1.step()
            optimizer_2.step()

            logging.info('iteration %d : loss : %03f, loss_sup_1: %03f, loss_cross: %03f, cps_loss2: %03f' % (iter_num, loss, loss_sup_1, loss_seg, cps_loss1))
            writer.add_scalar('Labeled_loss/loss_sup_1', loss_sup_1, iter_num)
            writer.add_scalar('Labeled_loss/los_sup_2', loss_sup_2, iter_num)
            writer.add_scalar('Co_loss/cps_loss', cps_loss, iter_num)

            # change lr
            # if iter_num % 2500 == 0:
            #     lr_ = base_lr * 0.1 ** (iter_num // 2500)
            #     for param_group in optimizer_1.param_groups:
            #         param_group['lr'] = lr_
            #     for param_group in optimizer_2.param_groups:
            #         param_group['lr'] = lr_

            if iter_num >= 800 and iter_num % 200 == 0:
                ins_width = 2
                B, C, H, W, D = out.size()
                # print(y_prob_t.shape)
                snapshot_img = torch.zeros(size = (D, 3, (2) *H + (2) * ins_width, W + ins_width), dtype = torch.float32)

                volume_batch_l = np.squeeze(volume_batch_l, axis=1)
                target = label_batch_l[0, :, ...].permute(2, 0, 1)
                # print(target.shape)
                train_img = volume_batch_l[0, :, ...].permute(2, 0, 1)
                # print(train_img.shape)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:,:, :,W:W+ins_width] = 1
                for idx in range(2):
                    begin_grid = idx+1
                    snapshot_img[:,:, begin_grid*H + ins_width:begin_grid*H + begin_grid*ins_width,:] = 1

                # begin = 2
                # end = 3
                # a = begin * H + begin * ins_width
                # b = end*H + begin*ins_width
                # snapshot_img[:, 0, begin*H + begin*ins_width:end*H + begin*ins_width, :W] = y_prob_t[0, 1].permute(2, 0, 1)
                # snapshot_img[:, 1, begin*H + begin*ins_width:end*H + begin*ins_width, :W] = y_prob_t[0, 1].permute(2, 0, 1)
                # snapshot_img[:, 2, begin*H + begin*ins_width:end*H + begin*ins_width, :W] = y_prob_t[0, 1].permute(2, 0, 1)
                writer.add_images('Epoch_%d_Iter_%d_unlabel'% (epoch_num, iter_num), snapshot_img)
                
            if iter_num >= 800 and iter_num % 200 == 0:
                model_1.eval()
                if args.dataset_name =="LA":
                    dice_sample = test_patch.var_all_case(model_1, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4, dataset_name = 'LA')
                elif args.dataset_name =="Pancreas_CT":
                    dice_sample = test_patch.var_all_case(model_1, num_classes=num_classes, patch_size=patch_size, stride_xy=16, stride_z=16, dataset_name = 'Pancreas_CT')
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model_1.state_dict(), save_mode_path)
                    torch.save(model_1.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model_1.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model_1.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
            iterator.close()
            break
    writer.close()
