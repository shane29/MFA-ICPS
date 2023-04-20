import os
import sys

import h5py
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging

import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import ramps, losses, metrics, test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

# 制造高斯噪声 3d
def GAussian_noise(x, std, range):
        ns = torch.clamp(torch.randn_like(x) * std, -range, range)
        return ns

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--model', type=str,  default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int,  default=6000, help='maximum iteration to train')
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

args = parser.parse_args()

snapshot_path = args.root_path + "model\{}_{}_labeled/{}".format(args.dataset_name, args.labelnum, args.model)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path+'data/LA'
    args.max_samples = 80

train_data_path = args.root_path

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

    labelnum = args.labelnum
    # 标签数据
    labeled_idxs = list(range(labelnum))
    # 无标签数据
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-labeled_bs)

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed+worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer_1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    consistency_criterion = losses.mse_loss
    dice_loss = losses.dice_loss
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    cps_weight = 1.5

    iter_num = 0
    best_dice1 = 0
    best_dice2 = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            noise = GAussian_noise(volume_batch, 0.1, 0.2)
            noisy_input = volume_batch + noise

            model_1.train()
            model_2.train()
            # from torchinfo import summary
            #
            # batch_size = 4
            # summary(model_1, input_size=(batch_size, 1, 112, 112, 80))

            out, pred = model_1(volume_batch)
            out_noisy, pred_noisy = model_2(noisy_input)
            out_l = out[:labeled_bs]
            out_noisy_l = out_noisy[:labeled_bs]
            label_batch_l = label_batch[:labeled_bs]

            # cps loss
            consistency_weight = get_current_consistency_weight(iter_num//100)
            _, max_out = torch.max(out, dim=1)
            _, max_out_noisy = torch.max(out_noisy, dim=1)
            max_out = max_out.long()
            max_out_noisy = max_out_noisy.long()

            cps_loss1_1 = (criterion(out, max_out_noisy) + criterion(out_noisy, max_out))
            cps_loss2_1 = losses.mse_loss(pred, pred_noisy)

            cps_loss1 = consistency_weight * cps_loss1_1
            cps_loss2 = consistency_weight * cps_loss2_1
            cps_loss = cps_loss1 + cps_loss2

            ### standard cross entropy loss ###
            out_l_soft = F.softmax(out_l, dim=1)
            # 效果好的 90.43
            loss_dice1 = 1 * dice_loss(out_l_soft[:, 1, ...], label_batch_l == 1)
            loss_seg = criterion(out_l, label_batch_l)
            loss_seg = 0.2 * loss_seg
            loss_seg = loss_seg
            loss_sup_1 = loss_dice1 + loss_seg

            out_l_noisy_soft = F.softmax(out_noisy_l, dim=1)
            loss_dice2 = 1 * dice_loss(out_l_noisy_soft[:, 1, ...], label_batch_l == 1)
            loss_seg2 = 0.2 * criterion(out_noisy_l, label_batch_l)
            loss_sup_2 = loss_seg2 + loss_dice2

            iter_num = iter_num + 1
            loss = cps_loss + loss_sup_1 + loss_sup_2
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            loss.backward()
            optimizer_1.step()
            optimizer_2.step()

            logging.info('iteration %d : loss : %03f, loss_sup_1: %03f, loss_sup_2: %03f, cps_loss1: %03f' % (iter_num, loss, loss_sup_1, loss_sup_2, cps_loss1))

            writer.add_scalar('Labeled_loss/loss_dice1', loss_dice1, iter_num)
            # writer.add_scalar('Labeled_loss/los_seg_2', loss_seg2, iter_num)
            writer.add_scalar('Labeled_loss/loss_dice2', loss_dice2, iter_num)
            writer.add_scalar('Labeled_loss/loss_sup1', loss_sup_1, iter_num)
            writer.add_scalar('Labeled_loss/loss_sup2', loss_sup_2, iter_num)
            writer.add_scalar('Co_loss/cps_loss1_no', cps_loss1_1, iter_num)
            # writer.add_scalar('Co_loss/cps_loss2_no', cps_loss2_1, iter_num)
            writer.add_scalar('Co_loss/cps_loss1', cps_loss1, iter_num)
            # writer.add_scalar('Co_loss/cps_loss2', cps_loss2, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('Weight/con_weight', consistency_weight, iter_num)
            # writer.add_images('Epoch_%d_Iter_%d_noisy' % (epoch_num, iter_num), noisy_input)

            # if iter_num >= 4500 and iter_num % 100 == 0:
            #     ins_width = 2
            #     B, C, H, W, D = out.size()
            #     b, c, h, w, d = B, C, H // 2, W // 2, D // 2
            #     # print(y_prob_t.shape)
            #     snapshot_img = torch.zeros(size=(D, 3, (5) * H + (5) * ins_width, W + ins_width), dtype=torch.float32)
            #     Att_img = torch.zeros(size=(d, 3, (2) * h + (2) * ins_width, w + ins_width), dtype=torch.float32)
            #     # print(snapshot_img.size())
            #     # 加噪图像 （2，112，112，80）
            #     noisy_input_l = np.squeeze(noisy_input_l, axis=1)
            #     # 原图像 （2，112，112，80）
            #     volume_batch_l = np.squeeze(volume_batch_l, axis=1)
            #
            #     # 原图像 row_image (80, 112, 112)
            #     train_img = volume_batch_l[0, :, ...].permute(2, 0, 1)
            #     # print(train_img.size())
            #     # 加噪后的图像 noisy_image
            #     noisy_img = noisy_input_l[0, :, ...].permute(2, 0, 1)
            #     # print(noisy_img.size())
            #     # 真实标签 gt (80, 112, 112)
            #     target = label_batch_l[0, :, ...].permute(2, 0, 1)
            #     # print(target.size())
            #
            #     # 网络1 分割结果 (80, 112, 112)
            #     pred1 = out_l[0, 1].permute(2, 0, 1)
            #     # print(pred1.size())
            #     # 网络2 分割结果 (80, 112, 112)
            #     pred2 = out_noisy_l[0, 1].permute(2, 0, 1)
            #     # print(pred2.size())
            #
            #     # 网络1 多维特征注意力图
            #     Att1 = l[0, 1].permute(2, 0, 1)
            #     # print(Att1.size())
            #     # 网络2 多维特征注意力图
            #     Att2 = noisy_l[0, 1].permute(2, 0, 1)
            #     # print(Att2.size())
            #
            #     # 原图像
            #     snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (
            #                 torch.max(train_img) - torch.min(train_img))
            #     snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (
            #                 torch.max(train_img) - torch.min(train_img))
            #     snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (
            #                 torch.max(train_img) - torch.min(train_img))
            #
            #     # 加噪后的图像 归一化不？ (noisy_img-torch.min(noisy_img))/(torch.max(noisy_img)-torch.min(noisy_img))
            #     snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = noisy_img
            #     snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = noisy_img
            #     snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = noisy_img
            #
            #     snapshot_img[:, :, :, W:W + ins_width] = 1
            #     for idx in range(2):
            #         begin_grid = idx + 1
            #         snapshot_img[:, :, begin_grid * H + ins_width:begin_grid * H + begin_grid * ins_width, :] = 1
            #
            #     begin = 2
            #     end = 3
            #     a = begin * H + begin * ins_width
            #     b = end * H + begin * ins_width
            #     snapshot_img[:, 0, a:b, :W] = target
            #     snapshot_img[:, 1, a:b, :W] = target
            #     snapshot_img[:, 2, a:b, :W] = target
            #
            #     begin = 3
            #     end = 4
            #     a = begin * H + begin * ins_width
            #     b = end * H + begin * ins_width
            #     snapshot_img[:, 0, a:b, :W] = pred1
            #     snapshot_img[:, 1, a:b, :W] = pred1
            #     snapshot_img[:, 2, a:b, :W] = pred1
            #
            #     begin = 4
            #     end = 5
            #     a = begin * H + begin * ins_width
            #     b = end * H + begin * ins_width
            #     snapshot_img[:, 0, begin * H + begin * ins_width:end * H + begin * ins_width, :W] = pred2
            #     snapshot_img[:, 1, begin * H + begin * ins_width:end * H + begin * ins_width, :W] = pred2
            #     snapshot_img[:, 2, begin * H + begin * ins_width:end * H + begin * ins_width, :W] = pred2
            #
            #     Att_img[:, 0, :h, :w] = Att1
            #     Att_img[:, 1, :h, :w] = Att1
            #     Att_img[:, 2, :h, :w] = Att1
            #     Att_img[:, 0, h + ins_width:2 * h + ins_width, :w] = Att2
            #     Att_img[:, 1, h + ins_width:2 * h + ins_width, :w] = Att2
            #     Att_img[:, 2, h + ins_width:2 * h + ins_width, :w] = Att2
            #
            #     # snapshot_img: 1、row_img 2、target 3、noisy_img 4、pred1 5、pred2
            #     writer.add_images('Epoch_%d_Iter_%d_unlabel' % (epoch_num, iter_num), snapshot_img)
            #     # Att_img: 1、Att1 2、Att2
            #     writer.add_images('Epoch_%d_Iter_%d_Att_unlabel' % (epoch_num, iter_num), Att_img)

            if iter_num >= 5000 and iter_num % 100 == 0:
                model_1.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model_1, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "Pancreas_CT":
                    dice_sample = test_patch.var_all_case(model_1, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=16, stride_z=16, dataset_name='Pancreas_CT')
                if dice_sample > best_dice1:
                    best_dice1 = dice_sample
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1-iter_{}_dice_{}.pth'.format(iter_num, best_dice1))
                    save_best_path = os.path.join(snapshot_path, 'model1-{}_best_model.pth'.format(args.model))
                    torch.save(model_1.state_dict(), save_mode_path)
                    torch.save(model_1.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice_model1/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice_model1/Best_dice1', best_dice1, iter_num)
                # model_1.train()
                model_2.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model_2, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "Pancreas_CT":
                    dice_sample = test_patch.var_all_case(model_2, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=16, stride_z=16, dataset_name='Pancreas_CT')
                if dice_sample > best_dice2:
                    best_dice2 = dice_sample
                    save_mode2_path = os.path.join(snapshot_path,
                                                   'model2-iter_{}_dice_{}.pth'.format(iter_num, best_dice2))
                    save_best2_path = os.path.join(snapshot_path, 'model2-{}_best_model.pth'.format(args.model))
                    torch.save(model_2.state_dict(), save_mode2_path)
                    torch.save(model_2.state_dict(), save_best2_path)
                    logging.info("save best model to {}".format(save_mode2_path))
                writer.add_scalar('Var_dice_model2/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice_model2/Best_dice2', best_dice2, iter_num)
                # model_2.train()

            if iter_num >= max_iterations:
                save_mode1_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model_1.state_dict(), save_mode1_path)
                logging.info("save model1 to {}".format(save_mode1_path))

                save_mode2_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model_2.state_dict(), save_mode2_path)
                logging.info("save model2 to {}".format(save_mode2_path))
                break
        if iter_num >= max_iterations:
            net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
            iterator.close()
            break
    writer.close()
