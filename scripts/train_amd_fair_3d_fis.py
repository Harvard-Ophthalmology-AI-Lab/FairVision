import os
import argparse
import random
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *
from torch.optim import *
import torch.nn.functional as F

from sklearn.metrics import *
from sklearn.model_selection import KFold
from sklearn.utils import *

import sys
sys.path.append('.')

from src.modules import *
from src.data_handler import *
from src import logger
from typing import NamedTuple

from fairlearn.metrics import *

from src.utils.utils import model_to_syncbn, Transform3D
from src.acsconv.models.convnext import convnext_base as ConvnextBase
from src.acsconv.converters import ACSConverter, Conv3dConverter, Conv2_5dConverter
from src.models_3d import *

import models_vit
from models_vit import NativeScalerWithGradNormCount, adjust_learning_rate, param_groups_lrd, interpolate_pos_embed
from timm.models.layers import trunc_normal_

class Dataset_Info(NamedTuple):
    beta: float = 0.9999
    gamma: float = 2.0
    samples_per_attr: list[int] = [0,0,0]
    loss_type: str = "sigmoid"
    no_of_classes: int = 2
    no_of_attr: int = 3

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

parser.add_argument('--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--seed', default=-1, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--start-epoch', default=0, type=int)

parser.add_argument('--pretrained-weights', default='', type=str)

parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--data_dir', default='.', type=str)
parser.add_argument('--model_type', default='efficientnet', type=str)
parser.add_argument('--task', default='cls', type=str)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--loss_type', default='bce', type=str)
parser.add_argument('--modality_types', default='slo_fundus', type=str, help='oct_bscans_3d|slo_fundus')
parser.add_argument('--fuse_coef', default=1.0, type=float)
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--attribute_type', default='race', type=str, help='race|gender|hispanic')
parser.add_argument('--subset_name', default='test', type=str)
parser.add_argument("--need_balance", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Oversampling or not")
parser.add_argument('--dataset_proportion', default=1., type=float)
parser.add_argument('--num_classes', default=4, type=int)
parser.add_argument('--bootstrap_repeat_times', default=100, type=int)
parser.add_argument('--conv_type', default='Conv3d', type=str)
parser.add_argument('--shape_transform',
                        help='for shape dataset, whether multiply 0.5 at eval',
                    action="store_true")
parser.add_argument('--pretrained_3d',
                        default='i3d',
                        type=str)
parser.add_argument('--fair_scaling_coef', default=0.5, type=float)
# parser.add_argument('--fair_scaling_group_weights', nargs='*', help='<Required> Set flag', required=True)
parser.add_argument('--fair_scaling_batchsize', default=8, type=int)
parser.add_argument('--fair_scaling_sinkhorn_blur', default=.1, type=float)
parser.add_argument('--fair_scaling_temperature', default=1., type=float)
parser.add_argument('--fair_right_asymptote', default=1., type=float)

# parser.add_argument('--optimizer', type=str, required=True,
#                     choices=['sgd', 'adam', 'adamw', 'adagrad', 'rmsprop', 'adadelta', 'adamax', 'asgd'],
#                     help='Name of the optimizer (e.g., sgd, adam)')
# parser.add_argument('--optimizer_arguments', type=str, default='{}',
#                     help='Additional optimizer parameters as a dictionary (e.g., \'{"weight_decay": 0.01}\')')
# parser.add_argument('--scheduler', type=str, required=True,
#                     choices=['step_lr', 'multi_step_lr', 'exponential_lr', 'cosine_annealing_lr',
#                              'reduce_lr_on_plateau', 'cyclic_lr', 'one_cycle_lr'],
#                     help='Name of the learning rate scheduler')
# parser.add_argument('--scheduler_arguments', type=str, default='{}',
#                     help='Additional scheduler parameters as a dictionary (e.g., \'{"step_size": 30, "gamma": 0.1}\')')

# ViT Args
parser.add_argument('--vit_weights', default='dinov2_registers', type=str)
parser.add_argument('--blr', type=float, default=1e-3,
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--min_lr', type=float, default=1e-6,
                    help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='epochs to warmup LR')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--layer_decay', type=float, default=0.75,
                    help='layer-wise lr decay from ELECTRA/BEiT')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def train(model, criterion, optimizer, scaler, train_dataset_loader, epoch, total_iteration, dataset_info=None, num_classes=2, args=None, attr_index=0, loss_scaler=None, group_dataloaders=[]):
    global device

    model.train()
    
    loss_batch = []
    top1_accuracy_batch = []
    top5_accuracy_batch = []
    
    preds = []
    gts = []
    attrs = []
    datadirs = []

    preds_by_attr = [ [] for _ in range(dataset_info.no_of_attr) ]
    gts_by_attr = [ [] for _ in range(dataset_info.no_of_attr) ]
    t1 = time.time()
    for i, (input, target, attr) in enumerate(train_dataset_loader):
        if args.model_type == 'ViT-B':
            # we use a per iteration (instead of per epoch) lr scheduler
            tmp_lr = adjust_learning_rate(optimizer, i / len(train_dataset_loader) + epoch, args)
            if i == 0:
                print(f"lr: {tmp_lr}")

        with torch.cuda.amp.autocast():
            input = input.to(device)
            target = target.to(device)

            pred = model(input) # .squeeze(1)

            smp_losses = []
            for x in group_dataloaders:
                smp_input, smp_target, smp_attr = next(x)
                smp_input = smp_input.to(device)
                smp_target = smp_target.to(device)
                with torch.no_grad():
                    if args.model_type in ['VideoMAE', 'ViViT'] and args.modality_types == 'oct_bscans':
                        smp_input = smp_input.unsqueeze(2)
                        smp_pred = model(smp_input).logits

                    else:
                        smp_pred = model(smp_input)

                    if smp_pred.shape[1] == 1:
                        smp_pred = smp_pred.squeeze(1)
                        smp_loss = criterion(smp_pred, smp_target)
                    elif smp_pred.shape[1] > 1:
                        smp_loss = criterion(smp_pred, smp_target.long())
                    smp_losses.append(smp_loss)

            if pred.shape[1] == 1:
                pred = pred.squeeze(1)
                loss = criterion(pred, target)
                pred_prob = torch.sigmoid(pred.detach())
            elif pred.shape[1] > 1:
                loss = criterion(pred, target.long())
                pred_prob = F.softmax(pred.detach(), dim=1)

            loss = loss_scaler(loss, smp_losses, attr=attr[attr_index])

            preds.append(pred_prob.detach().cpu().numpy())
            gts.append(target.detach().cpu().numpy())
            attr = torch.vstack(attr)
            attrs.append(attr.detach().cpu().numpy())

        loss_batch.append(loss.item())
        
        if num_classes == 2:
            top1_accuracy = accuracy(pred_prob.detach().cpu().numpy(), target.detach().cpu().numpy(), topk=(1,))
        elif num_classes > 2:
            top1_accuracy = accuracy(pred_prob, target, topk=(1,))
        
        top1_accuracy_batch.append(top1_accuracy)

        if args.model_type == 'ViT-B':
            scaler(loss, optimizer, clip_grad=None,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=True)
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        optimizer.zero_grad()


    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=1).astype(int)
    
#     idx = [i for i in range(len(preds)) if not np.isnan(preds[i])]
#     preds = preds[idx]
#     gts = gts[idx]
#     attrs = attrs[:,idx]
    
    cur_auc = compute_auc(preds, gts, num_classes=num_classes)
    # acc = np.mean(top1_accuracy_batch)
    if num_classes == 2:
        acc = accuracy(preds, gts, topk=(1,))
    elif num_classes > 2:
        acc = accuracy(torch.from_numpy(preds).cuda(), torch.from_numpy(gts).cuda(), topk=(1,))

    torch.cuda.synchronize()
    t2 = time.time()

    print(f"train ====> epcoh {epoch} loss: {np.mean(loss_batch):.4f} auc: {cur_auc:.4f} time: {t2 - t1:.4f}")

    t1 = time.time()

    return np.mean(loss_batch), acc, cur_auc, preds, gts, attrs
    

def validation(model, criterion, optimizer, validation_dataset_loader, epoch, result_dir=None, dataset_info=None, num_classes=2):
    global device

    model.eval()
    
    loss_batch = []
    top1_accuracy_batch = []
    top5_accuracy_batch = []

    preds = []
    gts = []
    attrs = []
    datadirs = []

    preds_by_attr = [ [] for _ in range(dataset_info.no_of_attr) ]
    gts_by_attr = [ [] for _ in range(dataset_info.no_of_attr) ]

    with torch.no_grad():
        for i, (input, target, attr) in enumerate(validation_dataset_loader):
            input = input.to(device)
            target = target.to(device)
            
            pred = model(input) # .squeeze(1)

            if pred.shape[1] == 1:
                pred = pred.squeeze(1)
                loss = criterion(pred, target).mean()
                pred_prob = torch.sigmoid(pred.detach())
            elif pred.shape[1] > 1:
                loss = criterion(pred, target.long()).mean()
                pred_prob = F.softmax(pred.detach(), dim=1)

            preds.append(pred_prob.detach().cpu().numpy())
            gts.append(target.detach().cpu().numpy())
            attr = torch.vstack(attr)
            attrs.append(attr.detach().cpu().numpy())
            

            loss_batch.append(loss.item())

            if num_classes == 2:
                top1_accuracy = accuracy(pred_prob.detach().cpu().numpy(), target.detach().cpu().numpy(), topk=(1,))
            elif num_classes > 2:
                top1_accuracy = accuracy(pred_prob, target, topk=(1,))
        
            top1_accuracy_batch.append(top1_accuracy)
        
    loss = np.mean(loss_batch)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=1).astype(int)
    
    #preds = np.nan_to_num(preds, 0)
#     idx = [i for i in range(len(preds)) if not np.isnan(preds[i])]
#     preds = preds[idx]
#     gts = gts[idx]
#     attrs = attrs[:,idx]
    
    cur_auc = compute_auc(preds, gts, num_classes=num_classes)

    if num_classes == 2:
        acc = accuracy(preds, gts, topk=(1,))
    elif num_classes > 2:
        acc = accuracy(torch.from_numpy(preds).cuda(), torch.from_numpy(gts).cuda(), topk=(1,))

    print(f"test <==== epcoh {epoch} loss: {np.mean(loss_batch):.4f} auc: {cur_auc:.4f}")

    return loss, acc, cur_auc, preds, gts, attrs


if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    logger.log(f'===> random seed: {args.seed}')

    logger.configure(dir=args.result_dir, log_suffix='train')

    with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.model_type == 'vit' or args.model_type == 'swin' or args.model_type == 'ViT-B':
        args.image_size = 224

    
    train_transform = Transform3D(mul='random') if args.shape_transform else Transform3D()
    eval_transform = Transform3D(mul='0.5') if args.shape_transform else Transform3D()
    
    
    train_havo_dataset = FairVision_AMD(os.path.join(args.data_dir, 'train'), modality_type=args.modality_types, task=args.task, resolution=args.image_size, attribute_type=args.attribute_type, needBalance=args.need_balance, transform=train_transform, dataset_proportion=args.dataset_proportion)
    val_havo_dataset = FairVision_AMD(os.path.join(args.data_dir, 'val'), modality_type=args.modality_types, task=args.task, resolution=args.image_size, transform=eval_transform, attribute_type=args.attribute_type)
    test_havo_dataset = FairVision_AMD(os.path.join(args.data_dir, 'test'), modality_type=args.modality_types, task=args.task, resolution=args.image_size, transform=eval_transform, attribute_type=args.attribute_type)

    args.num_classes = int(np.max(list(train_havo_dataset.disease_mapping.values())))+1
    logger.log(f'there are {args.num_classes} classes in total')
    logger.log(train_havo_dataset.disease_mapping)

    logger.log(f'train patients {len(train_havo_dataset)} with {len(train_havo_dataset)} samples, val patients {len(val_havo_dataset)} with {len(val_havo_dataset)} samples, test patients {len(test_havo_dataset)} with {len(test_havo_dataset)} samples')

    train_dataset_loader = torch.utils.data.DataLoader(
        train_havo_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    validation_dataset_loader = torch.utils.data.DataLoader(
        val_havo_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    test_dataset_loader = torch.utils.data.DataLoader(
        test_havo_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    
    print(len(train_dataset_loader))
    
    samples_per_attr = get_num_by_group_(train_dataset_loader)
    logger.log(f'group information:')
    logger.log(samples_per_attr)
    ds_info = Dataset_Info(no_of_attr=len(samples_per_attr))

    groups_in_attrs = [3, 2, 2, 3, 5]
    attr_to_idx = {'race': 0, 'gender': 1, 'hispanic': 2, 'language': 3, 'maritalstatus': 4}

    group_dataloaders = []
    for i in range(groups_in_attrs[attr_to_idx[args.attribute_type]]):

        group_havo_dataset = FairVision_AMD(os.path.join(args.data_dir, 'train'), modality_type=args.modality_types, task=args.task, resolution=args.image_size, attribute_type=args.attribute_type, needBalance=args.need_balance, dataset_proportion=args.dataset_proportion, select_group=i)
        group_dataset_loader = torch.utils.data.DataLoader(
            group_havo_dataset, batch_size=args.fair_scaling_batchsize, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False)

        group_dataloaders.append(endless_loader(group_dataset_loader))

    best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')
    lastep_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'last_{args.perf_file}')

    acc_head_str = ''
    auc_head_str = ''
    dpd_head_str = ''
    eod_head_str = ''
    esacc_head_str = ''
    esauc_head_str = ''
    group_disparity_head_str = ''
    if args.perf_file != '':
        if not os.path.exists(best_global_perf_file):
            for i in range(len(samples_per_attr)):
                auc_head_str += ', '.join([f'auc_attr{i}_group{x}' for x in range(len(samples_per_attr[i]))]) + ', '
            dpd_head_str += ', '.join([f'dpd_attr{i}' for i in range(len(samples_per_attr))]) + ', '
            eod_head_str += ', '.join([f'eod_attr{i}' for i in range(len(samples_per_attr))]) + ', '
            esacc_head_str += ', '.join([f'esacc_attr{i}' for i in range(len(samples_per_attr))]) + ', '
            esauc_head_str += ', '.join([f'esauc_attr{i}' for i in range(len(samples_per_attr))]) + ', '

            group_disparity_head_str += ', '.join([f'std_group_disparity_attr{i}, max_group_disparity_attr{i}' for i in range(len(samples_per_attr))]) + ', '
            
            with open(best_global_perf_file, 'w') as f:
                f.write(f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_head_str} path\n')

    if args.task == 'md':
        out_dim = 1
        criterion = nn.MSELoss()
        predictor_head = nn.Identity()
    elif args.task == 'cls': 
        out_dim = 1
        if args.num_classes == 2:
            out_dim = 1
            criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif args.num_classes > 2:
            out_dim = args.num_classes
            criterion = nn.CrossEntropyLoss(reduction='none')
        predictor_head = nn.Sigmoid()
    elif args.task == 'tds': 
        out_dim = 52
        criterion = nn.MSELoss()
        predictor_head = nn.Identity()

    # if args.modality_types == 'ilm' or args.modality_types == 'rnflt':
    #     in_dim = 1
    #     model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
    # elif args.modality_types == 'slo_fundus':
    #     in_dim = 3
    #     model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
    # elif args.modality_types == 'oct_bscans':
    #     in_dim = 128
    #     model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
    # elif args.modality_types == 'oct_bscans_3d':
    #     in_dim = 1
    #     model = ConvNet_3D(out_dim=out_dim)
    # elif args.modality_types == 'rnflt+ilm':
    #     in_dim = 2
    #     model = OphBackbone(model_type=args.model_type, in_dim=in_dim, coef=args.fuse_coef)

#     assert args.modality_types == 'slo_fundus'
    if args.model_type == 'ViT-B':
        args.lr = args.blr * args.batch_size / 256
        args.image_size = 224

        if args.vit_weights in ['scratch', 'imagenet', 'mae', 'mocov3', 'mae_chest_xray', 'mae_color_fundus']:
            if args.vit_weights == 'mae_color_fundus':
                model = models_vit.__dict__["vit_large_patch16"](
                    num_classes=out_dim,
                    drop_path_rate=args.drop_path,
                    global_pool=True,
                )
            else:
                model = models_vit.__dict__["vit_base_patch16"](
                    num_classes=out_dim,
                    drop_path_rate=args.drop_path,
                    global_pool=True,
                )

            if args.vit_weights != 'scratch':
                if args.vit_weights == 'imagenet':
                    imagenet_init_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)
                    print("Load ImageNet pre-trained checkpoint from timm")
                    checkpoint_model = imagenet_init_model.state_dict()
                    del imagenet_init_model
                else:
                    if args.vit_weights == 'mae':
                        checkpoint_path = "/scratch/mok232/Fairness_in_Eye_Disease_Screening/MODELHUB/mae_pretrain_vit_base.pth"
                    elif args.vit_weights == 'mocov3':
                        checkpoint_path = "/scratch/mok232/Fairness_in_Eye_Disease_Screening/MODELHUB/mocov3_pretrain_vit_base.pth"
                    elif args.vit_weights == 'mae_chest_xray':
                        checkpoint_path = "/scratch/mok232/Fairness_in_Eye_Disease_Screening/MODELHUB/vit-b_CXR_0.5M_mae.pth"
                    elif args.vit_weights == 'mae_color_fundus':
                        checkpoint_path = "/scratch/mok232/Fairness_in_Eye_Disease_Screening/MODELHUB/RETFound_cfp_weights.pth"
                    
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    print("Load pre-trained checkpoint from: %s" % checkpoint_path)
                    checkpoint_model = checkpoint['model']
                
                state_dict = model.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

                # interpolate position embedding
                interpolate_pos_embed(model, checkpoint_model)

                # load pre-trained model
                msg = model.load_state_dict(checkpoint_model, strict=False)
                print(msg)

                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

        elif args.vit_weights == 'dinov2':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            model.head = torch.nn.Linear(in_features=768, out_features=out_dim, bias=True)

        elif args.vit_weights == 'dinov2_registers':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            model.head = torch.nn.Linear(in_features=768, out_features=out_dim, bias=True)
        
        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

        # build optimizer with layer-wise lr decay (lrd)
        if args.vit_weights in ['mae', 'mocov3', 'mae_chest_xray', 'mae_color_fundus', 'scratch', 'imagenet']:
            no_weight_decay_list = model.no_weight_decay()
        elif args.vit_weights in ['dinov2', 'dinov2_registers']:
            no_weight_decay_list = {'pos_embed', 'dist_token', 'cls_token', 'mask_token', 'register_tokens'} # add mask_token, register_tokens for DINOv2
        
        param_groups = param_groups_lrd(model, args.weight_decay,
            no_weight_decay_list=no_weight_decay_list,
            layer_decay=args.layer_decay
        )
    
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        scaler = NativeScalerWithGradNormCount()

        # misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    else:
        scaler = torch.cuda.amp.GradScaler()

#         in_dim = 3
#         model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
        
        
        if args.modality_types == 'ilm' or args.modality_types == 'rnflt':
            in_dim = 1
            model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
        elif args.modality_types == 'slo_fundus':
            in_dim = 3
            model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
        elif args.modality_types == 'oct_bscans':
            in_dim = 128
            model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
        elif args.modality_types == 'oct_bscans_3d':
#             in_dim = 1
#             model = ConvNet_3D(out_dim=out_dim)
            in_dim = 128 # 200
            tmp_num_output = args.num_classes
            if args.num_classes == 2:
                tmp_num_output = 1
            if args.model_type == 'resnet18':
                model = ResNet18(in_channels=1, num_classes=tmp_num_output)
            elif args.model_type == 'resnet50':
                model = ResNet50(in_channels=1, num_classes=tmp_num_output)
            elif args.model_type == 'vgg16':
                model = VGG16(in_channels=1, num_classes=tmp_num_output)
            elif args.model_type == 'densenet121':
                model = DenseNet121(in_channels=1, num_classes=tmp_num_output)
            elif args.model_type == 'convnext_base':
                model = ConvNeXt_Base(in_channels=1, num_classes=tmp_num_output)
            elif args.model_type == 'unet':
                model = UNet(in_channels=1, num_classes=tmp_num_output)
            else:
                raise NotImplementedError

            if args.conv_type =='ACSConv':
                model = model_to_syncbn(ACSConverter(model))
                print('using ACSConv model')
            if args.conv_type=='Conv2_5d':
                model = model_to_syncbn(Conv2_5dConverter(model))
                print('using conv2.5d model')
            if args.conv_type=='Conv3d':
                if args.pretrained_3d == 'i3d':
                    print('using i3d pretrained weight')
                    model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
                    print('using Conv3d model')
                else:
                    model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))
                    print('using Conv3d model')
        elif args.modality_types == 'rnflt+ilm':
            in_dim = 2
            model = OphBackbone(model_type=args.model_type, in_dim=in_dim, coef=args.fuse_coef)
        
        
        

        optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.0, 0.1), weight_decay=args.weight_decay)
        #
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        # optimizer=get_optimizer(args.optimizer,model.parameters(),args.lr,args.optimizer_arguments)
        # scheduler=get_scheduler(args.scheduler,optimizer,args.scheduler_arguments)

        # raise Exception('not implemented')

    model = model.to(device)

    loss_scaler = Fair_Loss_Scaler(fair_scaling_coef=args.fair_scaling_coef, sinkhorn_blur=args.fair_scaling_sinkhorn_blur,
                                        fair_scaling_temperature=args.fair_scaling_temperature, right_asymptote=args.fair_right_asymptote)
    loss_scaler = loss_scaler.to(device)

    start_epoch = 0
    best_top1_accuracy = 0.

    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weights)

        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    attribute_index = attr_to_idx[args.attribute_type]

    total_iteration = len(train_havo_dataset)//args.batch_size

    best_auc_groups = None
    best_acc_groups = None
    best_pred_gt_by_attr = None
    best_auc = sys.float_info.min
    best_acc = sys.float_info.min
    best_es_acc = sys.float_info.min
    best_es_auc = sys.float_info.min
    best_ep = 0
    best_between_group_disparity = None

    best_val_auc = sys.float_info.min

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, train_auc, train_preds, train_gts, train_attrs = train(model, criterion, optimizer, scaler, train_dataset_loader, epoch, total_iteration, dataset_info=ds_info, num_classes=args.num_classes, args=args, attr_index=attribute_index, loss_scaler=loss_scaler, group_dataloaders=group_dataloaders)
        val_loss, val_acc, val_auc, val_preds, val_gts, val_attrs = validation(model, criterion, optimizer, validation_dataset_loader, epoch, dataset_info=ds_info, num_classes=args.num_classes)
        test_loss, test_acc, test_auc, test_preds, test_gts, test_attrs = validation(model, criterion, optimizer, test_dataset_loader, epoch, dataset_info=ds_info, num_classes=args.num_classes)
        if args.model_type != 'ViT-B': # we use per-iter lr scheduler for ViT
            scheduler.step()

        val_es_acc, val_es_auc, val_aucs_by_attrs, val_dpds, val_eods, val_between_group_disparity = evalute_comprehensive_perf(val_preds, val_gts, val_attrs, num_classes=args.num_classes)
        test_es_acc, test_es_auc, test_aucs_by_attrs, test_dpds, test_eods, test_between_group_disparity = evalute_comprehensive_perf(test_preds, test_gts, test_attrs, num_classes=args.num_classes)
        # test_acc, test_es_acc, test_auc, test_es_auc, test_aucs_by_attrs, test_dpds, test_eods, test_between_group_disparity, \
        # test_acc_std, test_es_acc_std, test_auc_std, test_es_auc_std, test_aucs_by_attrs_std, test_dpds_std, test_eods_std, test_between_group_disparity_std = bootstrap_performance(test_preds, test_gts, test_attrs, num_classes=args.num_classes, bootstrap_repeat_times=args.bootstrap_repeat_times)
        # print(test_auc)
        # sys.exit()

        if test_auc > best_val_auc:
            best_val_auc = test_auc
            best_auc = test_auc
            best_acc = test_acc
            best_ep = epoch
            best_auc_groups = test_aucs_by_attrs
            best_dpd_groups = test_dpds
            best_eod_groups = test_eods
            best_es_acc = test_es_acc
            best_es_auc = test_es_auc
            best_between_group_disparity = test_between_group_disparity
            best_test_preds, best_test_gts, best_test_attrs = test_preds, test_gts, test_attrs

            if args.model_type == 'ViT-B':
                state = {
                'epoch': epoch,# zero indexing
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scaler_state_dict' : scaler.state_dict(),
                # 'scheduler_state_dict' : scheduler.state_dict(),
                'train_auc': train_auc,
                'val_auc': val_auc,
                'test_auc': test_auc
                }
            else:
                state = {
                'epoch': epoch,# zero indexing
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scaler_state_dict' : scaler.state_dict(),
                # 'scheduler_state_dict' : scheduler.state_dict(),
                'train_auc': train_auc,
                'val_auc': val_auc,
                'test_auc': test_auc
                }
            torch.save(state, os.path.join(args.result_dir, f"model_best_epoch.pth"))

            if args.result_dir is not None:
                np.savez(os.path.join(args.result_dir, f'pred_gt_best_epoch.npz'), 
                            test_pred=test_preds, test_gt=test_gts, test_attr=test_attrs)

#         logger.log(f'---- best val AUC {best_val_auc:.4f} at epoch {best_ep}')
        logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
        logger.log(f'---- best AUC by groups and attributes at epoch {best_ep}')
        logger.log(best_auc_groups)

        logger.logkv('epoch', epoch)
        logger.logkv('train_loss', round(train_loss,4))
        logger.logkv('train_acc', round(train_acc,4))
        logger.logkv('train_auc', round(train_auc,4))

        logger.logkv('val_loss', round(val_loss,4))
        logger.logkv('val_acc', round(val_acc,4))
        logger.logkv('val_auc', round(val_auc,4))

        logger.logkv('test_loss', round(test_loss,4))
        logger.logkv('test_acc', round(test_acc,4))
        logger.logkv('test_auc', round(test_auc,4))

        for ii in range(len(val_es_acc)):
            logger.logkv(f'val_es_acc_attr{ii}', round(val_es_acc[ii],4))
        for ii in range(len(val_es_auc)):
            logger.logkv(f'val_es_auc_attr{ii}', round(val_es_auc[ii],4))
        for ii in range(len(val_aucs_by_attrs)):
            for iii in range(len(val_aucs_by_attrs[ii])):
                logger.logkv(f'val_auc_attr{ii}_group{iii}', round(val_aucs_by_attrs[ii][iii],4))

        for ii in range(len(val_between_group_disparity)):
            logger.logkv(f'val_auc_attr{ii}_std_group_disparity', round(val_between_group_disparity[ii][0],4))
            logger.logkv(f'val_auc_attr{ii}_max_group_disparity', round(val_between_group_disparity[ii][1],4))

        for ii in range(len(val_dpds)):
            logger.logkv(f'val_dpd_attr{ii}', round(val_dpds[ii],4))
        for ii in range(len(val_eods)):
            logger.logkv(f'val_eod_attr{ii}', round(val_eods[ii],4))


        for ii in range(len(test_es_acc)):
            logger.logkv(f'test_es_acc_attr{ii}', round(test_es_acc[ii],4))
        for ii in range(len(test_es_auc)):
            logger.logkv(f'test_es_auc_attr{ii}', round(test_es_auc[ii],4))
        for ii in range(len(test_aucs_by_attrs)):
            for iii in range(len(test_aucs_by_attrs[ii])):
                logger.logkv(f'test_auc_attr{ii}_group{iii}', round(test_aucs_by_attrs[ii][iii],4))

        for ii in range(len(test_between_group_disparity)):
            logger.logkv(f'test_auc_attr{ii}_std_group_disparity', round(test_between_group_disparity[ii][0],4))
            logger.logkv(f'test_auc_attr{ii}_max_group_disparity', round(test_between_group_disparity[ii][1],4))

        for ii in range(len(test_dpds)):
            logger.logkv(f'test_dpd_attr{ii}', round(test_dpds[ii],4))
        for ii in range(len(test_eods)):
            logger.logkv(f'test_eod_attr{ii}', round(test_eods[ii],4))

        logger.dumpkvs()

    if args.perf_file != '':
        if os.path.exists(best_global_perf_file):

            with open(best_global_perf_file, 'a') as f:

                esacc_head_str = ', '.join([f'{x:.4f}' for x in best_es_acc]) + ', '
                esauc_head_str = ', '.join([f'{x:.4f}' for x in best_es_auc]) + ', '

                auc_head_str = ''
                for i in range(len(best_auc_groups)):
                    auc_head_str += ', '.join([f'{x:.4f}' for x in best_auc_groups[i]]) + ', '

                group_disparity_str = ''
                for i in range(len(best_between_group_disparity)):
                    group_disparity_str += ', '.join([f'{x:.4f}' for x in best_between_group_disparity[i]]) + ', '
                
                dpd_head_str = ', '.join([f'{x:.4f}' for x in best_dpd_groups]) + ', '
                eod_head_str = ', '.join([f'{x:.4f}' for x in best_eod_groups]) + ', '

                path_str = f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}'
                f.write(f'{best_ep}, {best_acc:.4f}, {esacc_head_str} {best_auc:.4f}, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n')
                
    best_acc, best_es_acc, best_auc, best_es_auc, best_auc_groups, best_dpd_groups, best_eod_groups, best_between_group_disparity, \
    best_acc_std, best_es_acc_std, best_auc_std, best_es_auc_std, best_auc_groups_std, best_dpd_groups_std, best_eod_groups_std, best_between_group_disparity_std = bootstrap_performance(best_test_preds, best_test_gts, best_test_attrs, num_classes=args.num_classes, bootstrap_repeat_times=args.bootstrap_repeat_times)
    logger.log(f'(mean) best_acc, best_es_acc, best_auc, best_es_auc, best_auc_groups, best_dpd_groups, best_eod_groups, best_between_group_disparity')
    logger.log(best_acc, best_es_acc, best_auc, best_es_auc, best_auc_groups, best_dpd_groups, best_eod_groups, best_between_group_disparity)
    logger.log(f'(std) best_acc, best_es_acc, best_auc, best_es_auc, best_auc_groups, best_dpd_groups, best_eod_groups, best_between_group_disparity')
    logger.log(best_acc_std, best_es_acc_std, best_auc_std, best_es_auc_std, best_auc_groups_std, best_dpd_groups_std, best_eod_groups_std, best_between_group_disparity_std)

    os.rename(args.result_dir, f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}')
