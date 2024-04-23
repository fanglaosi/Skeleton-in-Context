import os
import shutil
import numpy as np
import argparse
import errno
import tensorboardX
from time import time
import random
import prettytable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from lib.utils.tools import *
from lib.utils.learning import *
from lib.data.datareader_h36m import DataReaderH36M
from lib.data.dataset import MotionDataset3D
from lib.model.loss import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='ckpt/default', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('\tSaving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss' : min_loss
    }, chk_path)


def evaluate_future_pose_estimation(args, test_loader, model, epoch=None):
    print('\tEvaluating Future Pose Estimation...')
    model.eval()
    num_samples = 0
    frame_list = [9, 14]
    mpjpe = np.zeros(len(frame_list))

    with torch.no_grad():
        for idx, (prompt_batch, query_batch, task) in enumerate(test_loader):
            assert (task == args.task_to_flag['FPE']).all()
            if torch.cuda.is_available():
                prompt_batch = prompt_batch.cuda()  # (B, clip_len*2, 17, 3)
                query_batch = query_batch.cuda()    # (B, clip_len*2, 17, 3)
            batch_size = len(prompt_batch)
            num_samples += batch_size

            # Model forward
            if args.backbone == 'SiC_dynamicTUP':
                rebuild, target = model(prompt_batch, query_batch, epoch)
            elif args.backbone == 'SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild, target = model(prompt_batch, pseudo_query_batch, query_target, epoch)

            pred = rebuild[:, frame_list, :, :]     # (B,T,17,3)
            gt = target[:, frame_list, :, :]        # (B,T,17,3)
            mpjpe_ = torch.sum(torch.mean(torch.norm(pred*1000 - gt*1000, dim=3), dim=2), dim=0)
            mpjpe += mpjpe_.cpu().data.numpy()
        mpjpe = mpjpe / num_samples     # (T,)
        mpjpe_avg = np.mean(mpjpe)
        summary_table = prettytable.PrettyTable()
        summary_table.field_names = ['FPE'] + ['Avg'] + [f'{(i + 1) * 20}' for i in frame_list]
        summary_table.add_row(['MPJPE'] + [mpjpe_avg] + list(mpjpe))
        summary_table.float_format = ".2"
        return mpjpe_avg, summary_table


def evaluate_motion_completion(args, test_loader, model, epoch=None):
    print('\tEvaluating Motion Completion...')
    model.eval()
    mpjpe_per_ratio = {ratio: 0 for ratio in args.data.drop_ratios_MC}
    count_per_ratio = {ratio: 0 for ratio in args.data.drop_ratios_MC}
    num_samples = 0
    with torch.no_grad():
        for idx, (prompt_batch, query_batch, task) in enumerate(test_loader):
            assert (task == args.task_to_flag['MC']).all()
            if torch.cuda.is_available():
                prompt_batch = prompt_batch.cuda()
                query_batch = query_batch.cuda()
            batch_size = len(prompt_batch)
            num_samples += batch_size

            # Model forward
            if args.backbone == 'SiC_dynamicTUP':
                rebuild, target = model(prompt_batch, query_batch, epoch)
            elif args.backbone == 'SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild, target = model(prompt_batch, pseudo_query_batch, query_target, epoch)

            for i in range(batch_size):
                pred_one_sample = rebuild[i]        # (clip_len, 17, 3)
                gt_one_sample = target[i]           # (clip_len, 17, 3)
                query_input_one_sample = query_batch[i, :args.data.clip_len]    # (clip_len, 17, 3)
                masked_frame_idx = torch.all(query_input_one_sample[:,1:].sum(dim=(0,2), keepdim=True) == 0, dim=0).squeeze(-1)
                masked_frame_idx = torch.cat([torch.tensor([False]).cuda(), masked_frame_idx])
                pred_ = pred_one_sample[:, masked_frame_idx]
                gt_ = gt_one_sample[:, masked_frame_idx]
                masked_frame_num = pred_.shape[1]
                assert masked_frame_num in [int(args.data.num_joints * ratio) for ratio in args.data.drop_ratios_MC]
                mpjpe_ = torch.mean(torch.norm(pred_*1000 - gt_*1000, dim=2))

                for ratio in count_per_ratio:
                    if masked_frame_num == int(ratio * args.data.num_joints):
                        count_per_ratio[ratio] += 1
                        mpjpe_per_ratio[ratio] += mpjpe_.cpu().data.numpy()

        assert sum([cnt for ratio, cnt in count_per_ratio.items()]) == num_samples
        for ratio in count_per_ratio:
            num_samples = count_per_ratio[ratio]
            mpjpe_per_ratio[ratio] = mpjpe_per_ratio[ratio] / num_samples
        mpjpe_avg = np.mean(np.array([err for ratio, err in mpjpe_per_ratio.items()]))

        summary_table = prettytable.PrettyTable()
        summary_table.field_names = ['MC'] + ['Avg'] + [ratio for ratio in args.data.drop_ratios_MC]
        summary_table.add_row(['MPJPE'] + [mpjpe_avg] + [err for ratio, err in mpjpe_per_ratio.items()])
        summary_table.float_format = ".2"
        return mpjpe_avg, summary_table


def evaluate_motion_prediction(args, test_loader, model, epoch=None):
    print('\tEvaluating Motion Prediction...')
    model.eval()
    num_samples = 0
    frame_list = [1, 3, 4, 7, 9]
    mpjpe = np.zeros(len(frame_list))

    with torch.no_grad():
        for idx, (prompt_batch, query_batch, task) in enumerate(test_loader):
            assert (task == args.task_to_flag['MP']).all()
            if torch.cuda.is_available():
                prompt_batch = prompt_batch.cuda()
                query_batch = query_batch.cuda()
            batch_size = len(prompt_batch)
            num_samples += batch_size

            # Model forward
            if args.backbone == 'SiC_dynamicTUP':
                rebuild, target = model(prompt_batch, query_batch, epoch)
            elif args.backbone == 'SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild, target = model(prompt_batch, pseudo_query_batch, query_target, epoch)

            pred = rebuild[:, frame_list, :, :].clone()     # (B,T,17,3)
            gt = target[:, frame_list, :, :].clone()        # (B,T,17,3)

            mpjpe_ = torch.sum(torch.mean(torch.norm(pred*1000 - gt*1000, dim=3), dim=2), dim=0)
            mpjpe += mpjpe_.cpu().data.numpy()

        mpjpe = mpjpe / num_samples     # (T,)
        mpjpe_avg = np.mean(mpjpe)
        summary_table = prettytable.PrettyTable()
        summary_table.field_names = ['MP'] + ['Avg'] + [f'{(i + 1) * 40}' for i in frame_list]
        summary_table.add_row(['MPJPE'] + [mpjpe_avg] + list(mpjpe))
        summary_table.float_format = ".2"
        return mpjpe_avg, summary_table


def evaluate_pose_estimation(args, model_pos, test_loader, datareader, epoch=None):
    print('\tEvaluating 3D Pose Estimation...')
    results_all = []
    model_pos.eval()

    with torch.no_grad():
        for idx, (prompt_batch, query_batch, task) in enumerate(test_loader):
            assert (task == args.task_to_flag['PE']).all()
            if torch.cuda.is_available():
                prompt_batch = prompt_batch.cuda()
                query_batch = query_batch.cuda()
            batch_size = len(prompt_batch)

            # Model forward
            if args.backbone == 'SiC_dynamicTUP':
                rebuild_part, target_part = model_pos(prompt_batch, query_batch, epoch)
            elif args.backbone == 'SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild_part, target_part = model_pos(prompt_batch, pseudo_query_batch, query_target, epoch)

            if args.flip_h36m_y_axis:
                rebuild_part[:, :, :, 1] = -rebuild_part[:, :, :, 1]
            if args.rootrel_target_PE:
                rebuild_part[:, :, 0, :] = 0
            scale_h36m_skel = args.get('scale_h36m_skeleton', 1.0)
            if scale_h36m_skel != 1.0:
                rebuild_part = rebuild_part / scale_h36m_skel

            results_all.append(rebuild_part.cpu().numpy())

    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
    block_list = ['s_09_act_05_subact_02', 's_09_act_10_subact_02', 's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:,None,None]     # (96,1,1)
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        # Root-relative Errors
        pred = pred - pred[:,0:1,:]
        gt = gt - gt[:,0:1,:]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)

    final_result = []
    final_result_procrustes = []
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))

    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['PE'] + ['Avg'] + action_names
    summary_table.add_row(['MPJPE'] + [e1] + final_result)
    summary_table.add_row(['P-MPJPE'] + [e2] + final_result_procrustes)
    summary_table.float_format = ".2"

    return e1, e2, summary_table


def train_epoch(args, model_pos, train_loader, losses, optimizer, epoch=None):
    model_pos.train()
    for idx, (prompt_batch, query_batch, task_flag) in enumerate(train_loader):
        if torch.cuda.is_available():
            prompt_batch = prompt_batch.cuda()
            query_batch = query_batch.cuda()
        batch_size = len(prompt_batch)

        # Model forward
        if args.backbone == 'SiC_dynamicTUP':
            rebuild_part, target_part = model_pos(prompt_batch, query_batch, epoch)    # (N,T,17,3)
        elif args.backbone == 'SiC_staticTUP':
            avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'avg_pose.npy'))).float().cuda()
            avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            query_input = query_batch[:, :args.data.clip_len]
            query_target = query_batch[:, args.data.clip_len:]
            pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
            rebuild_part, target_part = model_pos(prompt_batch, pseudo_query_batch, query_target, epoch)    # (N,T,17,3)
        else:
            raise ValueError('Undefined backbone type.')

        # Optimize
        optimizer.zero_grad()
        loss_total = 0
        for loss_name, loss_dict in losses.items():
            if loss_name == 'total':
                continue
            if loss_name == 'limb_var':
                loss = loss_dict['loss_function'](rebuild_part)
            else:
                loss = loss_dict['loss_function'](rebuild_part, target_part)
            loss_dict['loss_logger'].update(loss.item(), batch_size)
            weight = loss_dict['loss_weight']
            if weight != 0:
                loss_total += loss * weight
        losses['total']['loss_logger'].update(loss_total.item(), batch_size)
        loss_total.backward()
        optimizer.step()

        if idx in [len(train_loader) * i // 3 for i in range(1, 3)]:
            task_cnt = {task: 0 for task in args.tasks}
            for i in range(batch_size):
                task_cnt[args.flag_to_task[f'{task_flag[i]}']] += 1
            print(f"\tIter: {idx}/{len(train_loader)}; current batch has {task_cnt} samples")


def train_with_config(args, opts):

    assert 'bin' not in opts.checkpoint
    if args.use_partial_data:
        args.data = args.partial_data
    else:
        args.data = args.full_data
    print(f'Training on {len(args.tasks)} tasks: {args.tasks}')
    print(f'\nConfigs: {args}')

    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    
    if not os.path.exists(os.path.join(opts.checkpoint, os.path.basename(opts.config))):
        shutil.copy(opts.config, opts.checkpoint)

    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    print('\nLoading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 12,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    testloader_params = {
          'batch_size': args.test_batch_size,
          'shuffle': False,
          'num_workers': 12,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }


    train_dataset = MotionDataset3D(args, data_split='train')        
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)

    test_dataset = MotionDataset3D(args, data_split='test', prompt_list=train_dataset.prompt_list)
    
    print('Training sample count:', len(train_dataset))
    print('Testing sample count:', len(test_dataset))

    dataloader_dict = {}
    for task in args.tasks:
        dataloader_dict[task] = DataLoader(Subset(test_dataset, test_dataset.global_idx_list[task]), **testloader_params)

    eval_dict = {}
    for task in args.tasks:
        eval_dict[task] = {'min_err': 1000000, 'best_epoch': 1000000}
    eval_dict['all'] = {'min_err': 1000000, 'best_epoch': 1000000}

    if 'PE' in args.tasks:
        datareader_pose_estimation = DataReaderH36M(n_frames=args.data.clip_len, sample_stride=args.data.sample_stride, data_stride_train=args.data.train_stride, data_stride_test=args.data.clip_len, dt_root=args.data.root_path, dt_file=args.data.source_file_h36m)

    print('\nLoading model...')
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print(f'Trainable parameter count: {model_params/1000000}M')

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone

    if not opts.evaluate:
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0

        print('\nTraining on {} batches for {} epochs'.format(len(train_loader_3d), args.epochs))

        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                if 'PE' in eval_dict.keys():
                    eval_dict['PE']['min_err'] = checkpoint['min_loss']
                else:
                    eval_dict[list(eval_dict.keys())[0]]['min_err'] = checkpoint['min_loss']

        # Print global multitask results throughout training
        summary_table = prettytable.PrettyTable()
        summary_table.field_names = ['Epoch'] + [metric for task in args.tasks for metric in args.task_metrics[task]]

        # Training
        training_start_time = time()
        epoch_to_eval = [ep for ep in range(0,st+10, 2)] + \
                        [ep for ep in range(st+10,args.epochs-20, 5)] + \
                        [ep for ep in range(args.epochs-20, args.epochs, 2)]
        for epoch in range(st, args.epochs):
            print(f'[{epoch+1} start]')
            start_time = time()

            # Loss initialization at epoch start
            losses = {}
            for loss_name, loss_weight in args.losses.items():
                if loss_name == 'mpjpe':
                    loss_function = loss_mpjpe
                elif loss_name == 'n_mpjpe':
                    loss_function = n_mpjpe
                elif loss_name == 'velocity':
                    loss_function = loss_velocity
                elif loss_name == 'limb_var':
                    loss_function = loss_limb_var
                elif loss_name == 'limb_gt':
                    loss_function = loss_limb_gt
                elif loss_name == 'angle':
                    loss_function = loss_angle
                elif loss_name == 'angle_velocity':
                    loss_function = loss_angle_velocity
                else:
                    raise ValueError('Unknown loss type.')
                losses[loss_name] = {'loss_logger': AverageMeter(),
                                     'loss_weight': loss_weight, 'loss_function': loss_function}
            losses['total'] = {'loss_logger': AverageMeter()}

            # One epoch forward
            train_epoch(args, model_pos, train_loader_3d, losses, optimizer, epoch=epoch)

            # Post-epoch evaluation and loss logger update
            if args.no_eval:
                elapsed = (time() - start_time) / 60
                print(f"[{epoch+1} end] Time cost: {elapsed:.2f}min \t| lr: {lr:.8f} \t| train loss: {losses['mpjpe']['loss_logger'].avg :.6f}")
            else:
                if epoch in epoch_to_eval:
                    epoch_eval_results = {}
                    if 'PE' in args.tasks:
                        e1, e2, summary_table_PE = evaluate_pose_estimation(args, model_pos, dataloader_dict['PE'], datareader_pose_estimation, epoch=epoch)
                        epoch_eval_results['PE e1'] = e1; epoch_eval_results['PE e2'] = e2
                        train_writer.add_scalar('PE Error P1', e1, epoch + 1)
                        train_writer.add_scalar('PE Error P2', e2, epoch + 1)
                    if 'FPE' in args.tasks:
                        e1FPE, summary_table_FPE = evaluate_future_pose_estimation(args, dataloader_dict['FPE'], model_pos, epoch=epoch)
                        epoch_eval_results['FPE'] = e1FPE
                        train_writer.add_scalar('FPE MPJPE', e1FPE, epoch + 1)

                    if 'MP' in args.tasks:
                        mpjpe, summary_table_MP = evaluate_motion_prediction(args, dataloader_dict['MP'], model_pos, epoch=epoch)
                        epoch_eval_results['MP'] = mpjpe
                        train_writer.add_scalar('MP MPJPE', mpjpe, epoch + 1)

                    if 'MC' in args.tasks:
                        min_err_mc, summary_table_MC = evaluate_motion_completion(args, dataloader_dict['MC'], model_pos, epoch=epoch)
                        epoch_eval_results['MC'] = min_err_mc
                        train_writer.add_scalar('MC MPJPE', min_err_mc, epoch + 1)

                    summary_table.add_row([epoch+1] + [epoch_eval_results[metric] for task in args.tasks for metric in args.task_metrics[task]])

                for loss_name, loss_dict in losses.items():
                    train_writer.add_scalar(loss_name, loss_dict['loss_logger'].avg, epoch + 1)

            # Save checkpoints
            chk_path = os.path.join(opts.checkpoint, f'epoch_{epoch}.bin')
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            chk_path_best = {task: os.path.join(opts.checkpoint, f'best_epoch_{task}.bin') for task in args.tasks}
            chk_path_best['all'] = os.path.join(opts.checkpoint, 'best_epoch_all.bin')

            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, eval_dict['PE']['min_err'])

            # Save best checkpoint according to global best 
            if not args.no_eval:
                if epoch in epoch_to_eval:
                    if 'PE' in args.tasks and e1 < eval_dict['PE']['min_err']:
                        eval_dict['PE']['min_err'] = e1
                        eval_dict['PE']['best_epoch'] = epoch + 1
                        save_checkpoint(chk_path_best['PE'], epoch, lr, optimizer, model_pos, eval_dict['PE']['min_err'])
                        
                    if 'MP' in args.tasks and mpjpe < eval_dict['MP']['min_err']:
                        eval_dict['MP']['min_err'] = mpjpe
                        eval_dict['MP']['best_epoch'] = epoch + 1
                        save_checkpoint(chk_path_best['MP'], epoch, lr, optimizer, model_pos, eval_dict['MP']['min_err'])
                    
                    if 'FPE' in args.tasks and e1FPE < eval_dict['FPE']['min_err']:
                        eval_dict['FPE']['min_err'] = e1FPE
                        eval_dict['FPE']['best_epoch'] = epoch + 1
                        save_checkpoint(chk_path_best['FPE'], epoch, lr, optimizer, model_pos, eval_dict['FPE']['min_err'])
                    
                    if 'MC' in args.tasks and min_err_mc < eval_dict['MC']['min_err']:
                        eval_dict['MC']['min_err'] = min_err_mc
                        eval_dict['MC']['best_epoch'] = epoch + 1
                        save_checkpoint(chk_path_best['MC'], epoch, lr, optimizer, model_pos, eval_dict['MC']['min_err'])

                    if (e1 + mpjpe + e1FPE + min_err_mc) / 4 < eval_dict['all']['min_err']:
                        eval_dict['all']['min_err'] = (e1 + mpjpe + e1FPE + min_err_mc) / 4
                        eval_dict['all']['best_epoch'] = epoch + 1
                        save_checkpoint(chk_path_best['all'], epoch, lr, optimizer, model_pos, eval_dict['all']['min_err'])


                    # Print evaluation results
                    if 'PE' in args.tasks:
                        print(summary_table_PE)
                    if 'MP' in args.tasks:
                        print(summary_table_MP)
                    if 'MC' in args.tasks:
                        print(summary_table_MC)
                    if 'FPE' in args.tasks:
                        print(summary_table_FPE)

                elapsed = (time() - start_time) / 60
                print(f"[{epoch+1} end] Time cost: {elapsed:.2f}min \t| lr: {lr:.8f} \t| train loss: {losses['mpjpe']['loss_logger'].avg :.6f}")

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
        
        print(f"Training took {(time() - training_start_time) / 3600 :.2f}h")

        if not args.no_eval:
            summary_table.float_format = ".2"
            print("All results:")
            print(summary_table)
            print("Best results:")
            for task in args.tasks:
                print(f"{task} \t | \t {eval_dict[task]['min_err']}mm (epoch {eval_dict[task]['best_epoch']})")
            print(f"All \t | \t {eval_dict['all']['min_err']}mm (epoch {eval_dict['all']['best_epoch']})")

    if opts.evaluate:
        epoch = checkpoint['epoch']
        if 'PE' in args.tasks:
            _, _, summary_table_PE = evaluate_pose_estimation(args, model_pos, dataloader_dict['PE'], datareader_pose_estimation, epoch=epoch)
            print(summary_table_PE)
        if 'MP' in args.tasks:
            _, summary_table_MP = evaluate_motion_prediction(args, dataloader_dict['MP'], model_pos, epoch=epoch)
            print(summary_table_MP)
        if 'MC' in args.tasks:
            _, summary_table_MC = evaluate_motion_completion(args, dataloader_dict['MC'], model_pos, epoch=epoch)
            print(summary_table_MC)
        if 'FPE' in args.tasks:
            _, summary_table_FPE = evaluate_future_pose_estimation(args, dataloader_dict['FPE'], model_pos, epoch=epoch)
            print(summary_table_FPE)

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)
