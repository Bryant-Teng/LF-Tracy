
import os
import logging
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tools.data import test_dataset, SalObjDataset
from model.builder import model
from options import opt
from utils import clip_gradient, adjust_lr
from tools.train import (
    print_network,
    setup_logging,
    load_pretrained_weights,
    ModelSaver,
    Logger,
    preprocess_tensors,
    structure_loss,
    test_process_data,
    compute_mae,
    log_and_save
)

from tools.mixup import mixup_images as mi
from tools.mixup import mixup_images2 as mi2

def train(train_loader, model, optimizer, epoch, save_path):
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, focal) in enumerate(train_loader, start=1):            
            optimizer.zero_grad()
                      
            gts, gts1, gts2, gts3, gts4, focal, images,focal_stack = preprocess_tensors(gts, focal, images)
            images,focal_stack = mi2(images,focal_stack)
            out,out1,out2,out3,out4= model(focal,focal_stack, images)
            loss = structure_loss(out, gts)+structure_loss(out1, gts1)+structure_loss(out2, gts2)+structure_loss(out3, gts3)+structure_loss(out4, gts4)
            loss.backward()            
            clip_gradient(optimizer, opt.clip) 
            optimizer.step()
            epoch_step += 1  
            loss_all += loss.data

            if not opt.DDP or dist.get_rank() == 0:
                logger.log_step(epoch, opt.epoch, i, Iter, loss.data, optimizer.state_dict()['param_groups'][0]['lr'])

        loss_all /= epoch_step   
        if not opt.DDP or dist.get_rank() == 0:
            logger.log_epoch(epoch, opt.epoch, loss_all)
            if (epoch) % 5 == 0:
                saver.save_regular_checkpoint(epoch, model, save_path) 
            saver.save_resume_checkpoint(epoch, model, optimizer, save_path) if not opt.DDP or dist.get_rank() == 0 else None            
    except KeyboardInterrupt:
        if not opt.DDP or dist.get_rank() == 0:
            saver.save_interrupt_checkpoint(epoch, model, save_path)            
        raise

def test(test_loader, model, epoch, save_path,optimizer):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, focal, gt, name = test_loader.load_data()
            gt, focal, image, focal_stack = test_process_data(gt, focal, image)
            out,out1,out2,out3,out4 = model(focal, focal_stack, image) 
            mae_sum += compute_mae(out, gt)
        mae = mae_sum / test_loader.size
        best_mae, best_epoch = log_and_save(epoch, mae, best_mae, best_epoch, model, save_path,optimizer)

if __name__ == '__main__':
    logging.info("Start train...")       
    if opt.DDP == True: 
        local_rank = int(os.environ.get("LOCAL_RANK", 0))                         
        torch.cuda.set_device(local_rank)
        cudnn.benchmark = True
        dist.init_process_group(backend='nccl')
        print('opt.DDp',opt.DDP) if dist.get_rank() == 0 else None
        print("GPU available:", ",".join([str(i) for i in range(torch.cuda.device_count())])) if dist.get_rank() == 0 else None
    else:
        print('Single GPU')
    
    model = model() 
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    if opt.resume == False:  #Resume is not finished in this version
        start_epoch = 0
        load_pretrained_weights(model, opt)
    else:
        print('Start Resume')
        checkpoint = torch.load(opt.load_resume)
        for key in checkpoint.keys():
            print(key)
        
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  #weight_decay正则化系数
        start_epoch = checkpoint['epoch']
        print('Resume:  start_epoch{}'.format(start_epoch))
    model.cuda()      
    if not opt.DDP or dist.get_rank() == 0:
        model_params = print_network(model, 'lf_pvt')
    os.makedirs(opt.save_path, exist_ok=True)
    if opt.DDP: model = DistributedDataParallel(model, find_unused_parameters=True)
    if opt.DDP == True:
        print('load data...') if dist.get_rank() == 0 else None

    train_dataset = SalObjDataset(opt.rgb_root, opt.gt_root, opt.fs_root,trainsize=opt.trainsize)
    if opt.DDP == True:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt.batchsize, shuffle=False, pin_memory=True, sampler=train_sampler)
    else:
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt.batchsize, shuffle=True, pin_memory=True)
    
    test_loader = test_dataset(opt.test_rgb_root, opt.test_gt_root, opt.test_fs_root,testsize=opt.trainsize)
    Iter = len(train_loader)

    if not opt.DDP or dist.get_rank() == 0:
        setup_logging(opt.save_path, model_params, opt)
    best_mae = 1
    best_epoch = 0
    saver = ModelSaver()
    logger = Logger()
    
    for epoch in range(start_epoch, opt.epoch+1):       
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)       
        train(train_loader, model, optimizer, epoch, opt.save_path)
        test(test_loader, model, epoch, opt.save_path,optimizer)





































