import torch.nn.functional as F
import torch
import os
import logging
from tools.pytorch_utils import Save_Handle
from options import opt
import datetime
import numpy as np

def preprocess_tensors(gts, focal, images): 
    gts = gts.cuda() 
    basize, dim, height, width = focal.size()
    gts1 = F.interpolate(gts, size=(64, 64), mode='bilinear', align_corners=False)
    gts2 = F.interpolate(gts, size=(32, 32), mode='bilinear', align_corners=False)
    gts3 = F.interpolate(gts, size=(16, 16), mode='bilinear', align_corners=False)
    gts4 = F.interpolate(gts, size=(8, 8), mode='bilinear', align_corners=False)
    
    focal_stack = focal
    focal = focal.view(1, basize, dim, height, width).transpose(0, 1)  
    focal = torch.cat(torch.chunk(focal, chunks=12, dim=2), dim=1)  
    focal = torch.cat(torch.chunk(focal, chunks=basize, dim=0), dim=1) 
    focal = focal.view(-1, *focal.shape[2:])  
    
    focal_stack = torch.split(focal_stack,3,dim=1)
    focal_stack = [item.cuda() for item in focal_stack]
    images = images.cuda()
    
    return gts, gts1, gts2, gts3, gts4, focal, images,focal_stack

save_list = Save_Handle(max_num=1)

class ModelSaver:
    def save_regular_checkpoint(self, epoch, model, save_path):
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, '{}_{}.pth'.format(opt.dataset,epoch)))

    def save_interrupt_checkpoint(self, epoch, model, save_path):
        logging.info('Keyboard Interrupt: save model and exit.')
        torch.save(model.state_dict(), os.path.join(save_path, '{}_{}.pth'.format(opt.dataset,epoch + 1)))
        logging.info('save checkpoints successfully!')

    def save_resume_checkpoint(self, epoch, model, optimizer, save_path):
        temp_save_path = os.path.join(save_path, "resume_{}_{}_ckpt.tar".format(opt.dataset,epoch))
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict()
        }, temp_save_path)
        save_list.append(temp_save_path)

def compute_boundary(mask, kernel_size=3):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    if mask.is_cuda:
        sobel_x = sobel_x.cuda()
        sobel_y = sobel_y.cuda()
    sobel_x = sobel_x.type_as(mask)
    sobel_y = sobel_y.type_as(mask)
    edge_x = F.conv2d(mask, sobel_x, padding=1)
    edge_y = F.conv2d(mask, sobel_y, padding=1)
    edge = torch.sqrt(edge_x**2 + edge_y**2)

    return edge

def dice_loss(preds, targets, smooth=1e-6):
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return 1 - dice

def tversky_loss(preds, targets, alpha=0.5, beta=0.5, smooth=1e-6):
    intersection = (preds * targets).sum()
    false_negative = (preds * (1 - targets)).sum()
    false_positive = ((1 - preds) * targets).sum()
    return 1 - (intersection + smooth) / (intersection + alpha * false_negative + beta * false_positive + smooth)

def focal_loss(preds, targets, alpha=0.8, gamma=2.0):
    bce = F.binary_cross_entropy(preds, targets, reduction='none')
    bce_exp = torch.exp(-bce)
    focal_loss = alpha * (1 - bce_exp) ** gamma * bce
    return focal_loss.mean()

def iou_loss(preds, targets, smooth=1e-6):
    intersection = (preds * targets).sum()
    total = (preds + targets).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

def structure_loss(pred, mask):
    weit = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou = 1-(inter+1)/(union-inter+1)
    
    Dice_loss = dice_loss(pred,mask)   
    Tversky_Loss = tversky_loss(pred,mask)
    return (wbce+wiou+Tversky_Loss+Dice_loss).mean()

class Logger:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)

    def log_step(self, epoch, total_epochs, step, total_steps, loss_data, lr):
        message = '\r{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} '.format(
            datetime.datetime.now(), epoch, total_epochs, step, total_steps, loss_data)
        print(message, end='')
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f}'.format(
            epoch, total_epochs, step, total_steps, lr, loss_data))

    def log_epoch(self, epoch, total_epochs, avg_loss):
        message = '#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, total_epochs, avg_loss)
        print(message)
        logging.info(message)
              
def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()#numel用于返回数组中的元素个数
    print(name,' : ' ,'The number of parameters:{}'.format(num_params))
    return num_params

def setup_logging(save_path, model_params, opt):
    logging.basicConfig(filename=save_path + 'log.log', 
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', 
                        level=logging.INFO, 
                        filemode='a', 
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        force=True)

    logging.info("Net-Train")
    logging.info("Config")
    logging.info('params:{}'.format(model_params))
    logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, 
        opt.clip, opt.decay_rate, save_path, opt.decay_epoch))

def test_process_data(gt, focal, image):
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    
    dim, height, width = focal.size()
    basize = 1    
    focal = focal.view(1, basize, dim, height, width).transpose(0, 1)  
    focal = torch.cat(torch.chunk(focal, chunks=12, dim=2), dim=1)      
    focal_stack = [chunk.squeeze(1) for chunk in torch.chunk(focal, chunks=12, dim=1)]
    focal = torch.cat(torch.chunk(focal, chunks=basize, dim=0), dim=1)  
    focal = focal.view(-1, *focal.shape[2:])
    focal_stack = [item.cuda() for item in focal_stack]
    focal = focal.cuda()
    image = image.cuda()

    return gt, focal, image, focal_stack

def compute_mae(res, gt):
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    mae = np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
    return mae

def log_and_save(epoch, mae, best_mae, best_epoch, model, save_path,optimizer):   
    logging.info('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
    print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))

    if epoch == 1 or mae < best_mae:
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'lfsod_epoch_best.pth')
            best_save_path = os.path.join(save_path, "resume_best_ckpt.tar")
            torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict()}, best_save_path)            
        elif epoch == 1:
            best_mae = mae            
    logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))
    return best_mae, best_epoch

def load_pretrained_weights(model, opt):
    if opt.load_mit:
        model.focal_encoder.init_weights(opt.load_mit)
    else:
        print("No pre-train!")



