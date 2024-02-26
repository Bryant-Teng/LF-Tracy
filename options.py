import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='lfsod', help='dataset name')
parser.add_argument('--epoch', type=int, default=500, help='epoch number')
parser.add_argument('--model_name', type=str, default="LFNet", help='model name')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--load_mit', type=str, default='/media/tf/HDisk/Pfolder/PyProj/SODBest/bestlf/pretain_weight/pvt_v2_b2.pth', help='train from checkpoints')#default=r'.\pretrained_params\pvt_v2_b2.pth'
#parser.add_argument('--load_mit', type=str, default=r'/root/autodl-tmp/SOD/BaiduNetdiskDownload/预训练参数/pvt_v2_b2.pth', help='train from checkpoints')#default=r'.\pretrained_params\pvt_v2_b2.pth'
parser.add_argument('--gpu_id', type=str, default='0,1', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='/media/tf/HDisk/Pfolder/PyProj/SOD/adataset/train/train_images/', help='the training rgb images root')  #default='./dataset/focal_stack/trainset/noargument/train_images/'
parser.add_argument('--fs_root', type=str, default='/media/tf/HDisk/Pfolder/PyProj/SOD/adataset/train/train_focals/', help='the training depth images root')
#default='./dataset/focal_stack/trainset/noargument/train_focals/'
parser.add_argument('--gt_root', type=str, default='/media/tf/HDisk/Pfolder/PyProj/SOD/adataset/train/train_masks/', help='the training gt images root')
#default='./dataset/focal_stack/trainset/noargument/train_masks/'
parser.add_argument('--test_rgb_root', type=str, default='/media/tf/HDisk/Pfolder/PyProj/SOD/adataset/test/LFSD/test_images/', help='the test gt images root')
#default='./dataset/focal_stack/test_in_train/test_images/'
parser.add_argument('--test_fs_root', type=str, default='/media/tf/HDisk/Pfolder/PyProj/SOD/adataset/test/LFSD/test_focals/', help='the test fs images root')
#default='./dataset/focal_stack/test_in_train/test_focals/'
parser.add_argument('--test_gt_root', type=str, default='/media/tf/HDisk/Pfolder/PyProj/SOD/adataset/test/LFSD/test_masks/', help='the test gt images root')
#default='./dataset/focal_stack/test_in_train/test_masks/'
parser.add_argument('--save_path', type=str, default='./lfsod_cpts/', help='the path to save models and logs')
parser.add_argument('--local_rank', type=int, default=1, help='Local rank for distributed training')
parser.add_argument('--DDP', action='store_true', default=False, help='Single or Multi GPUs')
parser.add_argument('--resume', action='store_true', default=False, help='resume training processs')
parser.add_argument('--load_resume', type=str, default='/root/autodl-tmp/第二次/Code/lfsod_cpts/lfsod_epoch_best.pth', help='resume_checkpoint')

opt = parser.parse_args()