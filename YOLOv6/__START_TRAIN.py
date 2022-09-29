import tools.train as t
import argparse


#TODO: settare i valori di "default" con i valori corretti
parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Training', add_help=True)
parser.add_argument('--data-path', default='./data/coco.yaml', type=str, help='path of dataset')
parser.add_argument('--conf-file', default='./configs/yolov6n.py', type=str, help='experiments description file')
parser.add_argument('--img-size', default=640, type=int, help='train, val image size (pixels)')
parser.add_argument('--batch-size', default=32, type=int, help='total batch size for all GPUs')
parser.add_argument('--epochs', default=400, type=int, help='number of total epochs to run')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--eval-interval', default=20, type=int, help='evaluate at every interval epochs')
parser.add_argument('--eval-final-only', action='store_true', help='only evaluate at the final epoch')
parser.add_argument('--heavy-eval-range', default=50, type=int, help='evaluating every epoch for last such epochs (can be jointly used with --eval-interval)')
parser.add_argument('--check-images', action='store_true', help='check images when initializing datasets')
parser.add_argument('--check-labels', action='store_true', help='check label files when initializing datasets')
parser.add_argument('--output-dir', default='./runs/train', type=str, help='path to save outputs')
parser.add_argument('--name', default='exp', type=str, help='experiment name, saved to output_dir/name')
parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--gpu_count', type=int, default=0)
parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')
parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume the most recent training')
parser.add_argument('--write_trainbatch_tb', action='store_true', help='write train_batch image to tensorboard once an epoch, may slightly slower train speed if open')
parser.add_argument('--stop_aug_last_n_epoch', default=15, type=int, help='stop strong aug at last n epoch, neg value not stop, default 15')
parser.add_argument('--save_ckpt_on_last_n_epoch', default=-1, type=int, help='save last n epoch even not best or last, neg value not save')
parser.add_argument('--distill', action='store_true', help='distill or not')
parser.add_argument('--distill_feat', action='store_true', help='distill featmap or not')
parser.add_argument('--quant', action='store_true', help='quant or not')
parser.add_argument('--calib', action='store_true', help='run ptq')
parser.add_argument('--teacher_model_path', type=str, default=None, help='teacher model path')
parser.add_argument('--temperature', type=int, default=20, help='distill temperature')



t.main(parser.parse_args())

