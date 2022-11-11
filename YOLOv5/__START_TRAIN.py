import argparse
import os
import random

if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] != "YOLOv5":
        os.chdir("YOLOv5")


    os.chdir("YOLOv5_repo")
    #import va fatta qua, dopo il chdir cosi il file "infer" richiamato riesce a trovare le dipendenze
    import YOLOv5_repo.train as train


    # Variables to set for parser
    # Path of train file
    pathTrainSet = '../_train/custom.yaml'

    #Model.yaml Path
    pathYamlBaseWeight = './models/yolov5n.yaml'    # './models/yolov5s.yaml'
    # Model path 
    pathBaseWeight = '../_train/yolov5n.pt'  # '../_train/yolov5s-cls.pt
    # Path to save outputs
    outputDir = '../_train/output'
    # Experiment name, saved to outputDir/name
    name = 'trainExp'
    # Train image size (pixels)
    imgSize = 640
    # Number of samples (images) processed before the model is updated for all GPUs
    batchSize = 32
    # Number of complete passes through the training dataset
    epoches = 800
    # Number of workers simultaneously putting data into RAM (N.B. setting workers to number of cores is a good rule)
    workers = 8
    # The type of the device: cpu, 0 (for GPU), 0,1,2,3,... (for multi-GPU)
    device = '0'
    # Optimizer ['SGD', 'Adam', 'AdamW']
    optimizer = 'SGD'
    # Label smoothing epsilon
    smoothing = 0.0
    # Resume the most recent training
    resume = False
    # EarlyStopping patience (epochs without improvement)
    stoppingPatience = 100
    # Freeze layers [backbone=10, first3=0 1 2]
    freezeLayers = 0
    # Save checkpoint every x epochs, neg value (-1) not save
    saveLastNEpoch = -1
    # Global training seed ( between 0 and 2**32 )
    seed = random.randint(0,2**32)
    #hyperparameters path
    hyp = './data/hyps/hyp.scratch-low.yaml'
    #cache
    cache = "ram"

    # Parser
    parser = argparse.ArgumentParser(description='YOLOv5 Training', add_help=True)
    parser.add_argument('--weights', type=str, default=pathBaseWeight, help='initial weights path')
    parser.add_argument('--cfg', type=str, default=pathYamlBaseWeight, help='model.yaml path')
    parser.add_argument('--data', type=str, default=pathTrainSet, help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=epoches, help='total training epochs')
    parser.add_argument('--hyp', type=str, default=hyp, help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=batchSize, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=imgSize, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=resume, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache',default=cache, type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default=device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default=optimizer, help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=workers, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=outputDir, help='save to project/name')
    parser.add_argument('--name', default=name, help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=smoothing, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=stoppingPatience, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[freezeLayers], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=saveLastNEpoch, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=seed, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')


    train.main(parser.parse_args())
