import argparse
import os

if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] != "YOLOv6":
        os.chdir("YOLOv6")


    os.chdir("YOLOv6_repo")
    #import va fatta qua, dopo il chdir cosi il file "infer" richiamato riesce a trovare le dipendenze
    import YOLOv6_repo.tools.train as train


    # Variables to set for parser
    # Path of train file
    pathTrainSet = '../_train/custom.yaml'
    # Config description file
    pathConfigFile = './configs/yolov6n.py'
    # Path to save outputs
    outputDir = '../_train/output'
    # Experiment name, saved to outputDir/name
    name = 'trainExp'
    # Train image size (pixels)
    imgSize = 640
    # Number of samples (images) processed before the model is updated for all GPUs
    batchSize = 32
    # Number of complete passes through the training dataset
    epoches = 100
    # Number of workers simultaneously putting data into RAM (N.B. setting workers to number of cores is a good rule)
    workers = 8
    # The type of the device: cpu, 0 (for GPU), 0,1,2,3,... (for multi-GPU)
    device = '0'
    # The number of GPU's
    gpuCount = 1
    # After how many epochs does the model evaluate
    evalInterval = 20
    # Evaluating every epoch for last such epochs (can be jointly used with evalInterval)'
    heavyEvalRange = 50
    # Url used to set up distributed training
    distUrl = 'trn://'
    # DDP parameter (Distributed Data Parallel)
    localRank = -1
    # Resume the most recent training
    resume = False
    # Stop strong aug at last n epoch, neg value (-1) not stop
    stopAugLastNEpoch = 15
    # Save last n epoch even not best or last, neg value (-1) not save
    saveLastNEpoch = -1
    # If is present the teacher model path
    teacherModelPath = None
    # Number used to control the randomness of predictions
    temperature = 20

    # Parser
    parser = argparse.ArgumentParser(description='YOLOv6 Training', add_help=True)
    parser.add_argument('--data-path', default=pathTrainSet, type=str, help='path of train')
    parser.add_argument('--conf-file', default=pathConfigFile, type=str, help='experiments description file')
    parser.add_argument('--img-size', default=imgSize, type=int, help='train, val image size (pixels)')
    parser.add_argument('--batch-size', default=batchSize, type=int, help='total batch size for all GPUs')
    parser.add_argument('--epochs', default=epoches, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=workers, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--device', default=device, type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--eval-interval', default=evalInterval, type=int, help='evaluate at every interval epochs')
    parser.add_argument('--eval-final-only', action='store_true', help='only evaluate at the final epoch')
    parser.add_argument('--heavy-eval-range', default=heavyEvalRange, type=int,
                        help='evaluating every epoch for last such epochs (can be jointly used with --eval-interval)')
    parser.add_argument('--check-images', action='store_true', help='check images when initializing datasets') 
    parser.add_argument('--check-labels', action='store_true', help='check label files when initializing datasets')
    parser.add_argument('--output-dir', default=outputDir, type=str, help='path to save outputs')
    parser.add_argument('--name', default=name, type=str, help='experiment name, saved to output_dir/name')
    parser.add_argument('--dist_url', default=distUrl, type=str, help='url used to set up distributed training')
    parser.add_argument('--gpu_count', type=int, default=gpuCount)
    parser.add_argument('--local_rank', type=int, default=localRank, help='DDP parameter')
    parser.add_argument('--resume', nargs='?', const=True, default=resume, help='resume the most recent training')
    parser.add_argument('--write_trainbatch_tb', action='store_true',
                        help='write train_batch image to tensorboard once an epoch, may slightly slower train speed if open')
    parser.add_argument('--stop_aug_last_n_epoch', default=stopAugLastNEpoch, type=int,
                        help='stop strong aug at last n epoch, neg value not stop, default 15')
    parser.add_argument('--save_ckpt_on_last_n_epoch', default=saveLastNEpoch, type=int,
                        help='save last n epoch even not best or last, neg value not save')
    parser.add_argument('--distill', action='store_true', help='distill or not')
    parser.add_argument('--distill_feat', action='store_true', help='distill featmap or not')
    parser.add_argument('--quant', action='store_true', help='quant or not')
    parser.add_argument('--calib', action='store_true', help='run ptq')
    parser.add_argument('--teacher_model_path', type=str, default=teacherModelPath, help='teacher model path')
    parser.add_argument('--temperature', type=int, default=temperature, help='distill temperature')

    train.main(parser.parse_args())
