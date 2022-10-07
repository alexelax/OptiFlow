import argparse
import os

if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] != "YOLOv6":
        os.chdir("YOLOv6")

    
    os.chdir("YOLOv6_repo")
    #import va fatta qua, dopo il chdir cosi il file "infer" richiamato riesce a trovare le dipendenze
    import YOLOv6_repo.tools.infer as infer

    # Variables to set for parser
    # Model path for inference
    pathInference = '../_infer/models/train-n-100epoch.pt'
    # The path with the file to be evaluated
    pathEvalData = '../../Resources/infer_data/traffic1.mp4'
    # Path of train file
    pathTrainSet = '../_train/custom.yaml'
    # The image-size(h,w) in inference size (N.B. the size of the image should be the same of the train)
    imgSize = [640, 640]
    # Confidence threshold for inference
    threshold = 0.4
    # NMS IoU threshold for inference (Non-maximum Suppression)
    nms = 0.45
    # Maximal inferences per image
    maxInfsXImage = 1000
    # The type of the device to run the model: cpu, 0 (for GPU), 0,1,2,3,... (for multi-GPU)
    device = '0'
    # Path to save outputs
    outputDir = '../_infer/output'
    # Experiment name, saved to outputDir/name
    name = 'inferExp'
    # Hide labels
    hideLabels = False
    # Hide confidences
    hideConfidences = False
    # saveImage
    saveImage = True
    # viewImage
    viewImage = True

    # Parser
    parser = argparse.ArgumentParser(description='YOLOv6 Inference', add_help=True)
    parser.add_argument('--weights', type=str, default=pathInference, help='model path(s) for inference.')
    parser.add_argument('--source', type=str, default=pathEvalData, help='the source path, e.g. image-file/dir.')
    parser.add_argument('--yaml', type=str, default=pathTrainSet, help='data yaml file.')
    parser.add_argument('--img-size', nargs='+', type=int, default=imgSize,
                        help='the image-size(h,w) in inference size.')
    parser.add_argument('--conf-thres', type=float, default=threshold, help='confidence threshold for inference.')
    parser.add_argument('--iou-thres', type=float, default=nms, help='NMS IoU threshold for inference.')
    parser.add_argument('--max-det', type=int, default=maxInfsXImage, help='maximal inferences per image.')
    parser.add_argument('--device', default=device, help='device to run our model i.e. 0 or 0,1,2,3 or cpu.')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt.')
    parser.add_argument('--save-img', default=saveImage, action='store_false',
                        help='save visuallized inference results.')
    parser.add_argument('--save-dir', type=str, help='directory to save predictions in. See --save-txt.')
    parser.add_argument('--view-img', default=viewImage, action='store_true', help='show inference results')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by classes, e.g. --classes 0, or --classes 0 2 3.')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS.')
    parser.add_argument('--project', default=outputDir, help='save inference results to project/name.')
    parser.add_argument('--name', default=name, help='save inference results to project/name.')
    parser.add_argument('--hide-labels', default=hideLabels, action='store_true', help='hide labels.')
    parser.add_argument('--hide-conf', default=hideConfidences, action='store_true', help='hide confidences.')
    parser.add_argument('--half', action='store_true', help='whether to use FP16 half-precision inference.')

    infer.main(parser.parse_args())
