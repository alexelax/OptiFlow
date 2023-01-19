import argparse
import os

if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] != "YOLOv5":
        os.chdir("YOLOv5")

    os.chdir("YOLOv5_repo")
    #import va fatta qua, dopo il chdir cosi il file "infer" richiamato riesce a trovare le dipendenze
    import YOLOv5_repo.detect as detect

    # Variables to set for parser
    # Model path for inference
    pathInference = '../_infer/best_n.pt'
    # The path with the file to be evaluated
    pathEvalData = '../../Resources/infer_data/traffic1.mp4'
    # Path of train file
    pathTrainSet = '../_train/custom.yaml'
    # The image-size(h,w) in inference size (N.B. the size of the image should be the same of the train)
    imgSize = [320]
    # Confidence threshold for inference
    threshold = 0.25
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
    parser = argparse.ArgumentParser(description='YOLOv5 Infer', add_help=True)
    parser.add_argument('--weights', nargs='+', type=str, default=pathInference, help='model path or triton URL')
    parser.add_argument('--source', type=str, default=pathEvalData, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=pathTrainSet, help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=imgSize, help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=threshold, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=nms, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=maxInfsXImage, help='maximum detections per image')
    parser.add_argument('--device', default=device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=viewImage, action='store_true', help='show results')
    parser.add_argument('--save-txt', default=saveImage,  action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=outputDir, help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=hideLabels, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=hideConfidences, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    detect.main(opt)
