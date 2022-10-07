import argparse
import os
import YOLOv6_repo.yolov6.utils.general as general


if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] != "YOLOv6":
        os.chdir("YOLOv6")


    os.chdir("YOLOv6_repo")
    #import va fatta qua, dopo il chdir cosi il file "infer" richiamato riesce a trovare le dipendenze
    import YOLOv6_repo.tools.train as train

    #find last pt
    trainPath="..\\_train\\"
    last_pt= general.find_latest_checkpoint("..\\_train\\")

    if last_pt=='':
        print(f"last.pt not found in {trainPath}")


    # Variables to set for parser
    # Resume the most recent training
    # se specifichi il path del pt, lui parte da li
    resume = last_pt        

    # Parser
    parser = argparse.ArgumentParser(description='YOLOv6 Training', add_help=True)
    parser.add_argument('--resume', nargs='?', const=True, default=resume, help='resume the most recent training')

    train.main(parser.parse_args())
