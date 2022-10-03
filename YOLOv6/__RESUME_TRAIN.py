import tools.train as train
import argparse
import os 


if __name__ == '__main__':       
    if os.path.split(os.getcwd() )[1] != "YOLOv6":
        os.chdir("YOLOv6")


    # Variables to set for parser
    # Resume the most recent training
    resume = True


    # Parser
    parser = argparse.ArgumentParser(description='YOLOv6 Training', add_help=True)
  
    parser.add_argument('--resume', nargs='?', const=True, default=resume, help='resume the most recent training')
 

    train.main(parser.parse_args())
