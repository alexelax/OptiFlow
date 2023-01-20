api_key="NYrApM4g6udf8kLAw687"
WORKSPACE_ID="kmutt-wr7fp"
PROJECT_ID="rockpaperscissors-hjwgd"

from roboflow import Roboflow
import os
os.environ["DATASET_DIRECTORY"] = "/content/datasets"
rf = Roboflow(api_key=api_key)
project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)

VERSION =int(project.versions()[0].version.split("/")[-1])   #prende l'ultima versione disponibile su roboflow (dovrebbe funzionare... gli id contano da 1 )
dataset = project.version(VERSION).download("yolov5")