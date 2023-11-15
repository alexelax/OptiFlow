
#aprire 4 flussi video
#ciascun frame unirlo in un unico flusso video
#analizzare il frame con YOLO
#visualizzare l'output con cv2


import math


import numpy as np
from shapely.geometry import polygon

import cv2


#recupero il path del file corrente, prendo la cartella padre e la aggiunto al sys.path ( variabili di sistema del processo )
#cosi trova le librerie della cartella Libs
import os,sys
dir_path = os.path.dirname(os.path.realpath(__file__))+"\.."
sys.path.append(dir_path)



from  _libs.SMA import SMA_sequence
from  _libs.cv2_helper import *
from _libs.compatibilityLayer import *
from _libs.chrono import Chrono


#----------------------------------

settings={
    "saveOutputToFile":False,
    "frameByFrame":False,

}

#-------------------------------




def getFrame(caps):
    """
    prende i frame e ne fa il collage
    caps è un vettori di VideoCapture
     
    """
    frames=[]
    for cap in caps:
        ret, frame = cap.read()

        if not ret:
            #return None        -> chiude il programma
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)      #resetto il video da capo
            ret, frame = cap.read()

 
        frame=resizeInWidth(frame,320)      # Resize the frames to the same size     ( la funzione può essere ottimizzata in base al flusso video )
        #frame=resize(frame,320,320)      # sembra che onnx vuole entrambe le dim a 640

        frames.append(frame)


    #collage frame
    frame = concat_tile([
        [frames[0],frames[1]],
        [frames[2],frames[3]]]
        )

    return frame
        




def  main(settings):

    cv2_namedWindow('Video')       
    cv2_moveWindow('Video', 0,0)  



    # Creazione del modello YOLO
    #model=ModelCompatibilityLayerV5('../YOLOv5/YOLOv5_repo','../pts/yolov5/best_n.pt')
    #model=ModelCompatibilityLayerV5_TensorRT('../YOLOv5/YOLOv5_repo','../pts/yolov5/best_n.engine')       #TensorRT da testare su linux ( install su win di tensorrt è un dito nel culo)
    model=ModelCompatibilityLayerV8('../YOLOv8/PARAMETRO_NON_USATO','../pts/yolov8/best_n.pt')    
    

    #CPU
    #model=ModelCompatibilityLayerV5('../YOLOv5/YOLOv5_repo','../pts/yolov5/best_n.onnx')      
    #model=ModelCompatibilityLayerV5('../YOLOv5/YOLOv5_repo','../pts/yolov5/best_n_openvino_model/')

    
    #check per vedere se è stato scelto un modello qua sopra...
    try:
        model
    except NameError:
        assert False, "Scegli un modello!!"
    

    

    # Apertura del flusso video

    #0 per la webcam, 1 per una seconda webcam o un file video
    caps=[
        cv2.VideoCapture("../Resources/infer_data/traffic1.mp4"),
        cv2.VideoCapture("../Resources/infer_data/traffic2.mp4"),
        cv2.VideoCapture("../Resources/infer_data/traffic2.mp4"),
        cv2.VideoCapture("../Resources/infer_data/traffic1.mp4") 
    ]


    outToFile=None
    #per salvare un video su disco
    if settings["saveOutputToFile"]:
        #apro il flusso
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        outToFile=cv2.VideoWriter('output.mp4', fourcc, 20.0,(640, 320))

        

        


    #butto il primo frame per prendere le dimensioni di ogni singolo flusso ( DA RIVEDERE IL RELEASE? )
    frame=getFrame(caps)
    original_height, original_width = frame.shape[:2]
    original_height=int(original_height/2)
    original_width =int(original_width/2)

    avrSample=4
    autoNumbersAvr=[ SMA_sequence(avrSample),SMA_sequence(avrSample),SMA_sequence(avrSample),SMA_sequence(avrSample)]

    chrono = Chrono()
    while True:


       
        chrono.start()


        # Acquisizione di un frame dai video
        frame=getFrame(caps)
        if frame is None:
            break

        
        # analisi del frame tramite YOLO
        results = model(frame)
        #results = model.track(frame)       #TODO: solo in v8 -> x traking


        

        autoNumbers=[ 0,0,0,0]

        #TODO: vedere se basta usare i centri per vedere se non si "sovrappongono" oppure le bounding box
        centers=[]
        # visualizzazione dei risultati sull'immagine
        #for result in results.xyxy[0]:
        for result in results:
            x1, y1, x2, y2,confidence,classNumber = result
            xc,yc= int((x1+x2)/2),int((y1+y2)/2)


            c=(xc,yc)
            valid=True
            for center in centers:
                if math.dist(center, c)<10:          #10  = da definire
                    valid=False
                    break
            if not valid:
                continue
            centers.append(c)

            

            #TODO: implementare una maschera di selezione per determinare l'index della "corsia" della macchina ( e quelle che sono su corsie sbagliate)

            

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame,(xc,yc),2,(255, 0, 0),2)
            cv2.putText(frame, f'{model.names[classNumber]} {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            #TODO: x traking -> vedere l'id dell'oggetto
            #cv2.putText(frame, f'{id} {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            #metodo ( merdoso ) per identificare la posizione di una macchina e metterla nell'array 
            index=0
            if( xc>original_width):
                index=2
            
            if( yc>original_height):
                index+=1


            autoNumbers[index]+=1

        
        #aggiorno le medie delle macchine
        for id,x in enumerate(autoNumbers):
            autoNumbersAvr[id].AddValue(x)
        #sostituisco il conteggio con la media
        autoNumbers = [round(auto.current) for auto in autoNumbersAvr]

        print(chrono.stop())
        

        #--------------------------------------------------------
        #da questo punto in poi, posso fare le operazioni "lente" 
        #--------------------------------------------------------

        # visualizzazione dell'immagine con i risultati
        cv2.imshow('Video', frame)


        #salvo il frame nell'output
        if settings["saveOutputToFile"]:
            outToFile.write(frame)

        # interruzione con tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if settings["frameByFrame"]:
        #premere "c" per continuare con il frame successivo
            while not (cv2.waitKey(1) & 0xFF == ord('c')):
                pass
    
        

    #chiudo il flusso 
    if settings["saveOutputToFile"]:
        outToFile.release()

    # rilascio del flusso video e della finestra
    [cap.release() for cap in caps]
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main(settings)