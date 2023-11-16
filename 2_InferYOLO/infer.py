#TODO: sto implementado anche il track delle auto, ma non va messo qua!! va messo nella "Extrapolate"
#questo codice dovrebbe fare l'infer più velocemente possibile in modo da analizzare i tempi "reali" di YOLO


#SCOPO:
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
from _libs.auto import *



#----------------------------------

settings={
    "saveOutputToFile":False,
    "frameByFrame":False,
    "Track":False,
    

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

        finalSize=320   #640

        frame=resizeInWidth(frame,finalSize)      # Resize the frames to the same size     ( la funzione può essere ottimizzata in base al flusso video )
        #frame=resize(frame,finalSize,finalSize)      # sembra che onnx vuole entrambe le dim a 320/640 ( quadrato ) 

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


    

        

    #butto il primo frame per prendere le dimensioni di ogni singolo flusso ( DA RIVEDERE IL RELEASE? )



    frame=getFrame(caps)
    original_height, original_width = frame.shape[:2]
    original_height=int(original_height/2)
    original_width =int(original_width/2)


    
    outToFile=None
    #per salvare un video su disco
    if settings["saveOutputToFile"]:
        #apro il flusso
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outToFile=cv2.VideoWriter('output.mp4', fourcc, 20.0,(frame.shape[1], frame.shape[0]))      #si, nella "shape" larghezza e altezza sono invertiti... ma porco...



    avrSample=4
    autoNumbersAvr=[ SMA_sequence(avrSample),SMA_sequence(avrSample),SMA_sequence(avrSample),SMA_sequence(avrSample)]

    chrono = Chrono()
    autos=  autoCollection()
    while True:


       
        chrono.start()


        # Acquisizione di un frame dai video
        frame=getFrame(caps)
        if frame is None:
            break

        
        # analisi del frame tramite YOLO
        if settings["Track"]:
            results = model.track(frame)
        else:
            results = model(frame)
      


        

        autoNumbers=[ 0,0,0,0]

        #TODO: vedere se basta usare i centri per vedere se non si "sovrappongono" oppure le bounding box
        centers=[]
        # visualizzazione dei risultati sull'immagine
        #for result in results.xyxy[0]:
        for r in results:
            r:Result
            

            #xc,yc= int((r.start.x+r.end.x)/2),int((r.start.y+r.end.Y)/2)


            c= r.start.middle(r.end)
            valid=True
            for center in centers:
                if c.distance(center)<10:          #10  = da definire
                    valid=False
                    break
            if not valid:
                continue
            centers.append(c)

            

            #TODO: implementare una maschera di selezione per determinare l'index della "corsia" della macchina ( e quelle che sono su corsie sbagliate)



            cv2.rectangle(frame, r.start.unpack(), r.end.unpack(), (0, 0, 255), 2)
            cv2.circle(frame,c.unpack(),2,(255, 0, 0),2)
            #{model.names[r.classNumber]} -> permette di visualizzare il tipo di oggetto ma visto che sono tutti "veichle" non ha tanto senso
            cv2.putText(frame, f'{r.id if r.id!=None else ""}  {r.confidence:.2f}', r.start.moveNew(0,-10).unpack(), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            #metodo ( merdoso ) per identificare la posizione di una macchina e metterla nell'array 
            index=0
            if( c.X>original_width):
                index=2
            
            if( c.Y>original_height):
                index+=1


            autoNumbers[index]+=1

            if r.id!=None:
                a = autos.getOrCreate(r.id)
                a.moved(c.X,c.Y)
                if a.getTrackLen() >2:

                    cv2.polylines(frame,  [np.array(a.getTrack(),np.int32)],   False,(255,0,0),1)

        
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
        if outToFile!=None:
            outToFile.write(frame)

        # interruzione con tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if settings["frameByFrame"]:
        #premere "c" per continuare con il frame successivo
            while not (cv2.waitKey(1) & 0xFF == ord('c')):
                pass
    
        

    #chiudo il flusso 
    if outToFile!=None:
        outToFile.release()

    # rilascio del flusso video e della finestra
    [cap.release() for cap in caps]
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main(settings)