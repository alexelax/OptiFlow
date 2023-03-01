
#aprire 4 flussi video
#ciascun frame unirlo in un unico flusso video
#analizzare il frame con YOLO
#visualizzare l'output con cv2


import math
import cv2
import time
import numpy as np
from compatibilityLayer import *
from SMA import SMA_sequence
from shapely.geometry import polygon

#TODO: modificare tutti i rif di cv2 in questo modo ( dovrebbe essere più veloce)
from cv2 import namedWindow as cv2_namedWindow
from cv2 import moveWindow as cv2_moveWindow

def resizeInWidth(frame,width):
    original_height, original_width = frame.shape[:2]
    # Set the desired width of the resized frame

    # Calculate the aspect ratio of the original frame
    aspect_ratio = original_height / original_width

    # Calculate the height of the resized frame
    desired_height = int(width * aspect_ratio)

    # Resize the frame
    return cv2.resize(frame, (width, desired_height))
def resizeInHeigth(frame,height):
    original_height, original_width = frame.shape[:2]
    # Set the desired width of the resized frame

    # Calculate the aspect ratio of the original frame
    aspect_ratio = original_height / original_width

    # Calculate the height of the resized frame
    desired_width = int(height / aspect_ratio)

    # Resize the frame
    return cv2.resize(frame, (height, desired_width))
def resize(frame,height,width):
    # Resize the frame
    return cv2.resize(frame, (height, width))



def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])



def showCrossroads(autos,semaphores):
    """
    aggiorna il pannello con la vista dell'incrocio

    i dati sono passati in 2 vettori da 4 elementi in ordine
    0 -> alto
    1 -> destra
    2 -> basso
    3 -> sinistra

    :param int[] autos: numeri di auto ad ogni strada
    :param srt[] semaphores: v-g-r, stato del semaforo
    """
    w=300
    h=300

    
    img = np.zeros((w,h,3), np.uint8)
    cv2.rectangle(img,(0,0),(w,h),(255,255,255),-1)

    verticalX = 150
    orizzontalY = 150
    streetSize=100
    stopThickness=5
    black=(0,0,0)
    #alto
    cv2.line(img,(int(verticalX-streetSize/2),0),(int(verticalX-streetSize/2),100),black,2)   #sx
    cv2.line(img,(int(verticalX+streetSize/2),0),(int(verticalX+streetSize/2),100),black,2)   #dx

    cv2.line(img,(verticalX,50),(verticalX,100),black,2)   #centro

    cv2.rectangle(img,(int(verticalX-streetSize/2),100-stopThickness),(verticalX,100+1),black,-1) #stop



    #destra
    cv2.line(img,(200,int(orizzontalY-streetSize/2)),(300,int(orizzontalY-streetSize/2)),black,2) #dx
    cv2.line(img,(200,int(orizzontalY+streetSize/2)),(300,int(orizzontalY+streetSize/2)),black,2) #sx
    
    cv2.line(img,(200,orizzontalY),(250,orizzontalY),black,2) #centro

    cv2.rectangle(img,(200-1,int(orizzontalY-streetSize/2)),(200+stopThickness,orizzontalY),black,-1) #stop



    #basso
    cv2.line(img,(int(verticalX-streetSize/2),200),(int(verticalX-streetSize/2),300),black,2)   #sx
    cv2.line(img,(int(verticalX+streetSize/2),200),(int(verticalX+streetSize/2),300),black,2)   #dx

    cv2.line(img,(verticalX,200),(verticalX,250),black,2)   #centro

    cv2.rectangle(img,(int(verticalX+streetSize/2),200+stopThickness),(verticalX,200-1),black,-1) #stop



    #sinistra
    cv2.line(img,(0,int(orizzontalY-streetSize/2)),(100,int(orizzontalY-streetSize/2)),black,2) #dx
    cv2.line(img,(0,int(orizzontalY+streetSize/2)),(100,int(orizzontalY+streetSize/2)),black,2) #sx
    
    cv2.line(img,(50,orizzontalY),(100,orizzontalY),black,2)
    
    cv2.rectangle(img,(100+1,orizzontalY),(100-stopThickness,int(orizzontalY+streetSize/2)),black,-1) #stop
   


    #numeri auto
    cv2.putText(img,str(autos[0]),(110, 75),cv2.FONT_HERSHEY_SIMPLEX,.8,black,1,cv2.LINE_AA)    #alto
    cv2.putText(img,str(autos[1]),(210, 135),cv2.FONT_HERSHEY_SIMPLEX,.8,black,1,cv2.LINE_AA)   #destra
    cv2.putText(img,str(autos[2]),(160, 230),cv2.FONT_HERSHEY_SIMPLEX,.8,black,1,cv2.LINE_AA)   #basso
    cv2.putText(img,str(autos[3]),(65, 180),cv2.FONT_HERSHEY_SIMPLEX,.8,black,1,cv2.LINE_AA)    #sinistra


    #semafori ( colori in BGR non RGB!)
    colors={'v':(0,255,0),'r':(0,0,255)}
    cv2.circle(img,(125, 115),10,colors[semaphores[0]],-1)  #alto
    cv2.circle(img,(185, 125),10,colors[semaphores[1]],-1)  #destra
    cv2.circle(img,(175, 185),10,colors[semaphores[2]],-1)  #basso
    cv2.circle(img,(115, 175),10,colors[semaphores[3]],-1)  #sinistra

    #cv2.line(img,(100,0),(100,100),(0,0,0),2)
    cv2.imshow('crossroads', img)



start_time=0
def ChronoInizio():
    global start_time
    start_time = time.perf_counter()

def ChronoFine():
    elapsed_time = time.perf_counter() - start_time
    print (elapsed_time*1000)



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
        




def  main():

    cv2_namedWindow('Video')       
    cv2_moveWindow('Video', 0,0)  

    cv2_namedWindow('crossroads')       
    cv2_moveWindow('crossroads', 700,0)  



    # Creazione del modello YOLO
    model=ModelCompatibilityLayerV5('YOLOv5/YOLOv5_repo','pts/yolov5/best_n.pt')
    #model=ModelCompatibilityLayerV5_TensorRT('YOLOv5/YOLOv5_repo','pts/yolov5/best_n.engine')       #TensorRT da testare su linux ( install su win di tensorrt è un dito nel culo)
    #model=ModelCompatibilityLayerV8('YOLOv8/PARAMETRO_NON_USATO','pts/yolov8/best_2023_02_22__23_47_13.pt')    
    

    #CPU
    #model=ModelCompatibilityLayerV5('YOLOv5/YOLOv5_repo','pts/yolov5/best_n.onnx')       #TensorRT da testare su linux ( install su win di tensorrt è un dito nel culo)
    #model=ModelCompatibilityLayerV5('YOLOv5/YOLOv5_repo','pts/yolov5/best_n_openvino_model/')

    
    


    # Apertura del flusso video

    #0 per la webcam, 1 per una seconda webcam o un file video
    caps=[
        cv2.VideoCapture("Resources/infer_data/traffic1.mp4"),
        cv2.VideoCapture("Resources/infer_data/traffic2.mp4"),
        cv2.VideoCapture("Resources/infer_data/traffic2.mp4"),
        cv2.VideoCapture("Resources/infer_data/traffic1.mp4") 
    ]



    #per salvare un video su disco
    """
    #apro il flusso
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out=cv2.VideoWriter('output.mp4', fourcc, 20.0,(640, 320))

    #salvo il frame nell'output
    out.write(cv2.resize(frame, (640, 320)))

    #chiudo il flusso 
    out.release()
    """

    #butto il primo frame per prendere le dimensioni di ogni singolo flusso ( DA RIVEDERE IL RELEASE )
    frame=getFrame(caps)
    original_height, original_width = frame.shape[:2]
    original_height=int(original_height/2)
    original_width =int(original_width/2)

    avrSample=4
    autoNumbersAvr=[ SMA_sequence(avrSample),SMA_sequence(avrSample),SMA_sequence(avrSample),SMA_sequence(avrSample)]
    while True:


       
        ChronoInizio()



        # Acquisizione di un frame dai video
        frame=getFrame(caps)
        if frame is None:
            break

       

        
        # analisi del frame tramite YOLO
        results = model(frame)


        

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

            

            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame,(xc,yc),2,(255, 0, 0),2)
            cv2.putText(frame, f'{model.names[classNumber]} {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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

        
        # visualizzazione dell'immagine con i risultati
        cv2.imshow('Video', frame)

        #visualizzo l'incrocio
        showCrossroads(autoNumbers,['v','r','v','r'])
        
        
        # interruzione con tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ChronoFine()    
        
        

        #premere "c" per continuare con il frame successivo
        #while not (cv2.waitKey(1) & 0xFF == ord('c')):
        #    pass


    # rilascio del flusso video e della finestra
    [cap.release() for cap in caps]
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()