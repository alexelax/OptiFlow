
#aprire 4 flussi video
#ciascun frame unirlo in un unico flusso video
#analizzare il frame con YOLO
#visualizzare l'output con cv2


import cv2
import torch
import time

def resizeInWidth(frame,width):
    original_height, original_width = frame.shape[:2]
    # Set the desired width of the resized frame

    # Calculate the aspect ratio of the original frame
    aspect_ratio = original_height / original_width

    # Calculate the height of the resized frame
    desired_height = int(width * aspect_ratio)

    # Resize the frame
    return cv2.resize(frame, (width, desired_height))

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])





def  main():



    # Creazione del modello YOLOv5
    model = torch.hub.load('YOLOv5/YOLOv5_repo', 'custom', 'YOLOv5/_infer/best_n.pt', source='local')



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

    while True:
        # Acquisizione di un frame del video

        start_time = time.perf_counter()



        
        frames=[]
        endVideo=False
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                endVideo=True
                break
            frame=resizeInWidth(frame,320)      # Resize the frames to the same size     ( la funzione può essere ottimizzata in base al flusso video )
            frames.append(frame)
            

        if endVideo:
            break

        
        #collage frame
        frame = concat_tile([
            [frames[0],frames[1]],
            [frames[2],frames[3]]]
            )
        


        # analisi del frame tramite YOLOv5
        results = model(frame)
        
        
        # visualizzazione dei risultati sull'immagine
        for result in results.xyxy[0]:
            x1, y1, x2, y2 = int(result[0]),int(result[1]),int(result[2]),int(result[3])
            confidence = float(result[4])
            classNumber = int(result[5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'{model.names[classNumber]} {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # visualizzazione dell'immagine con i risultati
        cv2.imshow('Video', frame)

        
    
        # interruzione con tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print (elapsed_time*1000)


    # rilascio del flusso video e della finestra
    [cap.release() for cap in caps]
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()