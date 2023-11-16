import logging
import torch


#recupero il path del file corrente, prendo la cartella padre e la aggiunto al sys.path ( variabili di sistema del processo )
#cosi trova le librerie della cartella Libs
import os,sys
dir_path = os.path.dirname(os.path.realpath(__file__))+"\.."
sys.path.append(dir_path)

from _libs.point import Point



class ModelCompatibilityLayerV5_OnnxCPU:
    def __init__(self,folder,ptPath):
        #import onnxruntime
        #session = onnxruntime.InferenceSession(ptPath, providers=['CPUExecutionProvider'])
        #output_names = [x.name for x in session.get_outputs()]
        #meta = session.get_modelmeta().custom_metadata_map  # metadata


        self.model = torch.hub.load(folder,'custom', ptPath,source='local')
        #torch.hub.load('ultralytics/yolov5', 'custom', '/content/Smart-Traffic-Lights/yolov5/best_n.engine')  
        
    def __call__(self,frame):
        return resultsIterV5(self.model(frame))
    @property
    def names(self):
        return self.model.names

    def val(self):
        return self.val()
    
    def track(self,frame):
        raise "Track not implemented!"
    
class ModelCompatibilityLayerV5_TensorRT:
    def __init__(self,folder,ptPath):
        self.model = torch.hub.load(folder,'custom', ptPath,source='local')
        #torch.hub.load('ultralytics/yolov5', 'custom', '/content/Smart-Traffic-Lights/yolov5/best_n.engine')  
        
    def __call__(self,frame):
        return resultsIterV5(self.model(frame))
    @property
    def names(self):
        return self.model.names

    def val(self):
        return self.val()
    
    def track(self,frame):
        raise "Track not implemented!"

class ModelCompatibilityLayerV5:
    def __init__(self,folder,ptPath):
        self.model = torch.hub.load(folder,'custom', ptPath, source='local')
        
    def __call__(self,frame):
        return resultsIterV5(self.model(frame))
    @property
    def names(self):
        return self.model.names

    def val(self):
        return self.val()
    
    def track(self,frame):
        raise "Track not implemented!"
    
        
class ModelCompatibilityLayerV8:
    def __init__(self,folder,ptPath):
        from ultralytics import YOLO
        

        #toglie i log della v8
        logging.disable(logging.CRITICAL)

        #altro modo ( quello sopra toglie TUTTI i log, questo solo di ultralytics)
        #logger = logging.getLogger('ultralytics')
        #logger.disabled = True

        self.model = YOLO(ptPath)
        

        
    def __call__(self,frame):
        return resultsIterV8(self.model.predict(source=frame))
    
    def track(self,frame):
        return resultsIterV8(self.model.track(source=frame,persist=True))
        
    @property
    def names(self):
        return self.model.names




class resultsIterV5:
    def __init__(self, results):
            self.results = results.xyxy[0]
            self._results_size = len(self.results)
            self._current_index = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self._current_index < self._results_size:  
            results=self.results
            i=self._current_index
            res =  Result(int(results[i][0]),int(results[i][1]),int(results[i][2]),int(results[i][3]),float(results[i][4]),int(results[i][5]))
            self._current_index += 1
            return res
        raise StopIteration

class resultsIterV8:

    def _resetResulVariable(self):
        self.boxes = self.results[0].boxes.xyxy.cpu().numpy().astype(int)    
        self.probs = self.results[0].boxes.conf.cpu().numpy().astype(float)    
        self.cls = self.results[0].boxes.cls.cpu().numpy().astype(int)   
        self._boxes_size=len(self.boxes)

        self.ids = None
        if self.results[0].boxes.id!=None:
            self.ids= self.results[0].boxes.id.cpu().numpy().astype(int)      



    def __init__(self, results):
            self.results = results
            self._results_size = len(self.results)
            self._current_result=0


            if self._results_size != 0:
                self._resetResulVariable()
            self._current_boxes = 0


    def __iter__(self):
        return self
    def __next__(self):
        if self._current_result < self._results_size:  
            
            if self._current_boxes >= self._boxes_size:
                self._current_boxes=0
                self._current_result+= 1
                if( self._current_result < self._results_size):
                    self._resetResulVariable()
                else:
                    raise StopIteration
            
            i=self._current_boxes

            id=None
            if not self.ids is None:
                id= self.ids[i]


            res =  Result(self.boxes[i][0], self.boxes[i][1],   self.boxes[i][2],   self.boxes[i][3],   self.probs[i],  self.cls[i], id )
            self._current_boxes += 1
            return res
        raise StopIteration



class Result:
    def __init__(self,x1:int,y1:int,x2:int,y2:int,confidence:float,classNumber:int,id:int=None) -> None:
        self.start=Point(x1,y1)
        self.end=Point(x2,y2)
        self.confidence=confidence
        self.classNumber=classNumber
        self.id=id
