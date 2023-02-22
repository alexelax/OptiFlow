import logging
import torch



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
        
class ModelCompatibilityLayerV8:
    def __init__(self,folder,ptPath):
        from ultralytics.yolo.engine.model import YOLO

        #toglie i log della v8
        logging.disable(logging.CRITICAL)

        #altro modo ( quello sopra toglie TUTTI i log, questo solo di ultralytics)
        #logger = logging.getLogger('ultralytics')
        #logger.disabled = True

        self.model = YOLO(ptPath,type="v8")

        
    def __call__(self,frame):
        return resultsIterV8(self.model.predict(source=frame))
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
            member =  int(results[i][0]),int(results[i][1]),int(results[i][2]),int(results[i][3]),float(results[i][4]),int(results[i][5])
            self._current_index += 1
            return member
        raise StopIteration

class resultsIterV8:

    def _resetResulVariable(self):
        self.boxes = self.results[0].boxes.xyxy.cpu().numpy()
        self.probs = self.results[0].boxes.conf.cpu().numpy()
        self.cls = self.results[0].boxes.cls.cpu().numpy()
        self._boxes_size=len(self.boxes)


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
            member =  int(self.boxes[i][0]),int(self.boxes[i][1]),int(self.boxes[i][2]),int(self.boxes[i][3]),float(self.probs[i]),int(self.cls[i])
            self._current_boxes += 1
            return member
        raise StopIteration

