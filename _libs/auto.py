#recupero il path del file corrente, prendo la cartella padre e la aggiunto al sys.path ( variabili di sistema del processo )
#cosi trova le librerie della cartella Libs
import os,sys
dir_path = os.path.dirname(os.path.realpath(__file__))+"\.."
sys.path.append(dir_path)

from _libs.point import Point

class auto:
    '''classe che rapprenseta un auto, permette di tenere traccia della sua posizione ed ottenere il tracciato'''

    def __init__(self,id) -> None:
        self.positions=[]
        self.id=id

    def moved(self,x,y):
        self.positions.append(Point(x,y))       #TODO: li memorizzo già un tupla/array per recuperarli più velocemnente?

    def getTrack(self):
        return [[p.X,p.Y] for p in self.positions]
    
    def getTrackLen(self):
        return len(self.positions)


class autoCollection:
    
    def __init__(self) -> None:
        self.collections=dict()


    def getOrCreate(self,id):
        if id in self.collections:
            return self.collections[id]
        
        tmp = auto(id)
        self.collections[id]=tmp
        return tmp

    