
import array
from contextlib import closing
import multiprocessing
from multiprocessing.managers import BaseManager
import socket
import collections.abc
import threading

class PortManager:

    def __init__(self,lock=None) -> None:
        self.Ports={}
        self.lock=lock
        if self.lock == None :
            self.lock=threading.Lock()

    def setLock(self,lock):
        self.lock=lock


    def addPort(self,port:int | collections.abc.Sequence):

        if isinstance(port,collections.abc.Sequence):
            for p in port:
                self.addPort(p)
            return

        with self.lock:
            if self.__getPort(port) ==None:
                self.Ports[port]={"port":port,"free":True}

   

    def __getPort(self,port):
        if port in self.Ports:
            return self.Ports[port]
        return None
        
    def lockPort(self):
        with self.lock:
            for port in self.Ports:
                if self.Ports[port]["free"]: #se è libera ( True ) 
                    self.Ports[port]["free"]=False
                    return port
            return None
        
      
    def releasePort(self,port):
        with self.lock:
            p=self.__getPort(port)
            if p == None: return
            p["free"]=True


    def getFreeCount(self):
        with self.lock:
            i=0
            for port in self.Ports:
                if self.Ports[port]["free"]:
                    i+=1
            return i
            
