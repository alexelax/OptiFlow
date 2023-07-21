import sys
import threading
import time


class Singleton(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(Singleton, cls).__new__(cls)
      cls.lock = threading.Lock()

    return cls.instance
  
    
  def print(self,*args):
    with self.lock:
        print(*args)
        sys.stdout.flush()
   
singleton = Singleton()
