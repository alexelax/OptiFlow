

class SMA_sequence:


    def __init__(self, lenght):
        self.lenght = lenght
        self.values=[]
        self.current=None
       
    def AddValue(self,value):
        self.values.append(value)
        if len(self.values)>self.lenght:
            self.values.pop(0)
        self.current = sum(self.values) / len(self.values)    
        pass

    def isValid(self):
        if len(self.values)>=self.lenght:
            return True
        return False
      
 




