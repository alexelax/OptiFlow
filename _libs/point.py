from __future__ import annotations
import math



class Point(object):
    '''Creates a point on a coordinate plane with values x and y.'''

    COUNT = 0

    def __init__(self, x=0, y=0):
        '''Defines x and y variables'''
        self.X = x
        self.Y = y
    def clone(self):
        return Point(self.X,self.Y)
    
    def move(self, dx, dy):
        '''Determines where x and y move'''
        self.X = self.X + dx
        self.Y = self.Y + dy
    
    def moveNew(self, dx, dy):
        '''Determines where x and y move and return a new point'''
        p = self.clone()
        p.move( dx, dy)
        return p

    def __str__(self):
        return "Point(%s,%s)"%(self.X, self.Y) 


    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def distance(self, other):
        dx = self.X - other.X
        dy = self.Y - other.Y
        return math.sqrt(dx**2 + dy**2)


    def distance(self, p):
        dx = self.X - p.X
        dy = self.Y - p.Y
        return math.hypot(dx, dy)
    
    def unpack(self):
         '''return a tuple (x,y)'''
         return (self.X,self.Y)

    def middle(self,other:Point):
        '''return the middle point from self and another point'''
        return Point(int((self.X+other.X)/2),int((self.Y+other.Y)/2))