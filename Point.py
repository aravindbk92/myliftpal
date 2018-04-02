import math

class point:
    def __init__(self,x=0,y=0):
        self.x = x
        self.y = y
    
    def distance(self,other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.hypot(dx,dy)
        
