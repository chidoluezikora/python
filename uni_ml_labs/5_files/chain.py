import numpy as np

class Markov():
    def __init__(self, v0, M, colors = []):
        self.__i = 0
        self.__v = [v0]
        self.__M = M
        self.colors = colors
               
    def __str__(self):
        return "Timestep: {}\nState: {}".format(self.__i, self.__v[-1])
    
    def evolve(self):
        self.__v.append( np.dot(self.__M, self.__v[-1]) )
        self.__i += 1