import numpy as np

class Pde(object):
    def __init__(self,dim,range,equation,time_dependency = False):
        self.dimension = dim
        self.range = range
        self.time_dependency = time_dependency
        self.equation = equation

        self.IC = None
        self.BC = None

        '''
        dim: int
        IC: array
        BC: array
        range: array
        equation: lambda function with array input
        '''

    def set_initial_condition(self,initial_condition):
        self.IC = initial_condition


    def set_boundary_condition(self,boundary_condition):
        self.BC = boundary_condition

    def solution(self,points):
        if len(points) != self.dimension:
            Exception('dimension does not match')
        else:
            return self.equation(points)


    def boundary_condition(self):
        if self.BC is not None:
            return self.BC
        else:
            raise Exception('You should set boundary condition first')


    def initial_condition(self):
        if self.IC is not None:
            return self.IC
        else:
            raise Exception('You should set initial condition first')


    def solutions(self,points):
        if type(points[0]) is int or type(points[0]) is float:
            return self.solution(points)
        else:
            result = list()
            for i in points:
                result.append(self.solution(i))
            return np.array(result)
