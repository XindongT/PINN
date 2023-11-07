import numpy as np

class Pde:
    def __init__(self,dim,range,solution):
        self.dimension = dim
        self.range = range
        self.equation = solution
        self.IC = None
        self.BC = None

        '''
        dim: int
        IC: array
        BC: array
        range: array
        '''

    def set_initial_condition(self,initial_condition):
        self.IC = initial_condition

    def set_boundary_condition(self,boundary_condition):
        self.BC = boundary_condition

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

    def solution(self,points):
        if len(points) != self.dimension:
            Exception('dimension does not match')
        else:
            return self.equation(points)


    def solutions(self,points):
        if type(points[0]) is int or type(points[0]) is float:
            return self.solution(points)
        else:
            result = list()
            for i in points:
                result.append(self.solution(i))
            return np.array(result)

class transport_eq(Pde):
    def __init__(self,c,RHS):
        self.c = c
        self.auxiliary_condition = RHS


    def solution(self,points):
        if len(points) != self.dimension:
            Exception('dimension does not match')
        else:
            return self.equation(points)


    def solutions(self,points):
        if type(points[0]) is int or type(points[0]) is float:
            return self.solution(points)
        else:
            result = list()
            for i in points:
                result.append(self.solution(i))
            return np.array(result)
