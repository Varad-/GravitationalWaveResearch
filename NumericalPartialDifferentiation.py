# -*- coding: utf-8 -*-
"""
@author: Varadarajan Srinivasan

Numerical 2D partial differentiation using finite differences (averaging the difference on both sides of each point).
"""

import numpy as np

def delnDxSlice(tslice):
    """
    Takes a time slice and returns an evaluated 2d grid of the grid spacing (delta x) times the numerical derivative of f[row][col][time] 
    with respect to x (i.e. column index). Derivatives are only evaluated on the interior of the grid. The edges are left as 0.
    """
    nDx=np.zeros(np.shape(tslice))    
    nDx[1:-1,1:-1]=0.5*(tslice[1:-1,2:]-tslice[1:-1,:-2])
    return nDx
    

def delnDySlice(tslice):
    """
    Takes a time slice and returns an evaluated 2d grid of the grid spacing (delta y) times the numerical derivative of f[row][col][time] 
    with respect to y (i.e. row index). Note that in the matrix this means the y coordinate points downwards. Derivatives are only evaluated 
    on the interior of the grid. The edges are left as 0.
    """
    nDy=np.zeros(np.shape(tslice))    
    nDy[1:-1,1:-1]=0.5*(tslice[2:,1:-1]-tslice[:-2,1:-1])
    return nDy
    
def evalFuncOfxyAtVal(TwoVarFunc,rowval,colval): ##f MUST BE IN TERMS OF x,y
    """
    Evaluates a mathematical function of 2 variables written as a string in Python syntax in terms of x and y at a specified point (colval,rowval). 
    f must be in terms of x and y because it is tedious to allow any variable name. e.g. It is cumbersome to let python know which n
    is the variable in the string 'sin(n)'
    """
    y=rowval ##ignore the not-used error your IDE might give. The x and y will come from user input
    x=colval ##ignore error    
    """
    Note that there is no transposition being done. Rather, traditionally we order (x,y) for (val along horizontal, val along vertical),
    whereas we traditionally order (row number, column number) which corresponds to (val along vertical, val along horizontal).
    """
    return eval(TwoVarFunc)

rows = 7
cols = 8

f = np.zeros((rows,cols))
t0funcxy = raw_input('\nDefine a function of x and y that sets the t=0 grid (in Python syntax). f(x,y)=')
for row in range(0,rows):
    for col in range(0,cols):
        f[row,col]=evalFuncOfxyAtVal(t0funcxy,row,col)

print '\nf=\n',f
print 'Numerical approximation: \ndelta x * df/dx =\n', delnDxSlice(f)
print 'Numerical approximation: \ndelta y * df/dy =\n', delnDySlice(f)
print '\nThe numerical derivatives are not attempted along the edges (nominally left as 0).'