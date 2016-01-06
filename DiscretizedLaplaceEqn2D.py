# -*- coding: utf-8 -*-
#@author: Varadarajan Srinivasan
#
#This program discretizes and solves Laplace's Equation in 2 dimensions for user-inputted grid size and any boundary 
#conditions (given as functions or as individual values). Each i-iteration updates progressively smaller rectangles 
#starting from the boundary. Note that this program is designed to allow each rectangle to update as a whole instead 
#of simply updating point after point along each rectangle. Then, that entire iterative process is itself iterated 
#until the updates are below the adjustable tolerance level.

import numpy as np
import math as m
import matplotlib as mpl
import pylab as pl

#Very low tolerance produces very precise solution values, but slows the runtime dramatically. For the suggested tolerance
#of 0.01, grids larger than about 50x50 will take more than several seconds, but by that size the contours become virtually 
#perfectly smooth for visual purposes.
tolerance=0.01

def evalFuncOfxAtVal(func, val):
    """Evaluates a mathematical function written as a string in Python syntax in terms of x at a specified value. It is
    tedious to allow any variable name in cases such as sin(n), making it cumbersome to let the program know which instances
    of the letter n is the variable and which is part of an in-built function.
    
    The string func MUST BE IN TERMS OF x!
    """
    x=val #Ignore the "not used" error your IDE might show for this line. It is used as the user inputs in terms of x.
    return eval(func)

def initialize(rowcount, colcount, funcname):
    """creates a rowcount x colcount grid and returns it with user-inputted boundary conditions and zeros in the interior"""
    f = np.zeros((rowcount,colcount))
    print 'Make sure all values are much larger than this program\'s iteration-tolerance which is 10^%d (and is easily adjustable in the code)' % m.log10(tolerance)
    opt = None
    while(opt!='m' and opt!='f'):
        opt = raw_input('Press \'m\' to manually input each boundary point or \'f\' to input functions in Python syntax for boundary conditions. ')
        if opt == 'f':
            print '0th (bottom in graph, top in matrix) row as a function of x, representing the position along the row (column index number):'
            topRow=raw_input('%s(0,x) = ' % funcname)
            for col in range(0,colcount):
                f[0][col]=evalFuncOfxAtVal(topRow,col)
            
            print 'Opposite extreme row as a function of x, representing the position along the row (column index number):'
            bottomRow=raw_input('%s(%d,x) = ' % (funcname,rowcount-1))            
            for col in range(0,colcount):
                f[rowcount-1][col]=evalFuncOfxAtVal(bottomRow, col)
            
            print 'Leftmost (0th) column excluding previously inputted top and bottom entries as a function of x, representing the position along the column (row index number):'
            leftCol=raw_input('%s(x,0) = ' % funcname)
            for row in range(1,rowcount-1):
                f[row][0]=evalFuncOfxAtVal(leftCol, row)
            
            print 'Rightmost column excluding previously inputted top and bottom entries as a function of x, representing the position along the column (row index number):'
            rightCol=raw_input('%s(x,%d) = ' % (funcname,colcount-1))            
            for row in range(1,rowcount-1):
                f[row][colcount-1]=evalFuncOfxAtVal(rightCol, row)
            
        elif opt == 'm':
            print '0th (bottom in graph, top in matrix) row:'
            for col in range(0,colcount):
                f[0][col]=input('%s(0,%d) = ' % (funcname, col))
            
            print 'Opposite extreme row:'
            for col in range(0,colcount):
                f[rowcount-1][col]=input('%s(%d,%d) = ' % (funcname, rowcount-1, col))
            
            print 'Leftmost (0th) column excluding previously-inputted top and bottom entries:'
            for row in range(1,rowcount-1):
                f[row][0]=input('%s(%d,0) = ' % (funcname, row))

            print 'Rightmost column excluding previously-inputted top and bottom entries:'
            for row in range(1,rowcount-1):
                f[row][colcount-1]=input('%s(%d,%d) = ' % (funcname, row, colcount-1))
    return f

def Laplpt(func, r, c):
    """See accompanying documentation (DiscretizedLaplaceEqn.pdf) for derivation and details: the Laplace Equation, when discretized 
    implies that each point takes the average of the four neighboring point's values.
    """
    return 0.25*(func[r+1][c]+func[r-1][c]+func[r][c+1]+func[r][c-1])

def Laplace(oldphi, rows, columns):
    """solves the equation in Laplpt() once in decreasing rectangles"""
    newphi = np.zeros((rows, columns))
    newphi[:][:] = oldphi[:][:]
    i=1 #first iteration solves along the largest rectangle that fits in the grid without including the boundary points
    while i<rows/2.0 and i<columns/2.0:
        for col in range(i,columns-i): #horizontal sides
            newphi[i][col]=Laplpt(oldphi, i, col)
            newphi[rows-1-i][col]=Laplpt(oldphi, rows-1-i, col)
        for row in range(1+i,rows-1-i): #vertical sides excluding the corners that were already set
            if 1+i!=rows-1-i:
                newphi[row][i]=Laplpt(oldphi, row, i)
                newphi[row][columns-1-i]=Laplpt(oldphi, row, columns-1-i)
        oldphi[:][:]=newphi[:][:]
        i+=1
    return newphi

print '\nUse this program to solve the 2D Laplace Equation numerically for absolutely any boundary conditions.'

Rmax=input('Choose number of grid rows (25-50 for good results with short runtime): ')
Cmax=input('Choose number of grid columns (25-50 for good results with short runtime): ')

phi = initialize(Rmax, Cmax, 'phi') #0th i-iteration creates zero grid and initializes boundary conditions

phi = Laplace(phi, Rmax, Cmax) #first solution-iteration

absChange=np.zeros((Rmax,Cmax))
previousPhi=np.zeros((Rmax,Cmax))
absChange=np.absolute(phi) #initialized as the first iteration so the following loop runs as long as the inits are above tolerance

#We have a system of equations for all the points on the interior of the grid. See the accompanying documentation for a description 
#of the process. Essentially, we iterate the already iterative Laplace() until we are arbitrarily close to the solution.
while np.amax(absChange) > tolerance:
    previousPhi[:][:]=phi[:][:]
    phi=Laplace(phi, Rmax, Cmax) #2nd iteration when the loop runs for the first time
    absChange=np.absolute(np.subtract(phi,previousPhi))


#------------------------------------------------ End of Laplacian computations ------------------------------------------------
#Now we have the final solution stored in the 2d array phi. We can show this graphically:

contourcount = 10 #change this to plot with a different number of contours

x = np.linspace(0,Cmax-1,Cmax)
y = np.linspace(0,Rmax-1,Rmax)

fig = pl.figure()
panel = fig.add_subplot(1,1,1)
levs = np.linspace(np.amin(phi),np.amax(phi),contourcount)
pl.contourf(x, y, phi, levs, cmap=mpl.cm.gist_rainbow)

print '\nNumerical solution to Laplace Equation with the given boundary conditions and grid size to within an iteration-tolerance of 10^%d given as the following matrix and as a graphic pop-up with %d contours:' % (m.log10(tolerance), contourcount)

print phi #solution as a matrix
pl.show() #solution as a contour graph
