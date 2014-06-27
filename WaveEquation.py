# -*- coding: utf-8 -*-
"""
@author: Varadarajan Srinivasan

This program discretizes and solves the Wave Equation in 3 (2 spatial + 1 temporal) dimensions for user-inputted grid size and absolutely
any user-inputted boundary condition as individual values or as functions.
"""
import numpy as np
import pylab as pl
import matplotlib as mat
import matplotlib.animation as m

"""
Evaluates a mathematical function of 2 variables written as a string in Python syntax in terms of x and y at a specified point (xval,yval). 
f must be in terms of x and y because it is tedious to allow any variable name. e.g. It is cumbersome to let python know which n
is the variable in the string 'sin(n)'
"""
def evalFuncOfxyAtVal(TwoVarFunc,rowval,colval): ##f MUST BE IN TERMS OF x AND y
    y=rowval ##ignore the not-used error your IDE might give. The x and y will come from user input
    x=colval ##ignore error    
    """
    Note that there is no transposition being done. Rather, traditionally we order (x,y) for (val along horizontal, val along vertical),
    whereas we traditionally order (row number, column number) which corresponds to (val along vertical, val along horizontal).
    """
    return eval(TwoVarFunc)

"""
Initializes the 2+1 dimensional array with all zeros and initializes the t=0 boundary 
using point-by-point value entry or a user-inputted function of x and y
This is then copied to the t=1 slice as well because 2 initial slices are needed in the discretized wave equation
and it makes more sense to use the same value instead of having the slice before the boundary to be all 0s because
that would pollute the discretized representation of the second order PDE with an artificially high rate of change.
(more details in the accompanying documentation WaveEquation.pdf)
"""
def t0and1(rows, cols, tsteps):
    f = np.zeros((rows,cols,tsteps))

    opt = raw_input('Press \'m\' to manually input each boundary point on the t=0 grid or \'f\' to input the t=0 values as a function of x and y. ')

    if opt == 'f':
        t0funcxy = raw_input('Define a function of x and y that sets the t=0 grid (in Python syntax): ')
        for row in range(0,rows):
            for col in range(0,cols):
                f[row][col][0]=evalFuncOfxyAtVal(t0funcxy,row,col)
                f[row][col][1]=f[row][col][0]
    
    elif opt == 'm':
        for row in range(0,rows):
            for col in range(0,cols):
                f[row][col][0]=input('u(%d,%d) = ' % (row, col))
                f[row][col][1]=f[row][col][0]
                
    return f

"""
Updates a given time-slice of the 2+1d array based on the previously existing values. So, this only works if when a time-slice is called 
to be updated, the previous 2 time slices have already been computed and stored in the array.

A detailed explanation of how this discretized equation was derived is in the accompanying documentation file.
"""
def solveNextTimeSlice(f,t,rows,cols):
    for n in range(1,rows-1): #so n represents the y direction (rows)
        for m in range(1,cols-1): #so m represents the x direction (columns)
            f[m][n][t+1]=k*(f[m][n+1][t]+f[m][n-1][t]+f[m+1][n][t]+f[m-1][n][t]-4*f[m][n][t])+2*f[m][n][t]-f[m][n][t-1]
    return f

#just a function to print slices used for presentation, not actually computationally relevant to this program's purpose.
def giveTimeSlice(f,t,rows,cols):
    fslice=np.zeros((rows,cols))
    for i in range(0,rows):
        for j in range(0,cols):
            fslice[i][j]=f[i][j][t]
    return fslice


tstepcnt=150
rowcnt=50
colcnt=50
k=0.25
"""
k is a positive unitless constant related to the speed of the wave and the discretization step sizes in space and time. For this numerical
method to successfully approximate a solution to the wave equation, k must be less than 1. The accompanying documentation file explains 
what k means, how it arises from discretizing the wave equation, and why it must be less than 1.
"""

u = t0and1(rowcnt,colcnt,tstepcnt)
for time in range(2,tstepcnt):##no tstepcnt-1 required because solutions are done on time-1
    u=solveNextTimeSlice(u,time-1,rowcnt,colcnt)#starting at t=time-1 means that solving the next t slice computes slices starting at t=time
"""
------------------ numerical computations done, solutions at all times stored in u ------------------
"""
print u
for time in range(0,tstepcnt):
    print 't = ',time
    print u[:,:,time]    
    
    #print giveTimeSlice(u,time,rowcnt,colcnt)

#now graphically

fig = mat.pyplot.figure(figsize=(6, 3))

ax = fig.add_subplot(1,1,1)
ax.set_title('colorMap')
#mat.pyplot.imshow(ut)
ax.set_aspect('equal')

mat.pyplot.show()

t = np.linspace(0,tstepcnt-1,tstepcnt)
dt = 0.02

def frame(n):
    ax.clear()
    ax.imshow(u[:,:,n])
    ax.set_aspect('equal')
    
dummy = m.FuncAnimation(fig,frame,range(tstepcnt),interval=25)
pl.show()