# -*- coding: utf-8 -*-
"""
@author: Varadarajan Srinivasan

Solves and animates the (2+1)-dimensional discretized wave equation for user-inputted parameters.
Can set edges to be treated cyclically (periodic boundary condition) or as zeros.

Array slicing is used to quickly compute the grids (excluding the boundary in the edgeType==1 case of periodic boundary conditions).
"""
import numpy as np
import pylab as pl
import matplotlib as mat
import matplotlib.animation as anim

"""
--------------------------------------------------------------ADJUSTABLE PARAMETERS--------------------------------------------------------------
"""
edgeType=0 #0 for zeros along edges. 1 for cyclical edges (periodic boundary condition).

tstepcnt=200 #tstepcnt time-slices are indexed from t=0 to t=tstepcnt-1
rowcnt=100
colcnt=100

t0funcxy = 'np.exp(-((x-colcnt/3.0)**2/(colcnt/20.0)+(y-rowcnt/2.0)**2/(rowcnt/2.0)))+np.exp(-((y-colcnt/3.0)**2/(colcnt/20.0)+(x-rowcnt/2.0)**2/(rowcnt/2.0)))' #(string) function of x and y that initializes the t=0 slice
t1funcxy = 'np.exp(-(((x-K**0.5)-colcnt/3.0)**2/(colcnt/20.0)+(y-rowcnt/2.0)**2/(rowcnt/2.0)))+np.exp(-(((y-K**0.5)-colcnt/3.0)**2/(colcnt/20.0)+(x-rowcnt/2.0)**2/(rowcnt/2.0)))' #(string) function of x and y that initializes the t=1 slice
K=0.25 #must be well below 1
#K=(waveSpeed*dt/dx)**2
"""
-------------------------------------------------------------------------------------------------------------------------------------------------

When edgeType is set to 0, the points along the edge of the grid are killed to 0, and then the discretized wave equation is applied for 
all later times only to the points on the interior of the grid. So, we discard the data along the boundary.
When edgeType is set to 1, the boundary values are discarded and replaced with the corresponding values computed using the discretized 
wave equation and applying the opposite edges' values cyclically. This uses for loops instead of array splicing and so runs much more 
slowly.

K is a positive unitless constant related to the speed of the wave and the discretization step sizes in space and time. For this numerical
method to successfully approximate a solution to the wave equation, K must be less than 1. The accompanying documentation file explains 
what K means, how it arises from discretizing the wave equation, and why it must be less than 1. The lower K is, the more "frames" are 
computed for a given wave period. The way this program is built, the time-resolution gained by lowering K does not change the runtime. 
Instead the drawback is just how far into the future we can see the wave because the number of time-slices computed is set. A higher K 
means the delta t between time-slices is higher (k is prop. to (delta t)^2) so, with the same tstepcnt time-slices, we can model the wave 
further into the future, but with a lower accuracy. Unlike evaluating an analytical solution at various times, the error from a high K 
will grow with each time-slice because this numerical solution iteratively uses the previous 2 slices' values. Recommended K is 0.1 to 0.4.
"""

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

def initialSlices(rows, cols, ts, t0, t1):
    """
    Sets initial conditions.
    Initializes a 2+1 dimensional array with all zeros and initializes the t=0 and t=1 slices as the functions of x and y.
    The returned array effectively is 1 rectangle of 0s surrounding a rows x cols grid evaluated for x and y where the origin (0,0) is taken at 
    the top left of the interior nonzero array. That is, the origin is at row index 1, col index 1.
    """
    f = np.zeros((rows+2,cols+2,ts))
    if t0==t1: #both cases do the same thing, but if t0==t1, this code runs significantly faster
        for row in range(1,rows+1):
            for col in range(1,cols+1):
                f[row,col,0]=evalFuncOfxyAtVal(t0,row-1,col-1)
        f[:,:,1]=f[:,:,0]
    
    else:
        for row in range(1,rows+1):
            for col in range(1,cols+1):
                f[row,col,0]=evalFuncOfxyAtVal(t0,row-1,col-1)
                f[row,col,1]=evalFuncOfxyAtVal(t1,row-1,col-1)
    
    return f

print '\nWave equation computations of the points along the edge of the grid can be changed between taking the would-be points outside the grid as 0 (edgeType=0), or cyclically taking the corresponding boundary points on the opposite edge (edgeType=1).'

print '\nThese fully adjustable parameters are currently set as:'
print ' edgeType =',edgeType
print ' Number of rows = rowcnt = %d\n Number of columns = colcnt = %d\n Number of time slices = tstepcnt = %d' %(rowcnt,colcnt,tstepcnt)
print ' waveSpeed^2 (dt/dx)^2 = K =',K,'   [!Read docstrings and accompanying documentation before changing K!]'
print '\nFunction of x and y that initializes the t=0 timeslice:',t0funcxy
print 'Function of x and y that initializes the t=1 timeslice:',t1funcxy,'\n\n'

print 'Initializing...'
u=initialSlices(rowcnt,colcnt,tstepcnt, t0funcxy, t1funcxy)

print 'Computing...'

"""
The following discretized equations were derived from the wave equation, as explained in the accompanying documentation.
"""
if edgeType==0:
    for t in range(2,tstepcnt):
        u[1:-1,1:-1,t]=K*(u[:-2,1:-1,t-1]+u[2:,1:-1,t-1]+u[1:-1,:-2,t-1]+u[1:-1,2:,t-1]-4*u[1:-1,1:-1,t-1])+2*u[1:-1,1:-1,t-1]-u[1:-1,1:-1,t-2]
elif edgeType==1:
    for t in range(2,tstepcnt):
        u[1:-1,1:-1,t]=K*(u[:-2,1:-1,t-1]+u[2:,1:-1,t-1]+u[1:-1,:-2,t-1]+u[1:-1,2:,t-1]-4*u[1:-1,1:-1,t-1])+2*u[1:-1,1:-1,t-1]-u[1:-1,1:-1,t-2]
        for m in range(0,rowcnt): #so m represents the y direction (row index number)
            for n in range(0,colcnt): #so n represents the x direction (column index number)
                if m==0 and n!=0 and n!=colcnt-1: #0th row excl. corners
                    u[m][n][t]=K*(u[m][n+1][t-1]+u[m][n-1][t-1]+u[m+1][n][t-1]+ u[m-1][n][t-1] -4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
                elif m==rowcnt-1 and n!=0 and n!=colcnt-1: #last row ex.corners
                    u[m][n][t]=K*(u[m][n+1][t-1]+u[m][n-1][t-1]+ u[0][n][t-1] +u[m-1][n][t-1]-4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
                elif n==0 and m!=0 and m!=rowcnt-1: #0th column ex.corn.
                    u[m][n][t]=K*(u[m][n+1][t-1]+ u[m][n-1][t-1] +u[m+1][n][t-1]+u[m-1][n][t-1]-4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
                elif n==colcnt-1 and m!=0 and m!=rowcnt-1: #last column ex.c.
                    u[m][n][t]=K*( u[m][0][t-1] +u[m][n-1][t-1]+u[m+1][n][t-1]+u[m-1][n][t-1]-4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
            
                #corners:
                elif m==0 and n==0: #top left corner
                    u[m][n][t]=K*(u[m][n+1][t-1]+ u[m][n-1][t-1] +u[m+1][n][t-1]+ u[m-1][n][t-1] -4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
                elif m==0 and n==colcnt-1: #top right corner
                    u[m][n][t]=K*( u[m][0][t-1] +u[m][n-1][t-1]+u[m+1][n][t-1]+ u[m-1][n][t-1] -4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
                elif m==rowcnt-1 and n==0: #bottom left corner
                    u[m][n][t]=K*(u[m][n+1][t-1]+ u[m][n-1][t-1] + u[0][n][t-1] +u[m-1][n][t-1]-4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
                elif m==rowcnt-1 and n==colcnt-1: #bottom right corner
                    u[m][n][t]=K*( u[m][0][t-1] +u[m][n-1][t-1]+ u[0][n][t-1] +u[m-1][n][t-1]-4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]

#------------------ Numerical computations done. Solutions at all times stored in u. ------------------
print 'The computed animation has a spatial resolution of %d columns x %d rows and shows %d frames of time' % (colcnt, rowcnt, tstepcnt)

#Animating as a heat map showing each time-slice:
fig = mat.pyplot.figure(figsize=(6, 3))

ax = fig.add_subplot(1,1,1)
ax.set_title('colorMap')
ax.set_aspect('equal')

def frame(n):
    ax.clear()
    ax.imshow(u[1:-1,1:-1,n]) #doesn't show the outer 0 edges that were used as the outer boundary condition in the array slicing computations
    ax.set_aspect('equal')
    
dummy = anim.FuncAnimation(fig,frame,range(tstepcnt),interval=25)
pl.show()