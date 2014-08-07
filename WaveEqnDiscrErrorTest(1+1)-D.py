# -*- coding: utf-8 -*-
"""
@author: Varadarajan Srinivasan

This is just a test program to see how problematic the inherrent discretization error is when trying to model pulses with the wave equation.
It solves and animates the (1+1)-dimensional discretized wave equation.
"""
import numpy as np
import pylab as pl
import matplotlib as mat
import matplotlib.animation as anim

def evalFuncOfxAtVal(func, val): #string func MUST BE IN TERMS OF x
    x=val #Ignore the "not used" error your IDE might show for this line. It is used as the user inputs in terms of x.
    return eval(func)

def initialSlices(cols, ts, t0, t1):
    """
    Sets initial conditions.
    Initializes a 1+1 dimensional array with all zeros and initializes the t=0 and t=1 slices as the functions of x
    """
    f = np.zeros((cols,ts))
    if t0==t1: #both cases do the same thing, but if t0==t1, this code runs significantly faster
        for col in range(0,cols):
            f[col,0]=evalFuncOfxAtVal(t0,col)
        f[:,1]=f[:,0]
    
    else:
        for col in range(0,cols):
            f[col,0]=evalFuncOfxAtVal(t0,col)
            f[col,1]=evalFuncOfxAtVal(t1,col)
    
    return f

"""
--------------------------------------------------------------ADJUSTABLE PARAMETERS--------------------------------------------------------------

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
edgeType=0 #0 for zeros along edges. 1 for cyclical edges (periodic boundary condition).

tstepcnt=100 #tstepcnt time-slices are indexed from t=0 to t=tstepcnt-1
colcnt=100

t0funcx = 'np.exp(-((x-colcnt/3.0)**2/(colcnt/2.0)))' #(string) function of x that initializes the t=0 slice
t1funcx = 'np.exp(-(((x-K**0.5)-colcnt/3.0)**2/(colcnt/2.0)))' #(string) function of x that initializes the t=1 slice
K=0.25 #must be well below 1
#K=(waveSpeed*dt/dx)**2
"""
-------------------------------------------------------------------------------------------------------------------------------------------------
"""

print '\nWave equation computations of the points along the edge of the grid can be changed between taking the would-be points outside the grid as 0 (edgeType=0), or cyclically taking the corresponding boundary points on the opposite edge (edgeType=1).'

print '\nThese fully adjustable parameters are currently set as:'
print ' edgeType =',edgeType
print ' Number of columns = colcnt = %d\n Number of time slices = tstepcnt = %d' %(colcnt,tstepcnt)
print ' waveSpeed^2 (dt/dx)^2 = K =',K,'   [!Read docstrings and accompanying documentation before changing K!]'
print '\nFunction of x and y that initializes the t=0 timeslice:',t0funcx
print 'Function of x and y that initializes the t=1 timeslice:',t1funcx,'\n\n'

print 'Initializing...'
u=initialSlices(colcnt,tstepcnt, t0funcx, t1funcx)

print 'Computing...'

"""
The following discretized equations were derived from the wave equation, as explained in the accompanying documentation.
"""
for t in range(2,tstepcnt):
    u[1:-1,t]=K*(u[:-2,t-1]+u[2:,t-1]-2*u[1:-1,t-1])+2*u[1:-1,t-1]-u[1:-1,t-2]

"""
------------------ numerical computations done, solutions at all times stored in u ------------------
"""
print 'The computed animation has a spatial resolution of %d columns and shows %d frames of time' % (colcnt, tstepcnt)

#Animating as a heat map showing each time-slice:
fig = mat.pyplot.figure(figsize=(6, 3))

ax = fig.add_subplot(1,1,1)
ax.set_title('colorMap')
ax.set_aspect('equal')

def frame(n):
    ax.clear()
    ax.imshow(u[:,n])
    ax.set_aspect('equal')
    
dummy = anim.FuncAnimation(fig,frame,range(tstepcnt),interval=25)
pl.show()

#run the following loop to print the time-slices chronologically as 2d arrays:
"""
for time in range(0,tstepcnt):
    print 't = ',time
    print u[:,:,time]
"""