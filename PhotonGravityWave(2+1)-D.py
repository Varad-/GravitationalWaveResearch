# -*- coding: utf-8 -*-
"""
@author: Varadarajan Srinivasan

Produces a (2+1)D animation showing an electromagnetic potential being influenced by a passing gravitational wave after taking (3+1)D considerations 
into account such that results can be shown for any arbitrarily inclined angle at which the gravitational wave comes in. This represents a wire-like 
potential, but the code can easily be extended to full (3+1)D by including all components of the equations provided in the documentation.

Runtime has been enormously improved by array slicing over 4D arrays (treated as 2D matrices of 2D grids) instead of looping over 2D arrays.
The 2D matrices are to do with the 3x3 spatial slice of the metric tensor. 
The grids represent each point of our discretized plane.
"""

import numpy as np
import pylab as pl
import matplotlib as mat
import matplotlib.animation as anim

"""
--------------------------------------------------------------ADJUSTABLE PARAMETERS--------------------------------------------------------------
"""
tstepcnt=501 #tstepcnt time-slices are indexed from t=0 to t=tstepcnt-1. Must be >=2 for program to run (2 slices needed to initialize u)
rowcnt=150
colcnt=150

t0funcxy = 'np.exp(-((x-colcnt/3.0)**2/(colcnt/20.0)+(y-rowcnt/2.0)**2/(rowcnt/2.0)))+np.exp(-((y-colcnt/3.0)**2/(colcnt/20.0)+(x-rowcnt/2.0)**2/(rowcnt/2.0)))' #(string) function of x and y that initializes the t=0 slice
t1funcxy = 'np.exp(-(((x+K**0.5)-colcnt/3.0)**2/(colcnt/20.0)+(y-rowcnt/2.0)**2/(rowcnt/2.0)))+np.exp(-(((y+K**0.5)-colcnt/3.0)**2/(colcnt/20.0)+(x-rowcnt/2.0)**2/(rowcnt/2.0)))' #(string) function of x and y that initializes the t=1 slice

theta = np.pi/2 #angle of incidence of grav wave
eps=0.5 #meaning of epsilon is in the documentation
kgrav=0.02 #this is the k_grav from cos(kz-kt)
c=1 #c=1 for units of the wave speed

K=0.3 #K=(c*dt/dx)**2
"""
-------------------------------------------------------------------------------------------------------------------------------------------------

K is a positive unitless constant related to the speed of the wave and the discretization step sizes in space and time. For this numerical
method to successfully approximate a solution to the wave equation, K must be less than 1. The accompanying documentation file explains 
what K means, how it arises from discretizing the wave equation, and why it must be less than 1. The lower K is, the more "frames" are 
computed for a given wave period. The way this program is built, the time-resolution gained by lowering k does not change the runtime. 
Instead the drawback is just how far into the future we can see the wave because the number of time-slices computed is set. A higher K 
means the delta t between time-slices is higher (K is prop. to (delta t)^2) so, with the same tstepcnt time-slices, we can model the wave 
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
    The returned array effectively is 2 rectangles of 0s surrounding a rows x cols grid evaluated for x and y where the origin (0,0) is taken at 
    the top left of the interior nonzero array. That is, the origin is at row index 2, col index 2.
    """
    f = np.zeros((rows+4,cols+4,ts))
    if t0==t1: #both cases do the same thing, but if t0==t1, this code runs significantly faster
        for row in range(2,rows+2):
            for col in range(2,cols+2):
                f[row,col,0]=evalFuncOfxyAtVal(t0,row-2,col-2)
        f[:,:,1]=f[:,:,0]
    
    else:
        for row in range(2,rows+2):
            for col in range(2,cols+2):
                f[row,col,0]=evalFuncOfxyAtVal(t0,row-2,col-2)
                f[row,col,1]=evalFuncOfxyAtVal(t1,row-2,col-2)
    
    return f

print '\nCalculating animation of an electromagnetic potential influenced by a passing gravitational wave based on the following parameters which can be adjusted near the top of the code.'

print '\n Computational parameters'
print '  Number of rows: rowcnt = %d\n  Number of columns: colcnt = %d\n  Number of time slices: tstepcnt = %d' %(rowcnt,colcnt,tstepcnt)
print '  (speed of electromagnetic wave * delta t / delta x)^2: K =',K,' [!Read docstrings and accompanying documentation before changing K!]'

print '\n Physical parameters:'
print '  Incident angle of gravitational wave: theta =',theta
print '  In f=epsilon*cos(kz-kct),'
print '    amplitude: eps =',eps,'\n    k: kgrav =',kgrav,'\n    wave speed: c =',c

print '\n Function of x and y that initializes the t=0 timeslice:',t0funcxy
print ' Function of x and y that initializes the t=1 timeslice:',t1funcxy,'\n\n'

print 'Initializing...'
u=initialSlices(rowcnt,colcnt,tstepcnt, t0funcxy, t1funcxy)
#now u is a rowcnt+4 by colcnt+4 grid with two surrounding boundaries of 0s

"""
See documentation for the derivations of the following equations and the meaning of the variable names.

T is a 4-dimensional array which can be thought of as a 2D array representing the R.M.R^-1 matrix as defined in the documentation, but each 
"element" of that matrix is itself a 2D array each of whose grid-point corresponds to the value of that matrix element at that grid-point on 
our discretized XY plane.
"""
print 'Computing...'
T=np.zeros((3,3,rowcnt,colcnt))

zprime=np.zeros((rowcnt,colcnt))
for row in range(0,rowcnt): #Y
    for col in range(0,colcnt): #X
        zprime[row,col]=col*np.sin(theta)**2-row*np.cos(theta)*np.sin(theta) #this is the inverse rotation of Z for our XY plane Z=0

for t in range(2,tstepcnt):
    f_tminus1=eps*np.cos(kgrav*zprime-kgrav*c*(t-1))
    
    T[0,0,:,:]=0.125*(4*f_tminus1*np.cos(2*theta)+f_tminus1*np.cos(4*theta)+3*f_tminus1+8)
    T[1,1,:,:]=0.125*(-8*f_tminus1*np.cos(2*theta)-f_tminus1*np.cos(4*theta)+f_tminus1+8)
    T[2,2,:,:]=0.5*(f_tminus1*np.cos(2*theta)-f_tminus1+2)
    T[0,1,:,:]=T[1,0,:,:]=0.125*f_tminus1*(6*np.sin(2*theta)+np.sin(4*theta))
    T[0,2,:,:]=T[2,0,:,:]=f_tminus1*np.cos(theta)*np.sin(theta)**2
    T[1,2,:,:]=T[2,1,:,:]=-f_tminus1*np.sin(theta)*np.cos(theta)**2
    
    delsqrdLHS_tminus1=0.25*(T[0,0,:,:]*(u[2:-2,4:,t-1]-2*u[2:-2,2:-2,t-1]+u[2:-2,:-4,t-1])+T[0,1,:,:]*(u[3:-1,3:-1,t-1]-u[3:-1,1:-3,t-1]-u[1:-3,3:-1,t-1]+u[1:-3,1:-3,t-1])+T[1,0,:,:]*(u[3:-1,3:-1,t-1]-u[1:-3,3:-1,t-1]-u[3:-1,1:-3,t-1]+u[1:-3,1:-3,t-1])+T[1,1,:,:]*(u[4:,2:-2,t-1]-2*u[2:-2,2:-2,t-1]+u[:-4,2:-2,t-1]))
    u[2:-2,2:-2,t]=K*delsqrdLHS_tminus1+2*u[2:-2,2:-2,t-1]-u[2:-2,2:-2,t-2]

#------------------ Numerical computations done. Solutions at all times stored in u. ------------------
print 'The computed animation has a spatial resolution of %d columns x %d rows and shows %d frames of time' % (colcnt, rowcnt, tstepcnt)

#Animating as a heat map showing each time-slice:
fig = mat.pyplot.figure(figsize=(6, 3))

ax = fig.add_subplot(1,1,1)
ax.set_title('colorMap')
ax.set_aspect('equal')

def frame(n):
    ax.clear()
    ax.imshow(u[2:-2,2:-2,n]) #doesn't show the outer 0 edges that were used as the outer boundary condition in the array slicing computations
    ax.set_aspect('equal')
    
dummy = anim.FuncAnimation(fig,frame,range(tstepcnt),interval=25)
pl.show()