# -*- coding: utf-8 -*-
"""
@author: Varadarajan Srinivasan

Produces a (2+1)D animation showing a (electromagnetic) wave being influenced by a gravitational wave after taking some (3+1)D considerations 
into account such that results can be shown for any arbitrarily inclined angle at which the gravitational wave comes in.
"""

import numpy as np
import pylab as pl
import matplotlib as mat
import matplotlib.animation as anim

def xySliceOf3DRotConj(Q,angle):
    """
    Takes a 3x3 matrix Q and computes RQR^-1 where R is the 3x3 matrix for a rotation by the given angle about z' and then about x'.    
    Since R is orthogonal, R^-1=R^transpose
    Returns the xy slice of RQR^-1    
    """
    R=np.zeros((3,3))

    R[0,0]=np.cos(angle)
    R[0,1]=-np.cos(angle)*np.sin(angle)
    R[0,2]=np.sin(angle)**2
    R[1,0]=np.sin(angle)
    R[1,1]=np.cos(angle)**2
    R[1,2]=-np.cos(angle)*np.sin(angle)
    R[2,0]=0
    R[2,1]=np.sin(angle)
    R[2,2]=np.cos(angle)
    
    return (R.dot(Q).dot(R.T))[:-1,:-1]

"""
def delnDxSlice(tslice):
    
    Takes a time slice and returns an evaluated 2d grid of the grid spacing (delta x) times the numerical derivative of f[row][col][time] 
    with respect to x (i.e. column index). Derivatives are only evaluated on the interior of the grid. The edges are left as 0.
    
    nDx=np.zeros(np.shape(tslice))    
    nDx[1:-1,1:-1]=0.5*(tslice[1:-1,2:]-tslice[1:-1,:-2])
    return nDx

def delnDySlice(tslice):
    
    Takes a time slice and returns an evaluated 2d grid of the grid spacing (delta y) times the numerical derivative of f[row][col][time] 
    with respect to y (i.e. row index). Note that in the matrix this means the y coordinate points downwards. Derivatives are only evaluated 
    on the interior of the grid. The edges are left as 0.
    
    nDy=np.zeros(np.shape(tslice))
    nDy[1:-1,1:-1]=0.5*(tslice[2:,1:-1]-tslice[:-2,1:-1])
    return nDy
"""

def evalFuncOfxyAtVal(TwoVarFunc,rowval,colval): ##f MUST BE IN TERMS OF x,y
    """
    Evaluates a mathematical function of 2 variables written as a string in Python syntax in terms of x and y at a specified point (xval,yval). 
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
    """
    f = np.zeros((rows,cols,ts))
    if t0==t1: #both cases do the same thing, but if t0==t1, this code runs significantly faster
        for row in range(0,rows):
            for col in range(0,cols):
                f[row,col,0]=evalFuncOfxyAtVal(t0,row,col)
        f[:,:,1]=f[:,:,0]
    
    else:
        for row in range(0,rows):
            for col in range(0,cols):
                f[row,col,0]=evalFuncOfxyAtVal(t0,row,col)
                f[row,col,1]=evalFuncOfxyAtVal(t1,row,col)
    
    return f

"""
-----------------------parameters-----------------------
K is a positive unitless constant related to the speed of the wave and the discretization step sizes in space and time. For this numerical
method to successfully approximate a solution to the wave equation, K must be less than 1. The accompanying documentation file explains 
what K means, how it arises from discretizing the wave equation, and why it must be less than 1. The lower K is, the more "frames" are 
computed for a given wave period. The way this program is built, the time-resolution gained by lowering k does not change the runtime. 
Instead the drawback is just how far into the future we can see the wave because the number of time-slices computed is set. A higher K 
means the delta t between time-slices is higher (K is prop. to (delta t)^2) so, with the same tstepcnt time-slices, we can model the wave 
further into the future, but with a lower accuracy. Unlike evaluating an analytical solution at various times, the error from a high K 
will grow with each time-slice because this numerical solution iteratively uses the previous 2 slices' values. Recommended K is 0.1 to 0.4.
"""
tstepcnt=120 #tstepcnt time-slices are indexed from t=0 to t=tstepcnt-1
rowcnt=80
colcnt=80

t0funcxy = 'np.exp(-((x-colcnt/3.0)**2/(colcnt/2.0)+(y-rowcnt/2.0)**2/(rowcnt/2.0)))' #(string) function of x and y that initializes the t=0 slice
t1funcxy = 'np.exp(-(((x-K**0.5)-colcnt/3.0)**2/(colcnt/2.0)+(y-rowcnt/2.0)**2/(rowcnt/2.0)))' #(string) function of x and y that initializes the t=1 slice
theta = np.pi/3 #angle of incidence of grav wave
eps=0 #meaning of epsilon is in the documentation
kgrav=0.02 #this is the k_grav from cos(kz-kt)
K=0.03 #K=(c*dt/dx)**2
c=1 #c=1 for units of the wave speed

print '\nWave equation computations of the points along the edge of the grid can be changed between taking the would-be points outside the grid as 0 (edgeType=0), or cyclically taking the corresponding boundary points on the opposite edge (edgeType=1).'

print '\nThese are the values currently set for the fully adjustable parameters:'

print '\n Computational parameters'
print '  Number of rows: rowcnt = %d\n  Number of columns: colcnt = %d\n  Number of time slices: tstepcnt = %d' %(rowcnt,colcnt,tstepcnt)
print '  (speed of electromagnetic wave * delta t / delta x)^2: K =',K,' [!Read docstrings and accompanying documentation before changing K!]'

print '\n Physical parameters:'
print '  Incident angle of gravitational wave: theta =',theta
print '  In f=epsilon*cos(kz-kct),'
print '    amplitude: eps =',eps,'\n    k: kgrav =',kgrav,'\n    wave speed: c =',c

print ' \nFunction of x and y that initializes the t=0 timeslice:',t0funcxy
print ' Function of x and y that initializes the t=1 timeslice:',t1funcxy,'\n\n'

print 'Initializing...'
u=initialSlices(rowcnt,colcnt,tstepcnt, t0funcxy, t1funcxy)

"""
See documentation for the derivations of the following equations and the meaning of these variable names.
I have also subtracted one time step from every term since it makes more sense in this programming context.
"""
print 'Computing...'
M=np.zeros((3,3))
M[2,2]=1
"""
leaves the 2 outermost rectangles along the edges as 0 (waves reflect):
"""
for t in range(2,tstepcnt):
    for m in range(2,rowcnt-2): #row indices -> Y
        for n in range(2,colcnt-2): #column indices -> X
            zprime=n*np.sin(theta)**2-m*np.cos(theta)*np.sin(theta) #for our XY plane, zprime = inverse rotation of Z eval'd at Z=0
            f_tminus1=eps*np.cos(kgrav*zprime-kgrav*c*(t-1))
            M[0,0]=1+f_tminus1
            M[1,1]=1-f_tminus1
            T=xySliceOf3DRotConj(M,theta)
            delsqrdLHS_tminus1=0.25*(T[0,0]*(u[m,n+2,t-1]-2*u[m,n,t-1]+u[m,n-2,t-1])+T[0,1]*(u[m+1,n+1,t-1]-u[m+1,n-1,t-1]-u[m-1,n+1,t-1]+u[m-1,n-1,t-1])+T[1,0]*(u[m+1,n+1,t-1]-u[m-1,n+1,t-1]-u[m+1,n-1,t-1]+u[m-1,n-1,t-1])+T[1,1]*(u[m+2,n,t-1]-2*u[m,n,t-1]+u[m-2,n,t-1]))
            u[m,n,t]=K*delsqrdLHS_tminus1+2*u[m,n,t-1]-u[m,n,t-2]

"""
------------------ numerical computations done, solutions at all times stored in u ------------------
"""
print '100% Done.'
print 'The computed animation has a spatial resolution of %d columns x %d rows and shows %d frames of time' % (colcnt, rowcnt, tstepcnt)

#Animating as a heat map showing each time-slice:
fig = mat.pyplot.figure(figsize=(6, 3))

ax = fig.add_subplot(1,1,1)
ax.set_title('colorMap')
ax.set_aspect('equal')

def frame(n):
    ax.clear()
    ax.imshow(u[:,:,n])
    ax.set_aspect('equal')
    
dummy = anim.FuncAnimation(fig,frame,range(tstepcnt),interval=25)
pl.show()

#run the following loop to print the time-slices chronologically as 2d arrays:
"""
for time in range(0,tstepcnt):
    print 't = ',time
    print u[:,:,time]
"""