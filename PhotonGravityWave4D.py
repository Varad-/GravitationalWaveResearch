# -*- coding: utf-8 -*-
"""
@author: Varadarajan Srinivasan

(2+1)D animation showing a (electromagnetic) wave being influenced by a gravitational wave for the special case that 
the gravitational wave is perpendicular to the plane.
"""

import numpy as np
import pylab as pl
import matplotlib as mat
import matplotlib.animation as anim

def TwoDimRotConj(Q,angle):
    """
    Takes a 2x2 matrix Q and returns RQR^-1 where R is the 2x2 rotation matrix for the given angle. Since R is orthogonal, R^-1=R^transpose
    """
    R=np.zeros((2,2))
    R[0,0]=np.cos(angle)
    R[0,1]=-np.sin(angle)
    R[1,0]=np.sin(angle)
    R[1,1]=np.cos(angle)
    return R.dot(Q).dot(R.T)

def ThreeDimInvRotz(Q, angle):
    """
    Takes a 3x3 matrix Q and returns R^-1 Q where R is the 3x3 z-rotation matrix for the given angle. (R^-1 = R^transpose)
    """
    Rinv=np.zeros((3,3))
    Rinv[0,0]=np.cos(angle)  
    Rinv[1,1]=np.cos(angle)
    Rinv[2,2]=1
    Rinv[1,0]=-np.sin(angle)
    Rinv[0,1]=np.sin(angle)
    return Rinv.dot(Q)

def rotatez(angle):
    """
    ??? getting nonsensical results
    """
    
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


def initialSlices(rows, cols, ts):
    """
    Initializes a 2+1 dimensional array with all zeros and initializes the t=0 boundary as a user-inputted function of x and y.
    This is then copied to the t=1 slice as well because 2 initial slices are needed in the discretized wave equation
    and it makes more sense to use the same value instead of having the slice before the boundary to be all 0s because
    that would pollute the discretized representation of the second order PDE with an artificially high rate of change.
    """
    f = np.zeros((rows,cols,ts))
    t0funcxy = raw_input('\nDefine a function of x and y that sets the t=0 grid (in Python syntax). u(x,y)=')
    print 'Initializing...'
    for row in range(0,rows):
        for col in range(0,cols):
            f[row,col,0]=evalFuncOfxyAtVal(t0funcxy,row,col)
    
    f[:,:,1]=f[:,:,0]
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
tstepcnt=401 #tstepcnt time-slices are indexed from t=0 to t=tstepcnt-1
rowcnt=200
colcnt=200

theta = np.pi/6 #angle of incidence of grav wave
eps=0.3 #meaning of epsilon is in the documentation
kgrav=0.02 #this is the k_grav from cos(kz-kt)
K=0.25 #K=(c*dt/dx)**2

c=1 #c=1 for units of the wave speed

print '\nWave equation computations of the points along the edge of the grid can be changed between taking the would-be points outside the grid as 0 (edgeType=0), or cyclically taking the corresponding boundary points on the opposite edge (edgeType=1).'

print '\nThese are the values currently set for the fully adjustable parameters:'

print '\n Computational parameters'
print '  Number of rows: rowcnt = %d\n  Number of columns: colcnt = %d\n  Number of time slices: tstepcnt = %d' %(rowcnt,colcnt,tstepcnt)
print '  (speed of electromagnetic wave * delta t / delta x)^2: K =',K,' [!Read docstrings and accompanying documentation before changing K!]'

print '\n Physical parameters:'
print '  Incident angle of gravitational wave: theta =',theta
print '  In f=epsilon*cos(kz-kt),'
print '    epsilon: eps =',eps,'\n    k: kgrav =',kgrav
print '  Speed of electromagnetic wave: c =',c

u=initialSlices(rowcnt,colcnt,tstepcnt)
print 'Computing...'
"""
See documentation for the derivations of the following equations and the meaning of these variable names
"""
M=np.zeros((2,2))
z=rotatez(theta)#rotate R^-1 the z of the grav wave coordinate
for t in range(2,tstepcnt):
    f=eps*np.cos(kgrav*(z-c*t))
    M[0,0]=1+f
    M[1,1]=1-f
    T=TwoDimRotConj(M,theta)
    delsquaredLHS=delnDxSlice(T[0,0]*delnDxSlice(u[:,:,t-1])+T[0,1]*delnDySlice(u[:,:,t-1]))+delnDySlice(T[1,0]*delnDxSlice(u[:,:,t-1])+T[1,1]*delnDySlice(u[:,:,t-1]))
    u[:,:,t]=K*delsquaredLHS+2*u[:,:,t-1]-u[:,:,t-2]

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