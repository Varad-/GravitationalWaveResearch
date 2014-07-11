# -*- coding: utf-8 -*-
"""
@author: Varadarajan Srinivasan

"""
import numpy as np
import pylab as pl
import matplotlib as mat
import matplotlib.animation as anim


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

When edgeType is set to 0, the points along the edge of the grid are killed to 0, and then the discretized wave equation is applied for 
all later times only to the points on the interior of the grid. So, we discard the data along the boundary.
When edgeType is set to 1, the boundary values are discarded and replaced with the corresponding values computed using the discretized 
wave equation and applying the opposite edges' values cyclically. This uses for loops instead of array splicing and so runs much more 
slowly.

k is a positive unitless constant related to the speed of the wave and the discretization step sizes in space and time. For this numerical
method to successfully approximate a solution to the wave equation, k must be less than 1. The accompanying documentation file explains 
what k means, how it arises from discretizing the wave equation, and why it must be less than 1. The lower k is, the more "frames" are 
computed for a given wave period. The way this program is built, the time-resolution gained by lowering k does not change the runtime. 
Instead the drawback is just how far into the future we can see the wave because the number of time-slices computed is set. A higher k 
means the delta t between time-slices is higher (k is prop. to (delta t)^2) so, with the same tstepcnt time-slices, we can model the wave 
further into the future, but with a lower accuracy. Unlike evaluating an analytical solution at various times, the error from a high k 
will grow with each time-slice because this numerical solution iteratively uses the previous 2 slices' values. Recommended k is 0.1 to 0.4.
"""
edgeType=1 #0 for zeros along edges. 1 for cyclical edges (periodic boundary condition).

tstepcnt=801 #tstepcnt time-slices are indexed from t=0 to t=tstepcnt-1
rowcnt=400
colcnt=400

k=0.25 #k=(waveSpeed*dt/dx)**2

print '\nWave equation computations of the points along the edge of the grid can be changed between taking the would-be points outside the grid as 0 (edgeType=0), or cyclically taking the corresponding boundary points on the opposite edge (edgeType=1).'

print '\nThese fully adjustable parameters are currently set as:'
print ' edgeType =',edgeType
print ' Number of rows = rowcnt = %d\n Number of columns = colcnt = %d\n Number of time slices = tstepcnt = %d' %(rowcnt,colcnt,tstepcnt)
print ' waveSpeed^2 (dt/dx)^2 = k =',k,'   [!Read docstrings and accompanying documentation before changing k!]'

u=initialSlices(rowcnt,colcnt,tstepcnt)
print 'Computing...'

"""
The following discretized equations were derived from the wave equation, as explained in the accompanying documentation.
"""
if edgeType==0:
    for t in range(2,tstepcnt):
        u[1:-1,1:-1,t]=k*(u[:-2,1:-1,t-1]+u[2:,1:-1,t-1]+u[1:-1,:-2,t-1]+u[1:-1,2:,t-1]-4*u[1:-1,1:-1,t-1])+2*u[1:-1,1:-1,t-1]-u[1:-1,1:-1,t-2]
elif edgeType==1:
    for t in range(2,tstepcnt):
        u[1:-1,1:-1,t]=k*(u[:-2,1:-1,t-1]+u[2:,1:-1,t-1]+u[1:-1,:-2,t-1]+u[1:-1,2:,t-1]-4*u[1:-1,1:-1,t-1])+2*u[1:-1,1:-1,t-1]-u[1:-1,1:-1,t-2]
        for m in range(0,rowcnt): #so m represents the y direction (row index number)
            for n in range(0,colcnt): #so n represents the x direction (column index number)
                if m==0 and n!=0 and n!=colcnt-1: #0th row excl. corners
                    u[m][n][t]=k*(u[m][n+1][t-1]+u[m][n-1][t-1]+u[m+1][n][t-1]+ u[m-1][n][t-1] -4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
                elif m==rowcnt-1 and n!=0 and n!=colcnt-1: #last row ex.corners
                    u[m][n][t]=k*(u[m][n+1][t-1]+u[m][n-1][t-1]+ u[0][n][t-1] +u[m-1][n][t-1]-4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
                elif n==0 and m!=0 and m!=rowcnt-1: #0th column ex.corn.
                    u[m][n][t]=k*(u[m][n+1][t-1]+ u[m][n-1][t-1] +u[m+1][n][t-1]+u[m-1][n][t-1]-4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
                elif n==colcnt-1 and m!=0 and m!=rowcnt-1: #last column ex.c.
                    u[m][n][t]=k*( u[m][0][t-1] +u[m][n-1][t-1]+u[m+1][n][t-1]+u[m-1][n][t-1]-4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
            
                #corners:
                elif m==0 and n==0: #top left corner
                    u[m][n][t]=k*(u[m][n+1][t-1]+ u[m][n-1][t-1] +u[m+1][n][t-1]+ u[m-1][n][t-1] -4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
                elif m==0 and n==colcnt-1: #top right corner
                    u[m][n][t]=k*( u[m][0][t-1] +u[m][n-1][t-1]+u[m+1][n][t-1]+ u[m-1][n][t-1] -4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
                elif m==rowcnt-1 and n==0: #bottom left corner
                    u[m][n][t]=k*(u[m][n+1][t-1]+ u[m][n-1][t-1] + u[0][n][t-1] +u[m-1][n][t-1]-4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]
                elif m==rowcnt-1 and n==colcnt-1: #bottom right corner
                    u[m][n][t]=k*( u[m][0][t-1] +u[m][n-1][t-1]+ u[0][n][t-1] +u[m-1][n][t-1]-4*u[m][n][t-1])+2*u[m][n][t-1]-u[m][n][t-2]

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