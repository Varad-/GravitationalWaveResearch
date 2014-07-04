# -*- coding: utf-8 -*-
"""
@author: Varadarajan Srinivasan

This program discretizes and solves the wave equation in 2+1 dimensions for user-inputted grid size and any user-inputted boundary conditions.
"""
import numpy as np
import pylab as pl
import matplotlib as mat
import matplotlib.animation as m


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


def makeInit(rows, cols, tsteps):
    """
    Initializes the 2+1 dimensional array with all zeros and initializes the t=0 boundary 
    using point-by-point value entry or a user-inputted function of x and y
    This is then copied to the t=1 slice as well because 2 initial slices are needed in the discretized wave equation
    and it makes more sense to use the same value instead of having the slice before the boundary to be all 0s because
    that would pollute the discretized representation of the second order PDE with an artificially high rate of change.
    (more details in the accompanying documentation WaveEquation.pdf)
    """
    f = np.zeros((rows,cols,tsteps))

    if opt == 1:
        t0funcxy = raw_input('\nDefine a function of x and y that sets the t=0 grid (in Python syntax). u(x,y)=')
        for row in range(0,rows):
            for col in range(0,cols):
                f[row][col][0]=evalFuncOfxyAtVal(t0funcxy,row,col)
                f[row][col][1]=f[row][col][0]
    
    elif opt == 0:
        for row in range(0,rows):
            for col in range(0,cols):
                f[row][col][0]=input('u(%d,%d) = ' % (row, col))
                f[row][col][1]=f[row][col][0]
    else:
        print '\nError: opt must be set to 0 or 1'
        return
    return f


def solveTimeSlice(f,t,rows,cols,bndcnd):
    """
    Solves the discretized wave equation over a given time-slice of the 2+1d array based on the previously existing values. So, this only works if when a time-slice is called 
    to be updated, the previous 2 time-slices have already been computed and stored in the array.
    
    A detailed explanation of how this discretized equation was derived is in the accompanying documentation file.
    -------------------------------------------------
    Option 0: Zeros
    We imagine the grid is surrounded by a rectangle of zeros so that whenever the wave equation needs to be evaluated at edges,
    we take the outside imaginary point as having value 0, regardless of the environment.    
    
    Option 1: Cyclical
    We imagine the grid is surrounded by a rectangle of zeros so that whenever the wave equation needs to be evaluated at edges,
    we take the outside imaginary point as having value 0, regardless of the environment.
    -------------------------------------------------
    
    For easy readability in the specific case equations, their terms that differ from the general equation are surrounded by spaces.
    """
    #Option 0 (see docstring)
    if bndcnd == 0:
        for m in range(0,rows): #so m represents the y direction (row index number)
            for n in range(0,cols): #so n represents the x direction (column index number)
                if m!=0 and m!=rows-1 and n!=0 and n!=cols-1: #!!the general equation:
                    f[m][n][t]=k*(f[m][n+1][t-1]+f[m][n-1][t-1]+f[m+1][n][t-1]+f[m-1][n][t-1]-4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                
                #edges:
                elif m==0 and n!=0 and n!=cols-1: #0th row excl. corners
                    f[m][n][t]=k*(f[m][n+1][t-1]+f[m][n-1][t-1]+f[m+1][n][t-1]+ 0 -4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                elif m==rows-1 and n!=0 and n!=cols-1: #last row ex.corners
                    f[m][n][t]=k*(f[m][n+1][t-1]+f[m][n-1][t-1]+ 0 +f[m-1][n][t-1]-4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                elif n==0 and m!=0 and m!=rows-1: #0th column ex.corn.
                    f[m][n][t]=k*(f[m][n+1][t-1]+ 0 +f[m+1][n][t-1]+f[m-1][n][t-1]-4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                elif n==cols-1 and m!=0 and m!=rows-1: #last column ex.c.
                    f[m][n][t]=k*( 0 +f[m][n-1][t-1]+f[m+1][n][t-1]+f[m-1][n][t-1]-4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                    
                #corners:
                elif m==0 and n==0: #top left corner
                    f[m][n][t]=k*(f[m][n+1][t-1]+ 0 +f[m+1][n][t-1]+ 0 -4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                elif m==0 and n==cols-1: #top right corner
                    f[m][n][t]=k*( 0 +f[m][n-1][t-1]+f[m+1][n][t-1]+ 0 -4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                elif m==rows-1 and n==0: #bottom left corner
                    f[m][n][t]=k*(f[m][n+1][t-1]+ 0 + 0 +f[m-1][n][t-1]-4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                elif m==rows-1 and n==cols-1: #bottom right corner
                    f[m][n][t]=k*( 0 +f[m][n-1][t-1]+ 0 +f[m-1][n][t-1]-4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
    
    #Option 1 (see docstring)
    elif bndcnd == 1:
        for m in range(0,rows): #so m represents the y direction (row index number)
            for n in range(0,cols): #so n represents the x direction (column index number)
                if m!=0 and m!=rows-1 and n!=0 and n!=cols-1: #!!the general equation:
                    f[m][n][t]=k*(f[m][n+1][t-1]+f[m][n-1][t-1]+f[m+1][n][t-1]+f[m-1][n][t-1]-4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                
                #edges:                
                elif m==0 and n!=0 and n!=cols-1: #0th row excl. corners
                    f[m][n][t]=k*(f[m][n+1][t-1]+f[m][n-1][t-1]+f[m+1][n][t-1]+ f[m-1][n][t-1] -4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                elif m==rows-1 and n!=0 and n!=cols-1: #last row ex.corners
                    f[m][n][t]=k*(f[m][n+1][t-1]+f[m][n-1][t-1]+ f[0][n][t-1] +f[m-1][n][t-1]-4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                elif n==0 and m!=0 and m!=rows-1: #0th column ex.corn.
                    f[m][n][t]=k*(f[m][n+1][t-1]+ f[m][n-1][t-1] +f[m+1][n][t-1]+f[m-1][n][t-1]-4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                elif n==cols-1 and m!=0 and m!=rows-1: #last column ex.c.
                    f[m][n][t]=k*( f[m][0][t-1] +f[m][n-1][t-1]+f[m+1][n][t-1]+f[m-1][n][t-1]-4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                    
                #corners:
                elif m==0 and n==0: #top left corner
                    f[m][n][t]=k*(f[m][n+1][t-1]+ f[m][n-1][t-1] +f[m+1][n][t-1]+ f[m-1][n][t-1] -4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                elif m==0 and n==cols-1: #top right corner
                    f[m][n][t]=k*( f[m][0][t-1] +f[m][n-1][t-1]+f[m+1][n][t-1]+ f[m-1][n][t-1] -4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                elif m==rows-1 and n==0: #bottom left corner
                    f[m][n][t]=k*(f[m][n+1][t-1]+ f[m][n-1][t-1] + f[0][n][t-1] +f[m-1][n][t-1]-4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
                elif m==rows-1 and n==cols-1: #bottom right corner
                    f[m][n][t]=k*( f[m][0][t-1] +f[m][n-1][t-1]+ f[0][n][t-1] +f[m-1][n][t-1]-4*f[m][n][t-1])+2*f[m][n][t-1]-f[m][n][t-2]
    """
    I have written 2 separate blocks for Option 0 (bndcnd==0) and Option 1 (bndcnd==1). We could write these more simply as just one block 
    by using the code for the Option 1 block, but multiplying all the terms surrounded by spaces by bndcnd. This way, they stay the same
    in Option 1, since bndcnd==1, but reduce to 0 when bndcnd==0, as desired in the specific case equations for Option 0.
    
    I have deliberately not done this, however, so that these equations can be more readily copied for other uses.
    """
    return f

"""
-----------------------parameters-----------------------

When edgeType is set to 0, the points along the edge of the grid will be solved with the discretized wave equation imagining that the 
adjacent points that would be outside the grid are 0. When edgeType is set to 1, it cyclically takes the corresponding values from the 
opposite edges of the grid.

k is a positive unitless constant related to the speed of the wave and the discretization step sizes in space and time. For this numerical
method to successfully approximate a solution to the wave equation, k must be less than 1. The accompanying documentation file explains 
what k means, how it arises from discretizing the wave equation, and why it must be less than 1. The lower k is, the more "frames" are 
computed for a given wave period. The way this program is built, the time-resolution gained by lowering k does not change the runtime. 
Instead the drawback is just how far into the future we can see the wave because the number of time-slices computed is set. A higher k 
means the delta t between time-slices is higher (k is prop. to (delta t)^2) so, with the same tstepcnt time-slices, we can model the wave 
further into the future, but with a lower accuracy. Unlike evaluating an analytical solution at various times, the error from a high k 
will grow with each time-slice because this numerical solution iteratively uses the previous 2 slices' values. Recommended k is 0.25.
"""
edgeType=1
opt = 1 #0 for manual point-by-point boundary condition input.1 to input as a function of x and y

tstepcnt=400 #creates tstepcnt slices indexed from t=0 to t=tstepcnt-1
rowcnt=200
colcnt=200

k=0.25 #for grid spacings taken to be 1m, and a wave travelling at c, k=0.25 corresponds to a time-resolution of 6*10^8 s^-1

#dx = 1 #this is the distance between grid points which we can take to be 1m
#waveSpeed=299792458 #c in m/s
#dt = 0.000000001667820476 #time spacing in seconds
#k=(waveSpeed*dt/dx)**2

print '\nInput method for the boundary conditions (initial time slice) can be changed between manual (opt=0) and function (opt=1).'
print '\nWave equation computations of the points along the edge of the grid can be changed between taking the would-be points outside the grid as 0 (edgeType=0), or cyclically taking the corresponding boundary points on the opposite edge (edgeType=1).'

print '\nThese fully adjustable parameters are currently set as:\n'
print ' opt =',opt
print ' edgeType =',edgeType

print '\n Number of rows = rowcnt = %d\n Number of columns = colcnt = %d\n Number of time slices = tstepcnt = %d' %(rowcnt,colcnt,tstepcnt)
print '\n waveSpeed^2 (dt/dx)^2 = k =',k,'   [!Read docstrings and accompanying documentation before changing k!]'

u = makeInit(rowcnt,colcnt,tstepcnt)
print 'Computing...'
print 'Progress:'
for time in range(2,tstepcnt):
    u=solveTimeSlice(u,time,rowcnt,colcnt,edgeType)
    prog=100.0*time/tstepcnt    
    for progcheck in range(0,100,5):    
        if prog>=progcheck and prog-100.0/tstepcnt<progcheck :
            print '%d%%' % progcheck

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
    
dummy = m.FuncAnimation(fig,frame,range(tstepcnt),interval=25)
pl.show()


#run the following loop to print the time-slices chronologically as 2d arrays:
"""
for time in range(0,tstepcnt):
    print 't = ',time
    print u[:,:,time]
"""