import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from numpy import random as ran


'''
# -*- coding: utf-8 -*-
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

#draw cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r,r,r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s,e), color="b")

#draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x=np.cos(u)*np.sin(v)
y=np.sin(u)*np.sin(v)
z=np.cos(v)
ax.plot_wireframe(x, y, z, color="r")

#draw a point
ax.scatter([0],[0],[0],color="g",s=100)

#draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

a = Arrow3D([0,1],[0,1],[0,1], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
ax.add_artist(a)
plt.show()

'''


#--------------------
#tau_scatter
#Returns the possible optical depth to the next scattering depth
#INPUT: none
#Returns: tau_s
def tau_scatter():
    p = ran.uniform(0,1)
    tau_s = -np.log(p)
    return tau_s

#-------------------
#phi-to-xy
#Returns xy coordinates based on given angle
#NOTE: add to prev xyz value to get new position
#INPUT:
#    distance: path length xi of the ray
#    phi: the angle at whish the ray is cast
#Returns:
#    x,y: xy values corresponding to the change in position
def sphtocart(dist,phi,theta):

    x = dist*np.sin(phi)*np.cos(theta)
    y = dist*np.sin(phi)*np.sin(theta)
    z = dist*np.cos(phi)
    return x,y,z

#----------------
#rantheta
#Return a randomly calculated theta value from equ A5
#INPUT:
#    g: set initially, the asymmetry
#OUTPUT:
#    theta: The random theta value
def rantheta(g):
    p = ran.uniform(0,1)
    term1 = 1.+g**2
    term2 = 1.-g**2
    term3 = 1. - g + 2*g*p

    theta = (term1 - (term2/term3)**2)/(2.*g)

    return theta


#------------------
#ranphi
#Return a randomly selected phi value
#INPUT: None
#Output:
#    phi
def ranphi():
    p = ran.uniform(0,1)
    phi = 2.*np.pi*p

    return phi




#--------------
#ifexit
#Returns a boolean indicating if you've left the sphere of caring
#May change to vector inputs later
#INPUT:
#    initial coordinates and max R
#OUTPUT:
#    boolean indicating if at or beyond max radius
def ifexit(x0,y0,z0,x,y,z,R):
    return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2) >= R



#Steps:
#1) Walk until scattered (t=t0)
#2) Scatter
#3) Walk
#4) Repeat until beyond sphere edge
#---------------
#MonteCarloWalk
#Walks a ray until it scatters
#INPUT:
    # xyz starting positions (as array/vector)
    # sigma: absorption cross section
    # nH: number density of medium
    # g: the g value from init
    # R: maximum radius until ray end (as failsafe/shirt circuit
#OUTPUT:
    # xyz: ending ray xyz before it's scattered
    # dtau: The tau value that was added to the ray
def MonteCarloWalk(x,y,z,sigma,nH,g,R):
    xold = x
    yold = y
    zold = z
    t0 = tau_scatter()

    ds = -t0 /(sigma*nH)
    print ds

    theta = rantheta(g)
    phi = ranphi()

    dx,dy,dz = sphtocart(ds,phi,theta)

    xnew = xold + dx
    ynew = yold + dy
    znew = zold + dz

    # If beyond R but closer than 1.1R, or not exited yet
    # Maybe if almost exited as well
    if (ifexit(xold,yold,zold,xnew,ynew,znew,R) and ~ifexit(xold,yold,zold,xnew,ynew,znew,1.1*R) ) or ~ifexit(xold,yold,zold,xnew,ynew,znew,R):
        return xnew, ynew, znew, t0
    else:
        return MonteCarloWalk(xold,yold,zold,sigma,nH,g,R)
        #Redo sampling method recursively - may break and kill memory


    #TODO:
        # REject/rerun/truncate if beyond R

#-----------
#raystart
# Same as montecarlo walk, minus exit conditions (shouldn't happen), and with given angles
#def raystart()



'''
Plotting functions
'''
#---------------------
#drawArrow
#    Draws an arrow representing the path and direction between 2 points
#input A,B, axis
#    A,B: 2 points with the form [x,y] coordinates
#    axis: the axis that you want to draw said arrow on
#returns: none
def drawArrow(A, B, axis):
    axis.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              head_width=0.02, length_includes_head=True)
    return

#---------------------
#drawAllArrows
#Draws all the arrows of the path
#input: X,Y,axis
#   X,Y: 1-D Arrays of same size, containing series of x,y coordinates with matching indices
#   axis: axis you want these drawn on
#returns: none
def drawAllArrows(X,Y,axis):
    for i in xrange(len(X)-1):
        A = np.array([X[i],Y[i]])
        B = np.array([X[i+1],Y[i+1]])
        drawArrow(A,B,axis)
    return







'''
Pseudocode steps
1) Get initial point position and cloud info
2) generate tau of ray
3) Sample xi (path length) from tau
4) Calculate new angle of scatter
5) calculate new position and advance the ray
6) repeat 2-5 until ray reaches cloud boundary ( num trajectories=N)
7) repeat 2-6 for M rays
8) Calculate mean relative intensity at A(initial point)

A   : Observor placement point (const)
khat: Direction of ray - uniformly distributed over 4pi sr
N: Number of directions
M: Number of samplings of each direction

may need to plot only flattened data (xy only)
for impressiveness, plot in 3d


How to:
walk a ray
reject if tau isn't close to wanted tau
repeat until until it reaches the edge of the sphere
sum up all tau values along the rays
repeat for N/M rays
Sum up and average all intensities
?
'''

#-------------
#init
#Initializes all initial values
#returns:
#   A : Point of observer
#   M : Samplings/direction
#   N : Number of directions
#   I0: Initial intensity of ray
#   R : Radius of circle from origin from A
#   w : Albedo of the gas, mathis
#   g : asymmetry parameter, mathis
#   nH: number density of the medium, mathis, UNKNOWN
#   sigma: absorption cross section of Hydrogen
#   ds: The "wall distance" until the next scattering event

#
def init():
    Ax = 0.0
    Ay = 0.0
    Az = 0.0
    A = np.array([Ax,Ay,Az])
    M = 10
    N = 90
    I0 = 5.0E6
    R = 2.
    w = 0.8
    g = 0.8
    nH = 10**2.5
    sigma = .3326E-24 # cm
    ds = 1.


    return A,M,N,I0,R,w,g,nH,sigma,ds

def main():

    A,M,N,I0,R,w,g,nH,sigma,ds = init()
    ran.seed(0000)

    phis = np.linspace(0,2*np.pi,num=N)
    thetas = np.linspace(0,np.pi,num=N)


    for n in xrange(N):
        for m in xrange(M):
            theta0 = thetas[n]
            phi0 = phis[n]

            #Write initial raystart funciton


    #Generate testing points
    X = np.zeros(20)
    Y = np.zeros(20)
    Z = np.zeros(20)
    for i in xrange(20): # note: should start at point 0,0,0/origin
        theta = rantheta(g)
        phi = ranphi()
        X[i],Y[i],Z[i] = sphtocart(ds,phi,theta)
    #Plotting the final points
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)

    circle = plt.Circle(A, R, color='r', fill=False)
    ax1.add_artist(circle)

    ax1.scatter(X[0],Y[0],s=80,c='b',marker='*') #Colors
    ax1.plot(X[1:],Y[1:],'r.')
    drawAllArrows(X,Y,ax1)

    plt.show()


main()
