import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from numpy import random as ran



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
#isexit
#Returns a boolean indicating if you've left the sphere of caring
#May change to vector inputs later
#INPUT:
#    initial coordinates and max R
#OUTPUT:
#    boolean indicating if at or beyond max radius
def isexit(x0,y0,z0,x,y,z,R):
    return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2) >= R




#---------------




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
              head_width=0.2, length_includes_head=True)
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
#
def init():
    Ax = 0.0
    Ay = 0.0
    A = np.array([Ax,Ay])
    M = 10
    N = 90
    I0 = 5.0E6
    R = 5.
    w = 0.8
    g = 0.8
    nH = 10**2.5
    sigma = .3326E-24 # cm


    return A,M,N,I0,R,w,g,nH,sigma

def main():

    A,M,N,I0,R,w = init()


    #Generate testing points
    X = np.linspace(0,10,num=10)
    Y = np.linspace(-10,0,num=10)
    X[:] = 10*ran.random(10)[:]+X[:]
    Y[:] = 10*ran.random(10)[:]+Y[:]
    X[0] = A[0]
    Y[0] = A[1]
    #Plotting the final points
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)

    circle = plt.Circle(A, R, color='r', fill=False)
    ax1.add_artist(circle)

    ax1.plot(X[0],Y[0],'g*')
    ax1.plot(X,Y,'r.')
    drawAllArrows(X,Y,ax1)

    plt.show()


main()
