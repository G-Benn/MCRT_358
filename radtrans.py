import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from numpy import random as ran
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

plt.style.use('seaborn-dark-palette')

#For those on Python3
def xrange(x):
    return iter(range(x))
'''NOTE: SIGMA/NH IS CAUSING DS TO BECOME TOO LARGE. USING TEST VALUE'''


#--------------------
#tau_scatter
#Returns the possible optical depth to the next scattering depth
#INPUT: none
#Returns: tau_s
def tau_scatter():
    p = ran.uniform(0,1)
    tau_s = -np.log(p)
    #print "p",p
    #print "tau_s",tau_s
    return tau_s

#-------------------
#Returns the tau value that is a function of the albedo
#INPUT:
    # w: The albedo of the cloud
    # tau: The original scattered tau
#OUTPUT:
    # New albedo-scaled tau value
def alb_tau(w,tau):
    return tau*((1./w) - 1)


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


#------------
#updates the nH value becauce it varies/is not constant - normally
#INPT: nH: current density
#OUTPUT: nH+deltanH
def nHupdate(nH):
    rannum = np.random.normal(0,.04*nH)
    nH += rannum
    return nH




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
    #print "t0", t0
    nH = nHupdate(nH)
    ds = -t0 /(sigma*nH)
    #print "ds",ds

    theta = rantheta(g)
    phi = ranphi()

    dx,dy,dz = sphtocart(ds,phi,theta)

    xnew = xold + dx
    ynew = yold + dy
    znew = zold + dz

    # If beyond R but closer than 1.1R, or not exited yet
    # Maybe if almost exited as well
    #if (ifexit(xold,yold,zold,xnew,ynew,znew,R) and ~ifexit(xold,yold,zold,xnew,ynew,znew,1.1*R) ) or ~ifexit(xold,yold,zold,xnew,ynew,znew,R):
    return xnew, ynew, znew, t0
    #else:
    #    return MonteCarloWalk(xold,yold,zold,sigma,nH,g,R)
        #Redo sampling method recursively - may break and kill memory


    #TODO:
        # REject/rerun/truncate if beyond R

#-----------
#raystart
# Same as montecarlo walk, minus exit conditions (shouldn't happen), and with given angles
#INPUT:
    # pos: vector of xyz intial points
    # phi: initila phi of ray
    # theta: initial theta of ray
    # ds: the length of the ray
#OUTPUT:
    # pos: updated positions of the ray
def raystart(pos,phi,theta,ds):
    pos[0] += ds*np.sin(theta)*np.cos(phi)
    pos[1] += ds*np.sin(theta)*np.sin(phi)
    pos[2] += ds*np.cos(theta)

    #print ds*np.sin(theta)*np.cos(phi)
    #print ds*np.sin(theta)*np.sin(phi)
    #print ds * np.cos(theta)
    return pos


'''
Plotting functions
'''

#---------------------
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)



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
    I0 = 1.5E10
    R = .5
    w = 0.8
    g = 0.8
    nH = 1E3
    sigma = 3.326E-2#E-24 # cm # TEST VALUE
    ds = 5E-3


    return A,M,N,I0,R,w,g,nH,sigma,ds

def main():

    A,M,N,I0,R,w,g,nH,sigma,ds = init()
    ran.seed(0000)


    #setup figure and plot initial sphere
    fig = plt.figure(num=1,figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax._axis3don = False
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Random walk for a ray")

    colors = ['k','b','g','r','c','m','y','0.75','#eeefff','#efa023'] # Length M, will probably need to be changed, temp

    # draw outer sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = R * np.cos(u) * np.sin(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

    # draw initial starting point
    ax.scatter([A[0]], [A[1]], [A[2]], color="g", marker='$\\ast$', s=150)

    phis = np.linspace(0,2*np.pi,num=N)
    thetas = np.linspace(0,np.pi,num=N)

    taus = np.zeros((N,M))

    for n in xrange(N):
        for m in xrange(M):

            pos = np.full((1500, 3), None, dtype='float64')  # the array of points - index corresponds to pos number - 50 is placeholder
            # print pos
            # print A
            tau = 0

            theta0 = thetas[n]
            phi0 = phis[n]

            pos[0,:] = A[:]
            pos[1, :] = raystart(pos[0, :].copy(), phi0, theta0, ds)

            run=1
            while run < 1499 and ~ifexit(pos[0,0],pos[0,1],pos[0,2],pos[run,0],pos[run,1],pos[run,2],R):
                if run==1498: #last run
                    print "Need more possible runs"

                dx,dy,dz, dtau = MonteCarloWalk(pos[run,0],pos[run,1],pos[run,2],sigma,nH,g,R)
                pos[run+1,:] = np.array([dx,dy,dz])

                # Tau is being scaled by albedo here - may need to be placed into MCW instead
                tau += alb_tau(w,dtau)
                run += 1

                #print "runs", run
                #print pos[:5, :]
                #print "tau", tau


                #Mask and plot all arrows for run
                maskpos = np.isfinite(pos[:, 0])
                goodpos = pos[maskpos, :]
                '''
                for i in xrange(goodpos.shape[0] - 1):
                    a = Arrow3D([goodpos[i, 0], goodpos[i + 1, 0]], [goodpos[i, 1], goodpos[i + 1, 1]],
                                [goodpos[i, 2], goodpos[i + 1, 2]], mutation_scale=10, lw=1, arrowstyle="-|>",color=colors[m])
                    ax.add_artist(a)
                '''
            taus[n,m] = tau

            #break
        #break
    #print taus

    tau_max = np.amax(taus)
    tau_min = np.amin(taus)
    tau_avg = np.sum(taus) * (1./float(N*M))
    I_end = I0 * np.exp(-tau_avg)
    print("Average tau %.5e" % tau_avg) # will be too low, b/c array mostly zeros
    print("Max tau: %.5e \t Min tau %.5e" % (tau_max, tau_min))


    print("Starting Intensity %.5e" % I0)
    print ("Average ending Intensity %.5e" % I_end) # Too high, b/c tau too low
    print("Max Iend: %.5e \t Min Iend %.5e" % (I0 * np.exp(-tau_max), I0 * np.exp(-tau_min)))
    #Output of the Intensity

    plt.show()




main()
