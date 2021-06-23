import numpy as np
import pyvista as pv
import sys
sys.path.append('..') # add the path to the folder containing CGfunctions.py here
import CGfunctions as cg ## import the CGfunctions

# AC: This example show how the CG functions can be used to compute the average particle volume fraction in the sample.

#############################################
# user defined functions :
#############################################
def readfiles(rigidfile): # read my input files and extract the particles positions
    rigid = pv.read(rigidfile) # load particles data
    loc = rigid.points # read the position of the center of mass of the particles
    npart=rigid.n_points-2 # compute number of particles (-2 because of the two walls in my simulations
    return loc, npart


def gen_CG_grid(Xmin,Xmax,Ymin,Ymax,Zmin,Zmax,nx,ny,nz): # generate a regular grid
    x=np.linspace(Xmin,Xmax,nx) #X coordinates
    y=np.linspace(Ymin,Ymax,ny) #Y coordinates
    z=np.linspace(Zmin,Zmax,nz) #Z coordinates
    xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')
    return xx,yy,zz


##############################################
# Main :
##############################################

# parameters
mass = 0.002*(0.0005**2)*2700 #all the particles have the same mass and volume (rho=2700 kg m-3 here)
nx=9 #nb of  CG pts along X
ny=9 #nb of CG pts along Y
nz=9 #nb og CG pts along 2
w = 0.001 # size of the CG sphere
nstep = 10 # number of time step to be accounted for

# Generate a regular grid
xx,yy,zz = gen_CG_grid(0.001,0.009,0.001,0.009,0.001,0.009,nx,ny,nz) 

# Reshape in three array(nb_cg_pts,1) 
x_cg = np.reshape(xx,(nx*ny*nz)) 
y_cg = np.reshape(yy,(nx*ny*nz))
z_cg = np.reshape(zz,(nx*ny*nz))

# generate the position vector (nb of CG points,3) for CG functions
pos_cg=np.transpose([x_cg,y_cg,z_cg])


for i in range(nstep):
    print('########### step: ', i+1 , ' ###########', sep =None)
    nf = i+1
    print('Read files')
    loc, npart=readfiles("DISPLAY/rigids_%s.vtu" % nf)
    pmass = np.zeros((len(loc),1))+mass
    
    # generates the visibility tables for the particles and contact points
    print('Compute visibility table and CG weigths')
    visibility_p,W_p = cg.Compute_CG_visibility_weight(pos_cg,loc,2*w,w,'Lucy',int_vec = False)
    
    print("     number of CG_node - particles interactions : ", len(visibility_p), sep =None)
    
    print('Compute mass density')
    rho = cg.Compute_CG_scalar(visibility_p,W_p,pmass,nx*ny*nz)
    
    print('Average phi: ', np.average(rho[:,0]/2700))
