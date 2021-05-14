import numpy as np
import pyvista as pv
import sys
sys.path.append('..') # add the path to the folder containing CGfunctions.py.
import CGfunctions as cg ## import the CGfunctions precompiled here

# AC: This example show how the CG functions should be used. It return the average particle volume fraction in the sample.

#############################################
# user defined functions :
#############################################
def readfiles(ptcfile,rigidfile): # read my input files
    ptc = pv.read(ptcfile) # load contact points data
    rigid = pv.read(rigidfile) # load particles data
    force = rigid.get_array('Reac') # read sum of contact force on each particles
    fext = rigid.get_array('Fext') # read sum of the external forces on the particles
    loc = rigid.points # read the position of the center of mass of the particles
    vp = rigid.get_array('Velocy') # read the particles volocity vector
    npart=rigid.n_points-2 # compute number of particles (-2 because of the two walls in my simulations
    loc_ptc = ptc.points # read location of the contact points
    F_ptc = ptc.get_array('R') # read force vector of the contact points
    N_ptc = ptc.get_array('N') # read contact normal at the contact points
    return loc,vp,force, fext, npart, loc_ptc, F_ptc, N_ptc


def gen_CG_grid(Xmin,Xmax,Ymin,Ymax,Zmin,Zmax,nx,ny,nz): # generate a regular grid
    x=np.linspace(Xmin,Xmax,nx)
    y=np.linspace(Ymin,Ymax,ny)
    z=np.linspace(Zmin,Zmax,nz)
    xx,yy,zz = np.meshgrid(x,y,z)
    return xx,yy,zz


##############################################
# Main :
##############################################

# parameters
mass = 0.002*(0.0005**2)*2700 #all the particles have the same mass and volume
nx=9 #nb of  CG pts along X
ny=9 #nb of CG pts along Y
nz=9 #nb og CG pts along 2
w = 0.00105 # size of the CG sphere
nstep = 10 # number of time step to be accounted for

# Generate a regular grid
xx,yy,zz = gen_CG_grid(0.001,0.009,0.001,0.009,0.001,0.009,nx,ny,nz) 

# Reshape in three array(nb_cg_pts,1) 
x_cg = np.reshape(xx,(nx*ny*nz)) 
y_cg = np.reshape(yy,(nx*ny*nz))
z_cg = np.reshape(zz,(nx*ny*nz))

# generate the position vector for CG functions
pos_cg=np.transpose([x_cg,y_cg,z_cg])

# initiate some outputs
q_p=np.zeros(nx*ny*nz) # q/p ratio at the CG nodes
phi_avg=np.zeros(nstep) # volume fraction at the CG nodes

for i in range(nstep):
    print('########### step: ', i+1 , ' ###########', sep =None)
    nf = i+1
    print('Read files')
    loc,vp,force,fext,npart, loc_ptc, F_ptc, N_ptc=readfiles("DISPLAY/ptc_%s.vtp" % nf,"DISPLAY/rigids_%s.vtu" % nf)
    pmass = np.zeros((len(vp),1))+mass
    
    # generates the visibility tables for the particles and contact points
    print('Compute visibility table and CG weigths')
    visibility_p,W_p = cg.Compute_CG_visibility_weight(pos_cg,loc,3*w,w,'Lucy') #particles
    visibility_c,W_c = cg.Compute_CG_visibility_weight(pos_cg,loc_ptc,3*w,w,'Lucy') #contacts
    
    print("     number of CG_node - particles interactions : ", len(visibility_p), sep =None)
    print("     number of CG_node - contact_pts interactions : ", len(visibility_c), sep =None)
    
    # Compute CG momentum:
    print('Compute momentum')
    p = cg.Compute_CG_vector(visibility_p,W_p,vp,len(pos_cg),scalars = pmass)
    
    print('Compute mass density')
    rho = cg.Compute_CG_scalar(visibility_p,W_p,pmass,len(pos_cg))
    
    print('Compute flucturation velocity')
    u=p
    u[:,0] /= rho[:,0]
    u[:,1] /= rho[:,0]
    u[:,2] /= rho[:,0]
    diff = cg.Compute_diff_cg_pts(visibility_p,u,vp)
    
    print('Compute kinematic stress tensor')
    sig_kin = cg.Compute_CG_tensor(visibility_p,W_p,diff,diff,len(pos_cg))


    print('Compute contact stress tensor')
    sig_con = cg.Compute_CG_tensor(visibility_c,W_c,F_ptc,N_ptc,len(pos_cg))
    
    
    print('Average phi: ', np.average(rho[:,0]/2700))
