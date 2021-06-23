import numpy as np
import sys
sys.path.append('..') # add the path to the folder containing CGfunctions.py here
import CGfunctions as cg
import pyvista as pv

# AC: This example shows how the CGfunctions can be used to compute the integral of the the CG weight along a branch vector
# AC: The contact stress tensor at the CG points are then computed

# define some functions to read the input files and create the CG grid
def readfiles(particlefile,contactfile): # read my input files
    part = pv.read(particlefile) # load particle properties at center of mass
    cont = pv.read(contactfile) # load branch vector informations
    vp = part.get_array('Velocy') # read the particles velocity vector
    ppos = part.points # read location of the center of masses of the particles
    bvpos = cont.points # read location limits of each branch vector
    cforce = cont.get_array('Fx') # read the branch vector forces
    connectivity = np.reshape(cont.cells,(len(cforce),3)) # read the connectivity of the branch vectors
    return ppos, bvpos, vp, cforce, connectivity


def gen_CG_grid(Xmin,Xmax,Ymin,Ymax,Zmin,Zmax,nx,ny,nz): # generate a regular grid
    x=np.linspace(Xmin,Xmax,nx) #X coordinates
    y=np.linspace(Ymin,Ymax,ny) #Y coordinates
    z=np.linspace(Zmin,Zmax,nz) #Z coordinates
    xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')
    return xx,yy,zz

#########################################################################################################################
#########################################################################################################################

# parameters
nx=9 #nb of  CG cell along X
ny=9 #nb of CG cell along Y
nz=9 #nb og CG cell along 2
w = 0.001 # size of the CG sphere
nstep = 10 # number of time step to be accounted for







for i in range(nstep):
    print('########### step: ', i+1 , ' ###########', sep =None)
    nf = i+1
    print('Read files')
    # generate position of the CG points and convert it in a array (nb of CG points,3)
    xx,yy,zz = gen_CG_grid(0.001,0.009,0.001,0.009,0.001,0.009,nx,ny,nz) 
    x_cg = np.reshape(xx,(nx*ny*nz))
    y_cg = np.reshape(yy,(nx*ny*nz))
    z_cg = np.reshape(zz,(nx*ny*nz))
    pos_cg=np.transpose([x_cg,y_cg,z_cg])

    # read the particle and branch vector data
    ppos, bvpos, vp, cforce, connectivity = readfiles("DISPLAY/rigids_%s.vtu" % nf,"DISPLAY/bv.%s.vtu" % nf)

    #contact points artificially created at the middle of the branch vectors here:
    fake_cpts = np.zeros((len(connectivity),3))
    cforce_bis = np.zeros((len(cforce)*2,3))
    bv_bis = np.zeros((len(cforce)*2,2))
    bv_xyz = np.zeros((len(cforce)*2,3))


    # separate the branch vector in two: one from the contact point to the center of mass of the 1 particle involved in the contact and another one between the contact point and the mass center of the second particles
    for i in range(len(fake_cpts)):
        fake_cpts[i,:] = (bvpos[connectivity[i,1],:]+bvpos[connectivity[i,2],:])/2 #coordinates of the contact points
        cforce_bis[i*2,:] = cforce[i] #contact force particle 1
        cforce_bis[i*2+1,:] = -cforce[i] #contact force particle 2
        bv_bis[i*2,0] = i
        bv_bis[i*2,1] = connectivity[i,1]
        bv_bis[i*2+1,0] = i
        bv_bis[i*2+1,1] = connectivity[i,2]
        bv_xyz[i*2,:] = bvpos[connectivity[i,1],:]-fake_cpts[i,:] #coordinates of the vector between the contact point and 1st particle mass center
        bv_xyz[i*2+1,:] = bvpos[connectivity[i,2],:]-fake_cpts[i,:] #coordinates of the vector between the contact point and 2nd particle mass center


    visibility , W = cg.Compute_CG_visibility_weight(pos_cg,fake_cpts,2*w,w,'Lucy',int_vec=True,vec=bv_xyz)


    # compute the normal of the branch vectors
    norm_bv = np.zeros((len(bv_xyz),3))
    for i in range(len(bv_xyz)):
        norm_bv[i] = bv_xyz[i]/np.linalg.norm(bv_xyz[i])


    # compute the contact stress tensor at the CG points
    sig_c = cg.Compute_CG_tensor(visibility,W,cforce_bis,norm_bv,nx*ny*nz)
    print(sig_c)
