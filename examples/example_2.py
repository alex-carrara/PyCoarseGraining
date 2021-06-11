import numpy as np
import pyvista as pv
from pyevtk.hl import gridToVTK
import os
import sys
sys.path.append('..') # add the path to the folder containing CGfunctions.py.
import CGfunctions as cg ## import the CGfunctions precompiled here

#############################################
# user defined functions :
#############################################
def readfiles(rigidfile): # read my input files
    rigid = pv.read(rigidfile) # load particles data
    loc = rigid.points # read the position of the center of mass of the particles
    vp = rigid.get_array('Velocy') # read the particles volocity vector
    npart=rigid.n_points-2 # compute number of particles (-2 because of the two walls in my simulations
    return loc,vp,npart


def gen_CG_grid(Xmin,Xmax,Ymin,Ymax,Zmin,Zmax,nx,ny,nz): # generate a regular grid
    x=np.linspace(Xmin,Xmax,nx) #X coordinates
    y=np.linspace(Ymin,Ymax,ny) #Y coordinates
    z=np.linspace(Zmin,Zmax,nz) #Z coordinates
    xx,yy,zz = np.meshgrid(x,y,z)
    return xx,yy,zz


##############################################
# Main :
##############################################


mass = 0.002*(0.0005**2)*2700 #all the particles have the same mass and volume
nx=10 #nb of  CG cell along X
ny=10 #nb of CG cell along Y
nz=10 #nb og CG cell along 2
w = 0.00100 # size of the CG sphere
nstep = 10 # number of time step to be accounted for
npart = 1000 # number of particles

# create ouput dir if doesn't exist
if not os.path.isdir('./output_example2'):
    os.mkdir('./output_example2')


for i in range(nstep):
    print('########### step: ', i+1 , ' ###########', sep =None)
    nf = i+1
    print('Read files')
    loc,vp,npart=readfiles("DISPLAY/rigids_%s.vtu" % nf)
    
    
    # Generate the CG grid a function of the height of the domain
    DZ = loc[1,2]/nz # read upper plate height
    xx,yy,zz = gen_CG_grid(0.0005,0.0095,0.0005,0.0095,DZ/2,loc[1,2]-DZ/2,nx,ny,nz) #generate the CG grid (cell centers in the ouputs)
    xxp,yyp,zzp = gen_CG_grid(0.00,0.01,0.00,0.01,0.00,loc[1,2],nx+1,ny+1,nz+1) #generate the cell corners (points in the ouputs)
    x_cg = np.reshape(xx,(nx*ny*nz)) 
    y_cg = np.reshape(yy,(nx*ny*nz))
    z_cg = np.reshape(zz,(nx*ny*nz))
    pos_cg=np.transpose([x_cg,y_cg,z_cg]) # reshape to forme a (:,3) vector
    
    
    ### Copying the particle bed around the intial one to avoid issues near the boundaries
    loc_p = loc[2:npart+2,:] #suppr top and bottom plates
    loc_cp = np.zeros((len(loc_p)*27,3))
    loc_cp[:len(loc_p),:] = loc_p[:len(loc_p),:]
    loc_cp[len(loc_p):2*len(loc_p),:] = loc_p[:len(loc_p),:] + (np.zeros((len(loc_p),3))+[0.01,0,0])#copy +X
    loc_cp[2*len(loc_p):3*len(loc_p),:] = loc_p[:len(loc_p),:] + (np.zeros((len(loc_p),3))+[-0.01,0,0])#copy -X
    loc_cp[3000:4000,:] = loc_p[:1000,:] + (np.zeros((len(loc_p),3))+[0.0,0.01,0])#copy +y
    loc_cp[4000:5000,:] = loc_p[:1000,:] + (np.zeros((len(loc_p),3))+[0,-0.01,0])#copy -y
    loc_cp[5000:6000,:] = loc_p[:1000,:] + (np.zeros((len(loc_p),3))+[0.01,0.01,0])#copy +X+y
    loc_cp[6000:7000,:] = loc_p[:1000,:] + (np.zeros((len(loc_p),3))+[0.01,-0.01,0])#copy +X-y
    loc_cp[7000:8000,:] = loc_p[:1000,:] + (np.zeros((len(loc_p),3))+[-0.01,0.01,0])#copy -X+y
    loc_cp[8000:9000,:] = loc_p[:1000,:] + (np.zeros((len(loc_p),3))+[-0.01,-0.01,0])#copy -X-y
    # copy the same layer above (fct of the height of the top plate and bellow 
    loc_cp[9000:18000,:] = loc_cp[:9000,:] + (np.zeros((len(loc_p)*9,3))+[0,0,loc[1,2]])#copy -X-y
    loc_cp[18000:27000,:] = loc_cp[:9000,:] + (np.zeros((len(loc_p)*9,3))+[0,0,-loc[1,2]])#copy -X-y
    
    #same for particles velocities:
    vp1 = vp
    vp = vp[2:npart+2,:]
    vp_cp = np.zeros((len(loc_p)*27,3))
    vp_cp[:1000,:] = vp[:1000,:]
    vp_cp[1000:2000,:] = vp[:1000,:]
    vp_cp[2000:3000,:] = vp[:1000,:]
    vp_cp[3000:4000,:] = vp[:1000,:]
    vp_cp[4000:5000,:] = vp[:1000,:]
    vp_cp[5000:6000,:] = vp[:1000,:]
    vp_cp[6000:7000,:] = vp[:1000,:]
    vp_cp[7000:8000,:] = vp[:1000,:]
    vp_cp[8000:9000,:] = vp[:1000,:]
    
    vp_cp[9000:18000,:] = vp_cp[:9000,:]
    vp_cp[9000:18000,:] = vp1[1,:]
    vp_cp[18000:27000,:] = vp_cp[:9000,:]
    vp_cp[18000:27000,:] = vp1[0,:]

    
    print('Compute visibility table and CG weigths')
    visibility_p,W_p = cg.Compute_CG_visibility_weight(pos_cg,loc_cp,3*w,w,'Lucy')
    
    pmass = np.zeros(len(vp_cp))+mass  # create the particle mass vector  
    print('Compute momentum')
    rho = cg.Compute_CG_scalar(visibility_p,W_p,pmass,len(pos_cg)) #compute local density
    phi = rho[:,0]/2700 # get phi
    
    p = cg.Compute_CG_vector(visibility_p,W_p,vp_cp,len(pos_cg),pmass) #compute momentum
    
    print('Compute velocity')
    #get velocity
    ux = p[:,0]/rho[:,0]
    uy = p[:,1]/rho[:,0]
    uz = p[:,2]/rho[:,0]
    
    # reshape the outputs to form a grid
    phi = np.reshape(phi,(nx,ny,nz))
    ux = np.reshape(ux,(nx,ny,nz))
    uy = np.reshape(uy,(nx,ny,nz))
    uz = np.reshape(uz,(nx,ny,nz))
    
    print('Compute fluctuation velocity')
    # Measure of the fluctuation velocity:
    u=p
    u[:,0] /= rho[:,0]
    u[:,1] /= rho[:,0]
    u[:,2] /= rho[:,0]
    
    # compute flucutation velocity:
    diff = cg.Compute_diff_cg_pts(visibility_p,u,vp_cp)
    
    print('Compute kinetic stress tensor')
    #compute 
    sig_kin = cg.Compute_CG_tensor(visibility_p,W_p,diff,diff,len(pos_cg))
    
    print('Compute granular temperature')
    Tg = np.trace(sig_kin,axis1=1,axis2=2)/(3*rho[:,0]) # get the granular temperature
    Tg = np.reshape(Tg,(nx,ny,nz))  # reshape on a 3D grid
    
    
    print('Save CG cell data')
    gridToVTK("./output_example2/CG_%s" %i, xxp, yyp, zzp, cellData = {"phi" : phi, "U" : (ux,uy,uz), "Tg" : Tg }) #save data in a vtk file
    
