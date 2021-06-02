import math
import numpy as np
from numba import njit


@njit
def Compute_Dis_Weigth(dist,w,method):
    """Compute the weight of one discrete pts for one couple CG points - particle center of mass as function of the method requested by the user
    dist = distance between the CG point and particles. (scalar)
    w = size of the CG space (scalar)
    method = CG function to be used (string)"""
    ### AC - TO DO: -Compute the spacial intergral of W here too
    if(method == 'Gaussian'):
        c = 3*w
        if dist<c: #AC this test is not usefule since this function is not called if dist<c 
            Vw = 2 * math.sqrt(2) * w**3 * math.pi**(3/2) * math.erf((c*math.sqrt(2))/(2*w)) - 4 * c * w**2 * math.pi * math.exp(-(c**2/(2*w**2)))
            W = (1/Vw) * math.exp(-((dist**2))/(2*w**2))
        else:
            W = 0
    if(method == 'Lucy'):
        c = 3*w
        if dist<c:
            W = 105/(16*math.pi*c**3) * (-3*(dist/c)**4+8*(dist/c)**3-6*(dist/c)**2+1)
        else:
            W = 0
    return W



def Compute_CG_visibility_weight(CG_pts,Dis_pts,cutoff,w,method):
    """Create the visibility and W arrays as a function of the CG cutoff distance and compute the associated CG weigth
    CG_pts = CG points (array(nb_CG_pts,3))
    Dis_pts = Discrete points (array( nb_dis_pts,3))
    cutoff = if the distance between the CG pts and discrete pts is longer than cutoff, the pair is not included in the visibility table (scalar)"""
    visibility, W = Compute_CG_visibility_weight_lst(CG_pts,Dis_pts,cutoff,w,method)
    visibility = np.array(visibility)
    W = np.array(W)
    return visibility, W



@njit
def Compute_CG_visibility_weight_lst(CG_pts,Dis_pts,cutoff,w,method):
    """generate the visibility and W lists"""
    ### AC: This one can be long to compute for large datasets but reduces the computational cost later. In the current implementation, this function is supported by njit but should not be run in paralel. Need another startegy to run in parallel
    ### Key function to run fast!!!
    ### this fuction return two list that has to be converted to array afterwhat. The convertion from list to array is not yet supported by numba
    visibility = []
    W = []
    for i in range(len(CG_pts[:,0])):
        for j in range(len(Dis_pts[:,0])):
            dist = np.linalg.norm(CG_pts[i,:]-Dis_pts[j,:])
            if dist<cutoff:
                visibility.append(np.array([i,j]))
                W.append([Compute_Dis_Weigth(dist,w,method)])
    return visibility, W



@njit
def Compute_CG_tensor(visibility,W,Vec1,Vec2,n_cg_pts,scalars = None):
    """ Sum a tensor at the CG points:
    visibility = array(:,2) containing the CG nodes and particles indexes to be accounted sig_or_force
    W = array(len(visibility),1) CG weight of the CG point particle pair
    Vec1 = can be either array(n_cg_pts,3) or array(len(visibility,3). First vector to be accounted for in the outer product
    Vec2 = array(len(Vec1),3). Second vector to be accounted for in the outer product
    n_cg_pys = scalar. Number of CG nodes
    scalars = array(len(Vec1),1). scalar use in the computation of the tensor (mass or volume of the particles). If = None or unspecified: not accounted for"""
    tens=np.zeros((n_cg_pts,3,3))
    for i in range(len(visibility)):
        if len(Vec1)==len(visibility):
            if scalars is None:
                tens[visibility[i,0]] += np.outer(Vec1[i],Vec2[i])*W[i]
            else:
                tens[visibility[i,0]] += np.outer(Vec1[i],Vec2[i])*W[i]*scalars[i]
        else:
            if scalars is None:
                tens[visibility[i,0]] += np.outer(Vec1[visibility[i,1]],Vec2[visibility[i,1]])*W[i]
            else:
                tens[visibility[i,0]] += np.outer(Vec1[visibility[i,1]],Vec2[visibility[i,1]])*W[i]*scalars[visibility[i,1]]
    return tens


@njit
def Compute_CG_vector(visibility,W,Vec1,n_cg_pts,scalars = None):
    """ Sum a vector at the CG points:
    visibility = array(:,2) containing the CG nodes and particles indexes to be accounted sig_or_force
    W = array(len(visibility),1) CG weight of the CG point particle pair
    Vec1 = can be either array(n_cg_pts,3) or array(len(visibility,3). Vector to be accounted for in the outer product
    n_cg_pys = scalar. Number of CG nodes
    scalars = array(len(Vec1),1). scalar use in the computation of the tensor (mass or volume of the particles). If = None or unspecified: not accounted for"""
    vec=np.zeros((n_cg_pts,3))
    for i in range(len(visibility)):
        if len(Vec1)==len(visibility):
            if scalars is None:
                vec[visibility[i,0]] += Vec1[i]*W[i]
            else:
                vec[visibility[i,0]] += Vec1[i]*W[i]*scalars[i]
        else:
            if scalars is None:
                vec[visibility[i,0]] += Vec1[visibility[i,1]]*W[i]
            else:
                vec[visibility[i,0]] += Vec1[visibility[i,1]]*W[i]*scalars[visibility[i,1]]
    return vec

        
@njit
def Compute_CG_scalar(visibility,W,scalars,n_cg_pts):
    """ Sum a scalar at the CG points:
    visibility = array(:,2) containing the CG nodes and particles indexes to be accounted sig_or_force
    W = array(len(visibility),1) CG weight of the CG point particle pair
    scalars = array(len(Vec1),1). scalar use in the computation of the tensor (mass or volume of the particles)
    n_cg_pys = scalar. Number of CG nodes"""
    scl=np.zeros((n_cg_pts,1))
    for i in range(len(visibility)):
        if len(scalars)==len(visibility):
            scl[visibility[i,0]] += scalars[i]*W[i]
        else:
            scl[visibility[i,0]] += scalars[visibility[i,1]]*W[i]
    return scl


@njit
def Compute_diff_cg_pts(visibility,val_cg,val_pts):
    """Compute the difference between two arrays from the CG-node particle pair
    visibility = array(:,2) containing the CG nodes and particles indexes to be accounted sig_or_force
    val_cg = array at the CG nodes
    val_pts = array at the particles mass center"""
    if val_cg.ndim == 2:
        if val_cg.shape[1] == 1: #scalar
            diff = np.zeros((len(visibility),1))
            for i in range(len(visibility)):
                diff[i] = val_cg[visibility[i,0],0]-val_pts[visibility[i,1],0]
        elif val_cg.shape[1] == 3: #vector
            diff = np.zeros((len(visibility),3))
            for i in range(len(visibility)):
                diff[i] = val_cg[visibility[i,0]]-val_pts[visibility[i,1]]
    else: # tensor
        diff = np.zeros((len(visibility),3,3))
        for i in range(len(visibility)):
            diff[i] = val_cg[visibility[i,0]]-val_pts[visibility[i,1]]
    return diff



