import math
import numpy as np
from numba import njit


@njit
def Compute_Dis_Weigth(dist,w,method,integ_br_vec='False',dist2=None,dist3=None):
    """Compute the weight of one discrete pts for one couple CG points - particle center of mass as function of the method requested by the user
    dist = distance between the CG point and particles. (scalar)
    w = size of the CG space (scalar)
    method = CG function to be used (string)"""
    ### AC - TO DO: -Compute the spacial intergral of W for the guassian function too
    if(method == 'Gaussian'):
        c = 3*w
        if dist<c:
            Vw = 2 * math.sqrt(2) * w**3 * math.pi**(3/2) * math.erf((c*math.sqrt(2))/(2*w)) - 4 * c * w**2 * math.pi * math.exp(-(c**2/(2*w**2)))
            W = (1/Vw) * math.exp(-((dist**2))/(2*w**2))
        else:
            W = 0
        return W
    if(method == 'Lucy'):
        c = 3*w
        if dist<c:
            W = 105/(16*math.pi*c**3) * (-3*(dist/c)**4+8*(dist/c)**3-6*(dist/c)**2+1)
        else:
            W = 0
        if(integ_br_vec == 'True'):
            # compute the integral of W along a vector here (integral along "s" between 0 and 1):
            if dist3 is not None:
                intW1 = ComputeLucyIntegral(c,dist1,dist2)
                intW2 = ComputeLucyIntegral(c,dist1,dist3)
            else: 
                intW1 = ComputeLucyIntegral(c,dist1,dist2)
                intW2 = None
            return W, intW1, intW2
        else:
            return W



@njit
def ComputeLucyIntegral(c,dist1,dist2):
    CA = -315/(16*math.pi*c**7)
    CB = 840/(16*math.pi*c**6)
    CC = -630/(16*math.pi*c**5)
    CD = CA*dist2**4
    CE = 4*CA*dist*dist2**3 + CB*dist2**3
    CF = 6*CA*dist**2*dist2**2 + 3*CB*dist*dist2**2 + CC*dist2**2
    CG = 4*CA*dist**3*dist2 + 3*CB*dist**2*dist2 + 2*CC*dist*dist2
    CH = CA*dist**4 + CB*dist**3+CC*dist**2+1
    return CD/5+CE/4+CF/3+CG/2+CH


def Compute_CG_visibility_weight(CG_pts,Dis_pts,cutoff,w,method,integ_br_vec='False',dist2=None,dist3=None):
    """Create the visibility and W arrays as a function of the CG cutoff distance and compute the associated CG weigth
    CG_pts = CG points (array(nb_CG_pts,3))
    Dis_pts = Discrete points (array( nb_dis_pts,3))
    cutoff = if the distance between the CG pts and discrete pts is longer than cutoff, the pair is not included in the visibility table (scalar)
    method = currently supported: Gaussian and Lucy (only Lucy for the integral along s
    w = size of the CG window
    integ_br_vec = switch to true to compute the integral of W along vector2"""
    if(integ_br_vec == 'True'):
        if dist3 is not None:
            visibility, W, intW1, intW2 = Compute_CG_visibility_weight_lst(CG_pts,Dis_pts,cutoff,w,method,integ_br_vec,dist2,dist3)
            visibility = np.array(visibility)
            W = np.array(W)
            intW1 = np.array(intW1)
            intW2 = np.array(intW2)
            return visibility, W, intW1, intW2
        else:
            visibility, W,  intW1, intW2 = Compute_CG_visibility_weight_lst(CG_pts,Dis_pts,cutoff,w,method,integ_br_vec,dist2)
            visibility = np.array(visibility)
            W = np.array(W)
            intW1 = np.array(intW1)
            return visibility, W, intW1
    else:
        visibility, W = Compute_CG_visibility_weight_lst(CG_pts,Dis_pts,cutoff,w,method)
        visibility = np.array(visibility)
        W = np.array(W)
        return visibility, W



@njit
def Compute_CG_visibility_weight_lst(CG_pts,Dis_pts,cutoff,w,method,integ_br_vec='False',dist2=None):
    """generate the visibility and W lists"""
    ### AC: This one can be long to compute for large datasets but reduces the computational cost later. In the current implementation, this function is supported by njit but should not be run in paralel 
    ### this fuction return two list that has to be converted to array afterwhat. The convertion from list to array is not yet supported by numba
    visibility = []
    W = []
    if(integ_br_vec == 'True'):
        intW = []
        for i in range(len(CG_pts[:,0])):
            for j in range(len(Dis_pts[:,0])):
                dist = np.linalg.norm(CG_pts[i,:]-Dis_pts[j,:])
                if dist<cutoff:
                    visibility.append(np.array([i,j]))
                    Wb,intWb = Compute_Dis_Weigth(dist,w,method,method,integ_br_vec,dist2)
                    W.append(Wb)
                    intW.append(intW)
        return visibility, W, intW
    else:
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



