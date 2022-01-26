import math
import numpy as np
from numba import njit


################################################################################################################################################
#                                                                                                                                              #
#                                                            Top level functions                                                               #
#                                                                                                                                              #
################################################################################################################################################

def Compute_CG_visibility_weight(CG_pts,Dis_pts,cutoff,w,method,int_vec=False,vec=None):
    """Create the visibility and W arrays as a function of the CG cutoff distance and compute the associated CG weigth
    CG_pts: cooridiantes of the CG points for interpolation (number of CG points,3)
    Dis_pts: coordinates of the discrete point (for example particle centers or contact points (number of discrete points,3)
    cutoff: maximum distance between the CG and discrete points accounted for (scalar)
    w: diameter of the CG interpolation sphere (scalar)
    method: Can be either Lucy or Gaussian for points. (string)
    int_vec: Boolean to integrate along a vector. If True, W is integrated along vec from the disrete points (boolean)
    vec: vector along which W is integrated (number of discrete points, 3)
    Output:
    visibility = list of visibility between CG nodes and discrte pts (nb couples, 2)
    W = interpolation weight of the CG-discrete particle couples (nb of CG discrete points couples)"""
    # AC this function is necessary since numba doesn't manage the convertion from list to numpy array yet.
    if vec is None:
        vec=np.zeros((len(Dis_pts),3)) #requiered if vec undefined for numba
    visibility, W = Compute_CG_visibility_weight_lst(CG_pts,Dis_pts,cutoff,w,method,int_vec,vec)
    visibility = np.array(visibility)
    W = np.array(W)
    return visibility, W


@njit
def Compute_CG_tensor(visibility,W,Vec1,Vec2,n_cg_pts,scalars = None):
    """ Sum a tensor at the CG points:
    visibility = array containing the CG nodes and particles indexes to be accounted sig_or_force (nb of CG-discrete couples, 2)
    W =  CG weight of the CG point particle pair (nb of CG-discrete couples)
    Vec1 = First vector to be accounted for in the outer product. Can be either (nb CG pts,3) or (nb of CG-discrete couples,3)
    Vec2 = Second vector to be accounted for in the outer product (len(vec1),3)
    n_cg_pys = Number of CG nodes (scalar)
    scalars = Scalar use in the computation of the tensor (mass or volume of the particles). If = None or unspecified: not accounted for. (len(vec1),3)
    Output:
    tens = sum of the tensor at the CG points"""
    tens=np.zeros((n_cg_pts,len(Vec1[0,:]),len(Vec1[0,:])))
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
    visibility = array containing the CG nodes and particles indexes to be accounted sig_or_force (nb of CG-discrete couples, 2)
    W =  CG weight of the CG point particle pair (nb of CG-discrete couples)
    Vec1 = First vector to be accounted for in the outer product. Can be either (nb CG pts,3) or (nb of CG-discrete couples,3)
    n_cg_pys = Number of CG nodes (scalar)
    scalars = Scalar use in the computation of the vector (mass or volume of the particles). If = None or unspecified: not accounted for. (len(vec1),3)
    Output:
    vec = sum of the vector at the CG points"""
    vec=np.zeros((n_cg_pts,len(Vec1[0,:])))
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
    visibility = array containing the CG nodes and particles indexes to be accounted sig_or_force (nb of CG-discrete couples, 2)
    W =  CG weight of the CG point particle pair (nb of CG-discrete couples)
    scalars = Scalar use in the computation. Can be either (nb CG pts,3) or (nb of CG-discrete couples,3)
    n_cg_pys = Number of CG nodes (scalar)
    Output:
    scl = sum of the scalar at the CG points"""
    scl=np.zeros((n_cg_pts,1))
    for i in range(len(visibility)):
        if len(scalars)==len(visibility):
            scl[visibility[i,0]] += scalars[i]*W[i]
        else:
            scl[visibility[i,0]] += scalars[visibility[i,1]]*W[i]
    return scl


@njit
def Compute_diff_cg_pts(visibility,val_cg,val_pts):
    """Compute the difference between between an array at the discrete points (either scalar, vector, tenor) and the value at the associated CG point
    visibility = containing the CG nodes and particles indexes to be accounted (nb of CG-discrete couples, 2)
    val_cg = array at the CG nodes. Either (nb of CG pts), (nb of CG pts,3), or (nb of CG pts,(3,3))
    val_pts = array at the particles mass center. Either (nb of CG pts), (nb of CG pts,3), or (nb of CG pts,(3,3)) [has to have the same dimensions as val_cg.
    Output:
    diff = difference between the values at the discrete point and at the associated CG points. Either (nb of CG pts), (nb of CG pts,3), or (nb of CG pts,(3,3))"""
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




################################################################################################################################################
#                                                                                                                                              #
#                                                           lower level functions                                                              #
#                                                                                                                                              #
################################################################################################################################################

@njit
def Compute_CG_visibility_weight_lst(CG_pts,Dis_pts,cutoff,w,method,int_vec,vec):
    """generate the visibility and W lists
        CG_pts: cooridiantes of the CG points for interpolation (number of CG points,3)
        Dis_pts: coordinates of the discrete point (for example particle centers or contact points (number of discrete points,3)
        cutoff: maximum distance between the CG and discrete points accounted for (scalar)
        w: diameter of the CG interpolation sphere (scalar)
        method: Can be either Lucy or Gaussian for points. (string)
        int_vec: Boolean to integrate along a vector. If True, W is integrated along vec from the disrete points (boolean)
        vec: vector along which W is integrated (number of discrete points, 3)
        """
    ### AC: This one can be long to compute for large datasets but reduces the computational cost later. In the current implementation, this function is supported by njit but should not be run in parallel.
    visibility = [] #init list
    W = [] #initi list
    for i in range(len(CG_pts[:,0])):
        for j in range(len(Dis_pts[:,0])):
            dist = np.linalg.norm(CG_pts[i,:]-Dis_pts[j,:])
            if dist<cutoff:
                visibility.append(np.array([i,j]))
                if(int_vec == True):
                    Wb = Compute_Dis_Weigth(CG_pts[i,:]-Dis_pts[j,:],w,method,cutoff,int_vec,vec[j,:])
                else:
                    Wb = Compute_Dis_Weigth(CG_pts[i,:]-Dis_pts[j,:],w,method,cutoff,int_vec,vec[j,:])
                W.append(Wb)
    return visibility, W


@njit
def Compute_Dis_Weigth(vec1,w,method,c,int_vec,vec2):
    """Compute the weight of one discrete pts for one couple CG points - particle center of mass as function of the method requested by the user
    dist = distance between the CG point and particles. (scalar)
    w = size of the CG space (scalar)
    method = CG function to be used (string)
    vec1 = vector between the CG point and the discrete point (vector3)
    vec2 = branch vector along which the integral of W is computed (vector3)
    c= cuttoff distance (scalar)
    int_vec = boolean indicating if the integral of W have to be computed along vec2 (boolean)
    Output:
    W = CG interpolation weight"""
    if(int_vec=='True'):
        W = ComputeIntegralW(c,vec1,vec2,method,w)
    else:
        dist = np.linalg.norm(vec1)
        if(method == 'Lucy'):
            W = ComputeLucyWeight(c,dist)
        else:
            W = ComputeGaussianWeight(c,dist,w)
    return W


@njit
def ComputeIntegralW(c,vec1,vec2,method,w):
    """Compute the integral of the lucy function along (vec1+s*vec2) ds
    c = cutoff distance (scalar)
    vec1 = vector between the coarse grainig point and the initial point of the vectoe along which the integral of W is computed (vector)
    vec2 = vector between the initial and final points along which the integral of W is computed (vector)
    method = interpolation function ('Lucy' or 'Gaussian') (string)
    w = width of the interpolation window (for Lucy, w=c/2) (scalar)
    Output:
    Wint = integral of W along s
    """
    # AC: Currently numerical estimation using trapezoidal numerical integration. Check for analytical solutions
    s = np.arange(0,1.1,0.1)
    Wint = 0
    for i in range(len(s)-1):
        vecA = vec1 + s[i]*vec2
        vecB = vec1 + s[i+1]*vec2
        distA = np.linalg.norm(vecA)
        distB = np.linalg.norm(vecB)
        if method=='Lucy':
            W_vecA = ComputeLucyWeight(c,distA)
            W_vecB = ComputeLucyWeight(c,distB)
        else:
            W_vecA = ComputeGaussianWeight(c,distA,w)
            W_vecB = ComputeGaussianWeight(c,distB,w)
        Wint += min([W_vecA,W_vecB])*0.1+(max([W_vecA,W_vecB])-min([W_vecA,W_vecB]))*0.05
    return Wint

   
@njit
def ComputeLucyWeight(c,dist):
    """Compute the CG weight using a Lucy function
    c = cuteoff distance (scalar)
    dist = distance between the CG and discrete points (scalar)
    Output:
    W = interpolation weight
    """
    W = 105/(16*math.pi*c**3) * (-3*(dist/c)**4+8*(dist/c)**3-6*(dist/c)**2+1)
    return W


@njit
def ComputeGaussianWeight(c,dist,w):
    """Compute the CG weight using a Gaussian function
    c = cuteoff distance (scalar)
    dist = distance between the CG and discrete points (scalar)
    w = interpolation wondow width
    Output:
    W = interpolation weight
    """
    Vw = 2 * math.sqrt(2) * w**3 * math.pi**(3/2) * math.erf((c*math.sqrt(2))/(2*w)) - 4 * c * w**2 * math.pi * math.exp(-(c**2/(2*w**2)))
    W = (1/Vw) * math.exp(-((dist**2))/(2*w**2))
    return W











