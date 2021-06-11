# PyCoarseGraining #
PyCoarseGraining is an python implementation of the coarse graining functions for DEM presented in Weinhart & al. (2012, 2013, and 2016) and Breard et al. (2020). 


# References:
Weinhart, T., Thornton, A. R., Luding, S., & Bokhove, O. (2012). From discrete particles to continuum fields near a boundary. Granular Matter, 14(2), 289-294.
Weinhart, T., Hartkamp, R., Thornton, A. R., & Luding, S. (2013). Coarse-grained local and objective continuum description of three-dimensional granular flows down an inclined surface. Physics of Fluids, 25(7), 070605.
Weinhart, T., Labra, C., Luding, S., & Ooi, J. Y. (2016). Influence of coarse-graining parameters on the analysis of DEM simulations of silo flow. Powder technology, 293, 138-148.
Breard, E. C., Dufek, J., Fullard, L., & Carrara, A. (2020). The Basal Friction Coefficient of Granular Flows With and Without Excess Pore Pressure: Implications for Pyroclastic Density Currents, Water‐Rich Debris Flows, and Rock and Submarine Avalanches. Journal of Geophysical Research: Solid Earth, 125(12), e2020JB020203.


# Examples
 - Example 1 : Computation of the local particle volume fraction.
 - Example 2 : Computation of the particles velocity mean field and granular temperature. Outputs are exported as vtk cell data.
 - Example 3 : Computation of the contact stress tensor.

 
# Architecture: 
The python function CGfunctions.py currenlty contains all the CG functions

# Dependencies:
 - Coarse graining functions: -Numpy; -Math; -Numba
 - Example1: -Numpy; -Pyvista; 
 - Example2: -Numpy; - Pyvista; -Pyevtk(https://pypi.org/project/pyevtk/);
 - Example3: -Numpy; - Pyvista;



                                  
