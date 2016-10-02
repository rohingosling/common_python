# libray imports

import winsound

import numpy as np

from sklearn import manifold


# application imports.

from data_generator     import generate_data_gausian_archimedean_spiral
from data_visualization import plot_data

def sound ( f, p, n  ):
    for i in range ( n ):
        winsound.Beep ( f, p )

# Function: Main program function.    
    
def main ():
    
    # Initialize program.
        
    sound ( 200, 80, 2 )
    
    # Configure data parameters.
    
    xd           = 0.5
    yd           = 0.5 
    turn_count   = 2.0
    sigma        = 0.025   
    spiral_count = 2   
    data_count   = 1000 * 2
    
    # Generate data.    
    
    data = []        
    data = generate_data_gausian_archimedean_spiral ( data, xd, yd, sigma, turn_count, spiral_count, data_count )
    
    # Learn manifolds.
    
    n_neighbors  = 3
    n_components = 2
    
    manifold_transform = manifold.Isomap ( n_neighbors, n_components )
    
    x = data
    y = manifold_transform.fit_transform ( x )
    
    
    # Plot data
    
    np_data = np.array ( data )
    plot_data ( np_data, markersize = 1, alpha = 0.5, auto_limit_enabled = True )
    
    # Plot manifold transform.
    
    np_data = np.array ( y )
    plot_data ( np_data, markersize = 1, alpha = 0.5, auto_limit_enabled = False )
    
    # Print program data.
    
    print ( 'data_count = ' + str ( len(data) ) )
    sound ( 12000, 80, 2 )
    
# Program entry point.    
    
main ()