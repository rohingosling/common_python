# libray imports

import winsound
import numpy as np


# application imports.

from data_generator     import generate_data_gausian_archimedean_spiral
from data_visualization import plot_data

# Function: Main program function.    
    
def main ():
    
    # Initialize program.
        
    winsound.Beep ( 8000, 80 )
    
    # Configure data parameters.
    
    xd           = 0.5
    yd           = 0.5 
    turn_count   = 1.0
    sigma        = 0.05   
    spiral_count = 2   
    data_count   = 1000 * 2
    
    # Generate data.    
    
    data = []        
    data = generate_data_gausian_archimedean_spiral ( data, xd, yd, sigma, turn_count, spiral_count, data_count )
    
    # Plot data
    
    np_data = np.array ( data )
    plot_data ( np_data, markersize = 1, alpha = 0.5 )
    
    # Print program data.
    
    print ( 'data_count = ' + str ( len(data) ) )
    winsound.Beep ( 6000, 80 )
    winsound.Beep ( 6000, 80 )
    
# Program entry point.    
    
main ()