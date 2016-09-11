import matplotlib.pyplot as plt
import numpy             as np
import random            as rnd
import math
import pylab

# Function: Generate test data.

def generate_data_sinusoid  ( data, n ):
    
    # local constants
    
    PI = math.pi
    
    # Data paramters.

    f = 1.0
    a = 0.5
    p = 0.0    # 3 * PI / 2
    
    # Initialize loop parameters.
    
    x_min  = 0.0
    x_max  = 1.0
    x_step = x_max / ( n - 1 )
    
    # generate data    
    
    for x in np.arange ( x_min, x_max + x_step, x_step ):
        
        y = ( a / 2.0 ) * math.sin ( ( 2*PI * x * f ) + p ) + 0.5
        
        data.append ( [ x, y ] )
    
    return data
    
# Plot data.
    
def plot_data ( data ):
    
    # Constants

    X = 0
    Y = 1
    
    # Local variables.
    
    font_size_label = 16
    font_size_tick  = 8 
    
    xy_unit = 0.25    
    
    x_min  = 0.0
    x_max  = 1.0
    x_unit = xy_unit    
    
    y_min  = 0.0    
    y_max  = 1.0    
    y_unit = xy_unit
    
    x_label = r'$x$'
    y_label = r'$y$' + ' '*4
    
    line_style   = '-'    
    marker_style = '.'    
    plot_color   = 'k'
    plot_style   = plot_color + marker_style + line_style
    
    # Initialize axis ranges.
    
    x_range = np.arange ( x_min, x_max + x_unit, x_unit )
    y_range = np.arange ( y_min, y_max + y_unit, y_unit )        
    
    # Configure figure.
    
    plt.plot( data[:,X], data[:,Y], plot_style )

    plt.xticks ( x_range, fontsize = font_size_tick )
    plt.yticks ( y_range, fontsize = font_size_tick )
    
    pylab.ylim ( [ x_min, x_max ] )
    pylab.xlim ( [ y_min, y_max ] )

    plt.xlabel ( x_label, fontsize = font_size_label )
    plt.ylabel ( y_label, fontsize = font_size_label, rotation = 0 )
    
    plt.grid ( True )
    plt.axes().set_aspect ( 'equal' )

    plt.show()
    
# Function: Main program function.    
    
def main ():
    
    # Generate data.    
    
    data = []        
    data = generate_data_sinusoid ( data, 9 )
    data = np.array ( data )
    
    # Plot data
    
    plot_data ( data )
    
    # Test
    
    print ( 'Length of data = ' + str ( data[:,0].size ) ) 
    
# Program entry point.    
    
main ()