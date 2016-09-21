# libray imports

import matplotlib.pyplot as plt
import numpy             as np
import pylab

# Plot data.
    
def plot_data ( data, markersize = 2, alpha = 0.5 ):
    
    # Constants

    X = 0
    Y = 1
    
    # Local variables.
    
    font_size_label = 14
    font_size_tick  = 8
    
    xy_unit = 0.25    
    
    x_min  = 0.0
    x_max  = 1.0
    x_unit = xy_unit    
    
    y_min  = 0.0    
    y_max  = 1.0    
    y_unit = xy_unit
    
    x_label = r'$x_0$'
    y_label = r'$x_1$' + ' '*4
    
    line_style   = ''    
    marker_style = '.'    
    plot_color   = 'k'
    plot_style   = plot_color + marker_style + line_style
    
    # Initialize axis ranges.
    
    x_range = np.arange ( x_min, x_max + x_unit, x_unit )
    y_range = np.arange ( y_min, y_max + y_unit, y_unit )        
    
    # Configure figure.
    
    plt.plot( data[:,X], data[:,Y], plot_style, markersize = markersize, alpha = alpha )

    plt.xticks ( x_range, fontsize = font_size_tick )
    plt.yticks ( y_range, fontsize = font_size_tick )
    
    pylab.ylim ( [ x_min, x_max ] )
    pylab.xlim ( [ y_min, y_max ] )

    plt.xlabel ( x_label, fontsize = font_size_label )
    plt.ylabel ( y_label, fontsize = font_size_label, rotation = 0 )
    
    plt.grid ( True )
    plt.axes().set_aspect ( 'equal' )

    plt.show()    
