# libray imports

import matplotlib.pyplot as plt
import numpy             as np

# Plot data.
    
def plot_data ( data, markersize = 2, alpha = 0.5, auto_limit_enabled = False ):
    
    # Constants

    X = 0
    Y = 1
    
    # Local variables.
    
    font_size_label = 10
    font_size_tick  = 6
    
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
    
    plt.plot ( data[:,X], data[:,Y], plot_style, markersize = markersize, alpha = alpha )
    
    if auto_limit_enabled:        
        
        plt.xticks ( x_range, fontsize = font_size_tick )
        plt.yticks ( y_range, fontsize = font_size_tick )        
        
        plt.ylim ( [ x_min, x_max ] )
        plt.xlim ( [ y_min, y_max ] )
        
    else:
        
        #plt.xticks ( np.arange ( -2.0, 3.0, 1.0 ), fontsize = font_size_tick )
        #plt.yticks ( np.arange ( -2.0, 3.0, 1.0 ), fontsize = font_size_tick )        
        
        plt.xticks ( fontsize = font_size_tick )
        plt.yticks ( fontsize = font_size_tick )        
        
        #plt.ylim ( [ -1.0, 1.0 ] )
        #plt.xlim ( [ -3.0, 3.0 ] )

    plt.xlabel ( x_label, fontsize = font_size_label )
    plt.ylabel ( y_label, fontsize = font_size_label, rotation = 0 )
    
    plt.grid ( True )
    #plt.axes().set_aspect ( 'equal' )

    plt.show()    
