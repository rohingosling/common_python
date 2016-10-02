# libray imports

import matplotlib.pyplot as plt
import numpy             as np
from mpl_toolkits.mplot3d import Axes3D

# Patch 3D axis margins.

from mpl_toolkits.mplot3d.axis3d import Axis

if not hasattr(Axis, "_get_coord_info_old"):
    
    def _get_coord_info_new ( self, renderer ):
        
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
        
    Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info     = _get_coord_info_new

# Global Constants.

MARKER_SIZE_GLOBAL = 12
MARKER_SIZE_2D     = MARKER_SIZE_GLOBAL
MARKER_SIZE_3D     = MARKER_SIZE_GLOBAL

# Plot data 3D.

def plot_data_3d ( data, labels ):
    
    # Constants

    X = 0
    Y = 1    
    Z = 2
    
    # Local variables.
    
    auto_limit_enabled = False
    
    font_size_label = 10
    font_size_tick  = 6
    
    xyz_step = 0.5
    unit     =  1.0
    x_min    = -unit
    x_max    =  unit
    y_min    = -unit
    y_max    =  unit
    z_min    = -unit
    z_max    =  unit
    
    x_range = np.arange ( x_min, x_max + xyz_step, xyz_step )
    y_range = np.arange ( y_min, y_max + xyz_step, xyz_step )   
    z_range = np.arange ( z_min, z_max + xyz_step, xyz_step )   
    
    alpha             = 0.5
    marker_size       = MARKER_SIZE_3D
    marker_line_width = 0    
    marker_style      = '.'    
    marker_color      = [ point for point in labels ] 
    pane_color        = ( 1.0, 1.0, 1.0, 1.0 )
    
    # Configure plot.
    
    fig = plt.figure ( figsize = plt.figaspect ( 1.0 ) ) 
    ax  = fig.add_subplot ( 111, projection = '3d' )
        
    ax.scatter (
        data[:,X], data[:,Y], data[:,Z],
        marker = marker_style, 
        c      = marker_color, 
        s      = marker_size,
        lw     = marker_line_width,
        alpha  = alpha
    )   
    
    ax.set_xlabel ( r'$x = x_0$', fontsize = font_size_label )
    ax.set_ylabel ( r'$y = x_1$', fontsize = font_size_label )
    ax.set_zlabel ( r'$z = x_2$', fontsize = font_size_label )
    
    ax.set_xticks ( x_range )
    ax.set_yticks ( y_range )     
    ax.set_zticks ( z_range )     
    
    if not auto_limit_enabled:
        
        ax.set_ylim ( [ x_min, x_max ] )
        ax.set_xlim ( [ y_min, y_max ] )    
        ax.set_zlim ( [ z_min, z_max ] ) 
        
    else:
        
        s = 8.0
        
        ax.set_ylim ( [ x_min*s, x_max*s ] )
        ax.set_xlim ( [ y_min*s, y_max*s ] )    
        ax.set_zlim ( [ z_min*s, z_max*s ] ) 
    
    ax.w_xaxis.set_pane_color ( pane_color )
    ax.w_yaxis.set_pane_color ( pane_color )
    ax.w_zaxis.set_pane_color ( pane_color )
    
    ax.tick_params  ( labelsize = font_size_tick  )    
        
    # Plot data.
    
    plt.tight_layout ()
    plt.show()
    
    

# Plot data 2D.
    
def plot_data_2d ( data, labels, markersize = 2, alpha = 0.5, auto_limit_enabled = False ):
    
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
    
    x_label = r'$x = x_0$'
    y_label = r'$y = x_1$' + ' '*4
    
    alpha             = 1.0
    marker_style      = '.'
    marker_size       = MARKER_SIZE_2D    
    marker_line_width = 0
    plot_color        = [ point for point in labels ] 
    
    # Initialize axis ranges.
    
    x_range = np.arange ( x_min, x_max + x_unit, x_unit )
    y_range = np.arange ( y_min, y_max + y_unit, y_unit )        
    
    # Configure figure.
    
    fig = plt.figure ( figsize = plt.figaspect ( 1.0 ) ) 
    ax  = fig.add_subplot(111)
    
    ax.scatter (
        data[:,X], data[:,Y], 
        marker = marker_style, 
        s      = marker_size,
        c      = plot_color,
        lw     = marker_line_width,
        alpha  = alpha
    )
    
    if auto_limit_enabled:        
        
        ax.set_xlabel ( x_label, fontsize = font_size_label )
        ax.set_ylabel ( y_label, fontsize = font_size_label )        
        
        ax.set_xticks ( x_range )
        ax.set_yticks ( y_range )     
        
        ax.tick_params  ( labelsize = font_size_tick  )
        
        ax.set_ylim ( [ x_min, x_max ] )
        ax.set_xlim ( [ y_min, y_max ] )    

    else:
        
        #plt.xticks ( np.arange ( -2.0, 3.0, 1.0 ), fontsize = font_size_tick )
        #plt.yticks ( np.arange ( -2.0, 3.0, 1.0 ), fontsize = font_size_tick )        
        
        ax.tick_params  ( labelsize = font_size_tick  )      
        
        #plt.ylim ( [ -1.0, 1.0 ] )
        #plt.xlim ( [ -3.0, 3.0 ] )
    
    ax.grid ( True )
    #ax.axes().set_aspect ( 'equal' )

    plt.show()    
