# Library imports.

import matplotlib.pyplot as plt
import numpy             as np
import pylab

from sklearn.cluster       import DBSCAN
from sklearn.preprocessing import StandardScaler


# Local imports.

from gausian_cluster_generator import generate_data_gausian_archimedean_spiral
from gausian_cluster_generator import generate_data_gausian_ring_formation

    
# Generate spiral.
    
def generate_spiral ( data ):
    
    # Configure data parameters.
    
    x            = 0.5
    y            = 0.5 
    turn_count   = 2.0
    sigma        = 0.033    
    spiral_count = 2    
    data_count   = 1000 * 2
    
    # Generate data.

    data = generate_data_gausian_archimedean_spiral ( data, x, y, sigma, turn_count, spiral_count, data_count )
    
    return data
    

# Generate Rings.
    
def generate_rings ( data ):
    
    # Configure data parameters.
    
    cluster_x          = 0.5
    cluster_y          = 0.5
    cluster_radius_min = 0.25 / 2.0
    cluster_radius_max = 0.25 / 2.0 * 4.0
    cluster_sigma      = 0.015
    cluster_count      = 4
    cluster_data_count = 1000 * 5
    
    # Generate data.    

    data = generate_data_gausian_ring_formation (
               data, 
               cluster_x, 
               cluster_y,
               cluster_radius_min,
               cluster_radius_max,
               cluster_sigma,
               cluster_count,
               cluster_data_count
           )
    
    return data


# Plot data.
    
def plot_data ( data, data_labels, core_samples_mask ):
    
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
    
    marker_size       = 2
    line_style        = ''    
    marker_style      = '.'
    marker_edge_color = 'k'    
    plot_color        = 'k'
    plot_style        = plot_color + marker_style + line_style

    # Configure color range.
    
    unique_labels = set ( data_labels )
    colors        = plt.cm.Spectral ( np.linspace(0, 1, len ( unique_labels ) ) )
    
    
    # Initialize axis ranges.
    
    x_range = np.arange ( x_min, x_max + x_unit, x_unit )
    y_range = np.arange ( y_min, y_max + y_unit, y_unit )        
    
    # Configure figure.
    
    #plt.plot( data[:,X], data[:,Y], plot_style, markersize = 2, alpha = 1.0 )

    for k, col in zip(unique_labels, colors):
        
        if k == -1:
            # Black used for noise.
            col = 'k'
            ms = int ( round ( marker_size * 1.5 ) )
        else:
            ms = marker_size
        
        class_member_mask = ( data_labels == k )
        
        xy = data [ class_member_mask & core_samples_mask ]
        plt.plot ( xy [ :, X ], xy [ :, Y ], '.', markerfacecolor = col, markeredgecolor = col, markersize = ms )
    
        xy = data [ class_member_mask & ~core_samples_mask ]
        plt.plot ( xy [ :, X ], xy [ :, Y ], '.', markerfacecolor = col, markeredgecolor = col, markersize = ms )

    plt.xticks ( x_range, fontsize = font_size_tick )
    plt.yticks ( y_range, fontsize = font_size_tick )
    
    pylab.ylim ( [ x_min, x_max ] )
    pylab.xlim ( [ y_min, y_max ] )

    plt.xlabel ( x_label, fontsize = font_size_label )
    plt.ylabel ( y_label, fontsize = font_size_label, rotation = 0 )
    
    plt.grid ( True )
    plt.axes().set_aspect ( 'equal' )

    plt.show()  
    

# Main function.

def main ():
    
    # Generate data.    
    
    data = []        

    #data = generate_spiral ( data )
    data = generate_rings ( data )
    
    # CLuster data.
    
    db                                            = DBSCAN ( eps = 0.02, min_samples = 8 )    
    data_labels                                   = db.fit_predict ( data )
    core_samples_mask                             = np.zeros_like ( data_labels, dtype = bool )
    core_samples_mask [ db.core_sample_indices_ ] = True
    cluster_count                                 = len ( set ( data_labels ) ) - ( 1 if -1 in data_labels else 0 )
        
    # Plot data
    
    np_data = np.array ( data )
    plot_data ( np_data, data_labels, core_samples_mask )
    
    # Test
    
    print ( 'Length of data = ' + str ( len ( data )  ) )
    print ( 'Cluster count  = ' + str ( cluster_count ) )
    
    return data, data_labels

# Program entry point.

data, data_labels = main()

