import matplotlib.pyplot as plt
import numpy             as np
import random            as rnd
import math
import pylab

# Function: Generate sinusoidal test data.

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
    
    # Generate data    
    
    for x in np.arange ( x_min, x_max + x_step, x_step ):
        
        y = ( a / 2.0 ) * math.sin ( ( 2*PI * x * f ) + p ) + 0.5
        
        data.append ( [ x, y ] )
    
    return data

# Function: Generate Gausian cluster.

def generate_data_gausian_cluster ( data, x, y, sigma, data_count ):
    
    # Local data.
    
    mu = 0.0        
        
    # Generate data    
    
    for i in range ( data_count ):
        
        x_variance = 0.5 * np.random.normal ( mu, sigma )
        y_variance = 0.5 * np.random.normal ( mu, sigma )
        
        xd = x + x_variance
        yd = y + y_variance
        
        data.append ( [ xd, yd ] )
    
    return data

# Function: Generate Gausian ring cluster.

def generate_data_gausian_ring ( data, x, y, radius, sigma, data_count ):
    
    # local constants
    
    PI = math.pi    
    
    # Local data.
    
    mu = 0.0    
        
    # General outer cluster.
    
    for i in range ( data_count ):
        
        random_angle    = 2.0 * PI * np.random.random ()
        radius_variance = radius + np.random.normal ( mu, sigma )
                
        xd = x + radius_variance * math.sin ( random_angle )
        yd = y + radius_variance * math.cos ( random_angle )
        
        data.append ( [ xd, yd ] )    
    
    return data

# Function: Generate Gausian ring formation.

def generate_data_gausian_ring_formation ( data, x, y, radius_min, radius_max, sigma, cluster_count, data_count ):
    
    # local constants
    
    PI = math.pi  
    
    # Validate input parameters.

    if ( radius_min > radius_max ):
        radius_max = radius_min

    if data_count < 0:
        data_count = 0

    if cluster_count <= 0:
        cluster_count = 1
        
    # Compute radius step size.
    
    if cluster_count > 1:
        radius_step  = ( radius_max - radius_min ) / ( cluster_count - 1 )
    else:
        radius_step = 0.0
    
    # Generate cluster data.
    
    radius = radius_min
    
    if radius_min > 0:
        
        # If the minimum radius is greater than 0, then we will not include a cantral gausian distribution cluster.
        # - As such, the cluster data count factor is computerd diferently, because it excludes consideration
        #   for the a cantral gausian distribution cluster, who's data count is computered difrently to the rest
        #   of the gausian ring clusters.
        
        # Compute the data count factor used, to compute the number of datapoints to include in each concentric cluster.
        
        nc = float ( cluster_count )
        nd = float ( data_count )
        
        cluster_data_count_factor = nd / ( 0.5 * nc * ( nc + 1.0 ) )
        
        # Generate clusters.
        
        for cluster_index in range ( cluster_count ):
            
            # Compute the number of data points to include in this cluster.
            
            i = float ( cluster_index )
            
            cluster_data_count = ( i + 1 ) * cluster_data_count_factor
            
            # Generate cluster data.
            
            data = generate_data_gausian_ring ( data, x, y, radius, sigma, int ( round ( cluster_data_count ) ) )
            
            # Update radius for next concentric cluster.
            
            radius += radius_step        
    else:
        
        # If the minimum radius is zero, then we  shall include a cantral Gausian distribution.
        # - As such, our computation for the cluster data count factor will be altered to acomodate the
        #   number of data samples to generate for the central cluster, as the central cluster data 
        #   count is computered diferently.
        
        # Compute the data count factor used, to compute the number of datapoints to include in each concentric cluster.
        
        nc = float ( cluster_count )
        nd = float ( data_count )
        
        if nc * ( PI * nc + 1.0 ) != PI * nc:        
            cluster_data_count_factor = ( 2.0 * PI * nd ) / ( PI * nc**2 - PI * nc + nc )
        else:
            cluster_data_count_factor = 0
            
        # Generate clusters.
            
        for cluster_index in range ( cluster_count ):
            
            # Compute the number of data points to include in this cluster.
            
            i = float ( cluster_index )
            
            cluster_data_count = cluster_data_count_factor / ( 2.0 * PI ) + i * cluster_data_count_factor
            
            # Generate cluster data.            
            
            if cluster_index > 0:                            
                sigma_scale = 1.0
                data        = generate_data_gausian_ring ( data, x, y, radius, sigma * sigma_scale, int ( round ( cluster_data_count ) ) )
            else:                                
                sigma_scale = 2.0
                data        = generate_data_gausian_cluster ( data, x, y, sigma * sigma_scale, int ( round ( cluster_data_count ) ) )
            
            # Update radius for next concentric cluster.
            
            radius += radius_step     
        
    # Return data.
    
    return data


# Plot data.
    
def plot_data ( data ):
    
    # Constants

    X = 0
    Y = 1
    
    # Local variables.
    
    font_size_label = 18
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
    
    plt.plot( data[:,X], data[:,Y], plot_style, markersize = 2, alpha = 1.0 )

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
    
    # Configure data parameters.
    
    cluster_x          = 0.5
    cluster_y          = 0.5
    cluster_radius_min = 0.25 / 2.0
    cluster_radius_max = 0.25 / 2.0 * 3.0
    cluster_sigma      = 0.025
    cluster_count      = 3
    cluster_data_count = 3000
    
    # Generate data.    
    
    data = []            
    #data = generate_data_gausian_cluster ( data, cluster_x, cluster_y, cluster_sigma, cluster_data_count )
    #data = generate_data_gausian_ring    ( data, ring_x, ring_y, ring_radius, ring_sigma, ring_data_count )    
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
    
    # Plot data
    
    np_data = np.array ( data )
    plot_data ( np_data )
    
    # Test
    
    print ( 'Length of data = ' + str ( len(data) ) ) 
    
# Program entry point.    
    
main ()