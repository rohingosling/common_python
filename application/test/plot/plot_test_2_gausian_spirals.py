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

# Function generate gausian spirals.

def generate_data_gausian_archimedean_spiral ( data, x, y, sigma, turn_count, spiral_count, data_count ):
    
    # local constants
    
    PI = math.pi
    MU = 0.0        
    
    # Convert integers for floats for floating point calculations.
    
    f_spiral_count = float ( spiral_count )
    f_data_count   = float ( data_count )    
    
    # Configure polar coordinate space.
    
    t_min   = 0.0
    t_max   = 1.0
    r_scale = 0.5
    
    # Configure Archimedean spiral parameters.
    
    a = 1.0     # Spiral polar gradient.
    b = 0.0     # Spiral polar r intercept. 
    
    # Compute angular step sise that acomodates total data point count in 2*PI radians..
    
    t_step = f_spiral_count * t_max / ( f_data_count )
    
    # Compute radial phase step, for incrementing the phase of multiple spirals.
    
    t_phase_min  = 0.0
    t_phase_max  = 2 * PI
    t_phase_step = 2 * PI / f_spiral_count
    
    # Generate Gausian Archimedean spiral data.
    
    for t_phase in np.arange ( t_phase_min, t_phase_max, t_phase_step ):
    
        for t in np.arange ( t_min, t_max, t_step ):
            
            # Compute Archimedean spiral. i.e. A strait line in polar space.
            
            r = a*t + b
            
            # Compute turn count.
            
            t = t * turn_count 
            
            # Apply Gausian distortion.
            
            t += np.random.normal ( MU, sigma )
            r += np.random.normal ( MU, sigma )
                    
            # Scale polar r axis.
            
            r *= r_scale
            
            # Convert t to radians
            
            t_rad = 2.0 * PI * t
            
            # Compute phase shift for current spiral arm.
            
            t_rad += t_phase
                    
            # Perform cartesian to polar projection.        
            
            x_polar = x + r * math.sin ( t_rad )
            y_polar = y + r * math.cos ( t_rad )
            
            # Generate data point.
            
            data.append ( [ x_polar, y_polar ] )
    
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
    
# Function: Generae randome gausian clusters.
    
def generate_data_random_gausian_clusters ( data, x, y, radius, sigma_min, sigma_max, cluster_count, data_count ):
    
    # Validate data inputs.
    
    if radius < 0.0:
        radius = 0.0
    
    if sigma_min < 0.0:
        sigma_min = 0.0
    
    if sigma_max < sigma_min:
        sigma_max = sigma_min
        
    if ( cluster_count < 0.0 ):
        cluster_count = 0.0

    if ( data_count < 0.0 ):
        data_count = 0.0
    
    # Convert integer counts to reals for use in floating point computations.

    dn  = float ( data_count )    
    cn  = float ( cluster_count )
    
    # Compute the number of data points per clister.    
    
    dcn = dn / cn
    
    # Generate data.
    
    for ci in range ( cluster_count ):
        
        # compute random locaton for data cluster.
        
        xr = radius * ( 2.0 * np.random.random() - 1.0 )
        yr = radius * ( 2.0 * np.random.random() - 1.0 )
        
        # Compute random sigma.
        
        sr = sigma_min + ( sigma_max - sigma_min ) * np.random.random()
        
        # Generate gausian distribution.
        
        xd = x + xr 
        yd = y + yr
        
        data = generate_data_gausian_cluster ( data, xd, yd, sr, int ( round ( dcn ) ) )
    
    # return data to caller.
    
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
    
    x            = 0.5
    y            = 0.5 
    turn_count   = 2.0
    sigma        = 0.04    
    spiral_count = 2    
    data_count   = 1000 * 4
    
    # Generate data.    
    
    data = []        
    data = generate_data_gausian_archimedean_spiral ( data, x, y, sigma, turn_count, spiral_count, data_count )
    
    # Plot data
    
    np_data = np.array ( data )
    plot_data ( np_data )
    
    # Test
    
    print ( 'Length of data = ' + str ( len(data) ) ) 
    
# Program entry point.    
    
main ()