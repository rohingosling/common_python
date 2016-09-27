# Library imports.

import numpy  as np
import math

# FUNCTION: Polar projection.
#
# PRECONDITIONS:
# - Both polar coordinates must be normalised to the range, [0.0..1.0].

def polar_to_cartesian_projection ( t, r ):
    
    # Local constants.
    
    PI = math.pi
        
    # Convert angular coordinate to radians.
    # - We prsume the angular coordinate is expressed as a real in the range [0.0..1.0]
    
    t = 2.0 * PI * t    

    # Compute cartesian to polar projection.        
    
    x = r * math.sin ( t )
    y = r * math.cos ( t )
    
    return x, y

# Function generate gausian spirals.

def generate_data_gausian_archimedean_spiral ( data, xd, yd, sigma, turn_count, spiral_count, data_count ):
    
    # local constants
    
    MU = 0.0
    MARKER_CLUSTERS_ENABLED = True   

    # MArker cluster flags.

    marker_cluster_generated = [ False, False, False ]         
    
    # Convert integers for floats for floating point calculations.
    
    f_spiral_count = float ( spiral_count )
    f_data_count   = float ( data_count )    
    
    # Configure polar coordinate space.
    
    t_min   = 0.0
    t_max   = 1.0
    r_scale = 0.5
    
    # Configure Archimedean spiral parameters.
    
    a = 1.0           # Spiral polar gradient.
    b = 0.25 / 2.0    # Spiral polar r intercept. 
    
    # Compute angular step sise that acomodates total data point count in 2*PI radians..
    
    t_step = f_spiral_count * t_max / ( f_data_count )
    
    # Compute radial phase step, for incrementing the phase of multiple spirals.
    
    t_phase_min  = 0.0
    t_phase_max  = 1.0
    t_phase_step = 1.0 / f_spiral_count
    
    # Generate Gausian Archimedean spiral data.
    
    for t_phase in np.arange ( t_phase_min, t_phase_max, t_phase_step ):
    
        for t in np.arange ( t_min, t_max, t_step ):
            
            # Compute Archimedean spiral. i.e. A strait line in polar space.
            
            r = a*t + b
            
            # Compute turn count.
            
            t = t * turn_count
            
            # Compute phase shift for current spiral arm.
            
            t += t_phase
            
            # Generate random variables.
            
            rt =       np.random.normal ( MU, sigma )
            rr = abs ( np.random.normal ( MU, sigma ) )
            
                        
            # Apply Gausian distortion.
            
            t += rt
            r += rr
                    
            # Scale polar r axis.
            
            r *= r_scale
                    
            # Perform polar to cartesian projection. (x,y) <- (t,r).        
            
            x, y = polar_to_cartesian_projection ( t, r )
            
            # Translate geometry.
            
            x += xd
            y += yd
            
            # Add marker blobs.
            
            if MARKER_CLUSTERS_ENABLED:
                if ( t_phase >= t_phase_min ) and ( t_phase < t_phase_step ):
                    
                    t_range = 0.005
                    tm0     = 0.75                
                    tm1     = 1.0
                    tm2     = 1.25
                    m_count = 1000
                    m_sigma = 0.05
                    
                    if not marker_cluster_generated[0]:
                        if ( t >= tm0 - t_range ) and ( t < tm0 + t_range ):
                            data = generate_data_gausian_cluster ( data, x, y, m_sigma, m_count )
                            marker_cluster_generated[0] = True
                        
                    if not marker_cluster_generated[1]:
                        if ( t >= tm1 - t_range ) and ( t < tm1 + t_range ):
                            data = generate_data_random_gausian_clusters ( data = data, x = x, y = y, radius = 0.05, sigma_min = 0.008, sigma_max = 0.016, cluster_count = 16, data_count = 400 )
                            marker_cluster_generated[1] = True
                    
                    if not marker_cluster_generated[2]:  
                        if ( t >= tm2 - t_range ) and ( t < tm2 + t_range ):
                            data = generate_data_gausian_ring_formation ( data, x, y, 0.025, 0.075, 0.003, 3, 600 )
                            marker_cluster_generated[2] = True
            
            # Generate data point.
            
            data.append ( [ x, y ] )        
        
    return data

# Function: Generate Gausian cluster.

def generate_data_gausian_cluster ( data, xd, yd, sigma, data_count ):
    
    # Local data.
    
    mu = 0.0        
        
    # Generate data    
    
    for i in range ( data_count ):
        
        x_variance = 0.5 * np.random.normal ( mu, sigma )
        y_variance = 0.5 * np.random.normal ( mu, sigma )
        
        x = xd + x_variance
        y = yd + y_variance
        
        data.append ( [ x, y ] )
    
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
