# Library imports.

import numpy  as np
import math

from sklearn import preprocessing

# FUNCTION: generate_test_data

def generate_test_data ( data, labels ):
    
    test_case = 6
    
    if test_case == 1:      # Two rings.
        data, labels = test_data_1 ( data, labels)

    elif test_case == 2:    # Torus knot.                
        data, labels = test_data_2 ( data, labels)

    elif test_case == 3:    # Duel toru sknot.
        data, labels = test_data_3 ( data, labels)

    elif test_case == 4:    # Torus manifold.
        data, labels = test_data_4 ( data, labels)        

    elif test_case == 5:    # Geosphere.
        data, labels = test_data_5 ( data, labels)

    elif test_case == 6:    # Geosphere.
        data, labels = test_data_6 ( data, labels)
    
    else:
        print ( 'WARNING: No test case selected.\n' )

    # Return data.
    
    return data, labels

# FUNCTION: Test data 1 - Two Rings

def test_data_1 ( data, labels):
        
    # Initialize local variables.    
    
    data_count = int ( 1000 * 2 )
    sigma      = 0.1
    R          = 0.5
    
    # Generage data.
    
    data_torus = []
    data_torus = generate_gausian_torus  ( data_torus, R, sigma, data_count )    
    data_torus = transform_data_rotate   ( data_torus,  0.0, 0.0, 0.0 )
    data_torus = transform_data_displace ( data_torus, -0.25, 0.0, 0.0 )
    data       = data_torus
    
    data_labels = [ [ 0 ] ] * data_count     
    labels      = data_labels
    
    
    data_torus = []
    data_torus = generate_gausian_torus ( data_torus, R, sigma, data_count )
    data_torus = transform_data_rotate   ( data_torus,  0.25, 0.0, 0.0 )
    data_torus = transform_data_displace ( data_torus,  0.25, 0.0, 0.0 )
    data       = np.concatenate ( ( data, data_torus ), axis = 0 )
    
    data_labels = [ [ 1 ] ] * data_count     
    labels      = np.concatenate ( ( labels, data_labels ), axis = 0 )
    
    # Return data.
    
    return data, labels

# FUNCTION: Test data 2 - Torus Knot

def test_data_2 ( data, labels):
        
    # Initialize local variables.    
    
    data_count = int ( 1000 * 4.0 )
    p          = 2.0
    q          = 3.0
    sigma      = 0.33
    R          = 0.75
    
    # Generage data.
    
    data_x = []
    data_x = generate_gausian_torus_knot ( data_x, p, q, R, sigma, data_count )   
    data_x = transform_data_rotate       ( data_x,  0.0, 0.0, 0.0 )
    data_x = transform_data_displace     ( data_x,  0.0, 0.0, 0.0 )
    data   = data_x
    
    data_y = [ [ 0 ] ] * data_count     
    labels = data_y

    
    # Return data.
    
    return data, labels
    
# FUNCTION: Test data 3 - Duel Torus knots.

def test_data_3 ( data, labels):
        
    # Initialize local variables.    
    
    data_count = int ( 1000 * 1.0 )
    p          = 2.0
    q          = 3.0
    sigma      = 0.5
    R          = 0.75
    
    # Generage data.
    
    s      = [1,1,1]
    data_x = []    
    data_x = generate_gausian_torus_knot ( data_x, p, q, R, s, sigma, data_count )   
    data_x = transform_data_rotate       ( data_x,  0.0, 0.0, 0.0 )
    data_x = transform_data_displace     ( data_x,  0.0, 0.0, 0.0 )
    data   = data_x
    
    data_y = [ [ 0 ] ] * data_count     
    labels = data_y
    
    s      = [-1,1,-1]
    data_x = []
    data_x = generate_gausian_torus_knot ( data_x, p, q, R, s, sigma, data_count )   
    data_x = transform_data_rotate       ( data_x,  0.0, 0.0, 0.0 )
    data_x = transform_data_displace     ( data_x,  0.0, 0.0, 0.0 )
    data   = np.concatenate ( ( data, data_x ), axis = 0 )
    
    data_y = [ [ 1 ] ] * data_count     
    labels = np.concatenate ( ( labels, data_y ), axis = 0 )
    
    # Return data.
    
    return data, labels

# FUNCTION: Test data 4 - Matryoshka Torus manifold.

def test_data_4 ( data, labels ):
        
    # Initialize local variables.  
        
    duel_manifold = True
    
    data_count = int ( 1000 * 0.5 )
    sigma      = 0.001
    R          = 0.75
    r          = 0.25
    
    # Generage data.
    
    # 1/2
    
    data_x = []
    data_x = generate_gausian_torus_manifold  ( data_x, R, r, sigma, data_count )    
    data_x = transform_data_rotate            ( data_x,  0.0, 0.0, 0.0 )
    data_x = transform_data_displace          ( data_x,  0.0, 0.0, 0.0 )
    data   = data_x
    
    data_y = [ [ 0 ] ] * data_count     
    labels = data_y
    
    # 2/2    
    
    if duel_manifold:     
    
        data_x = []
        data_x = generate_gausian_torus_manifold  ( data_x, R, r/2.0, sigma, data_count )    
        data_x = transform_data_rotate            ( data_x,  0.0, 0.0, 0.0 )
        data_x = transform_data_displace          ( data_x,  0.0, 0.0, 0.0 )
        data   = np.concatenate ( ( data, data_x ), axis = 0 )
        
        data_y = [ [ 1 ] ] * data_count     
        labels = np.concatenate ( ( labels, data_y ), axis = 0 )

    # Return data.
    
    return data, labels
    
# FUNCTION: Test data 4 - Geosphere.

def test_data_5 ( data, labels ):
        
    # Initialize local variables.  
    
    data_count = int ( 1000 * 1 )
    sigma      = 0.1
    R          = 0.75
    
    # Generage data.
    
    # 1/3
    
    data_x = []
    data_x = generate_gausian_torus           ( data_x, R, sigma, data_count )    
    data_x = transform_data_rotate            ( data_x,  0.0, 0.0, 0.0 )
    data_x = transform_data_displace          ( data_x,  0.0, 0.0, 0.0 )
    data   = data_x
    
    data_y = [ [ 0 ] ] * data_count     
    labels = data_y
    
    # 2/3    
    
    data_x = []
    data_x = generate_gausian_torus           ( data_x, R, sigma, data_count )    
    data_x = transform_data_rotate            ( data_x,  0.25, 0.0, 0.0 )
    data_x = transform_data_displace          ( data_x,  0.0, 0.0, 0.0 )
    data   = np.concatenate ( ( data, data_x ), axis = 0 )
    
    data_y = [ [ 0 ] ] * data_count     
    labels = np.concatenate ( ( labels, data_y ), axis = 0 )

    # 3/3    
    
    data_x = []
    data_x = generate_gausian_torus           ( data_x, R, sigma, data_count )    
    data_x = transform_data_rotate            ( data_x,  0.0, 0.25, 0.0 )
    data_x = transform_data_displace          ( data_x,  0.0, 0.0,  0.0 )
    data   = np.concatenate ( ( data, data_x ), axis = 0 )
    
    data_y = [ [ 0 ] ] * data_count     
    labels = np.concatenate ( ( labels, data_y ), axis = 0 )

    # Return data.
    
    return data, labels


# FUNCTION: Test data 6 - Geosphere.

def test_data_6 ( data, labels ):
        
    # Initialize local variables.  
        
    duel_manifold = True
    n             = 4.0
    R             = 0.75
    r             = 0.25
    Rn            = n * 22.0 
    rn            = n * 7.0
    sigma         = 0.01
    data_count    = int ( Rn * rn )
    
    # Generage data.
    
    # 1/2
    
    data_x = []
    data_x = generate_regular_gausian_torus_manifold  ( data_x, R, r, Rn, rn, sigma )    
    data_x = transform_data_rotate                    ( data_x,  0.0, 0.0, 0.0 )
    data_x = transform_data_displace                  ( data_x,  0.0, 0.0, 0.0 )
    data   = data_x
    
    data_y = [ [ 0 ] ] * data_count     
    labels = data_y
    
    # 2/2    
    
    if duel_manifold:     
    
        data_x = []
        data_x = generate_regular_gausian_torus_manifold  ( data_x, R, r/2.0,  Rn, rn, sigma )    
        data_x = transform_data_rotate                    ( data_x,  0.0, 0.0, 0.0 )
        data_x = transform_data_displace                  ( data_x,  0.0, 0.0, 0.0 )
        data   = np.concatenate ( ( data, data_x ), axis = 0 )
        
        data_y = [ [ 1 ] ] * data_count     
        labels = np.concatenate ( ( labels, data_y ), axis = 0 )

    # Return data.
    
    return data, labels

# FUNCTION: Generate_gausian_torus

def generate_gausian_torus ( data, R, sigma, data_count ):
    
    # local variables.
    
    pi = math.pi
    mu = 0.0
        
    # Generate data.
    
    ti = 2.0 * pi / data_count    
    
    for t in np.arange ( 0.0, 2.0 * pi, ti ):
        
        # Compute random parameters.
        
        r = np.random.normal ( mu, sigma )    # Poloidal radius
        s = 2.0 * pi * np.random.random ()    # Toroidal angle.

        # Compute radial coeficient.        

        a = ( R + r * math.cos ( s ) )        
        
        # Cmopute torus.        
        
        x = a * math.cos ( t )
        y = a * math.sin ( t )
        z = r * math.sin ( s )
        
        # Add point to data set.        
        
        data.append ( [ x, y, z ] )           
    
    return data

# FUNCTION: generate_gausian_torus_manifold

def generate_gausian_torus_manifold ( data, R, r, sigma, data_count ):
    
    # local variables.
    
    pi = math.pi
    mu = 0.0
        
    # Generate data.
    
    ti = 2.0 * pi / data_count    
    
    for t in np.arange ( 0.0, 2.0 * pi, ti ):
        
        # Compute random parameters.
        
        rs = np.random.normal ( mu, sigma )    # Poloidal radius
        s  = 2.0 * pi * np.random.random ()    # Toroidal angle.

        # Compute radial coeficient.        

        a = ( R + ( r + rs ) * math.cos ( s ) )        
        
        # Cmopute torus.        
        
        x = a * math.cos ( t )
        y = a * math.sin ( t )
        z = ( r + rs ) * math.sin ( s )
        
        # Add point to data set.        
        
        data.append ( [ x, y, z ] )           
    
    return data

# FUNCTION: generate_gausian_torus_manifold

def generate_regular_gausian_torus_manifold ( data, R, r, Rn, rn, sigma ):
    
    # local variables.
    
    pi = math.pi
    mu = 0.0
        
    # Generate data.
    
    ti = 2.0 * pi / Rn
    si = 2.0 * pi / rn     
    
    for t in np.arange ( 0.0, 2.0 * pi, ti ):
        
        for s in np.arange ( 0.0, 2.0 * pi, si ):
            
            # Compute radial coeficient.        
    
            a = R + r * math.cos ( s )        
            
            # Cmopute torus.        
            
            x = a * math.cos ( t )
            y = a * math.sin ( t )
            z = r * math.sin ( s )
            
            # Compute random displacement.
        
            xn = np.random.normal ( mu, sigma )
            yn = np.random.normal ( mu, sigma )
            zn = np.random.normal ( mu, sigma )
            
            x += xn
            y += yn
            z += zn
            
            # Add point to data set.        
            
            data.append ( [ x, y, z ] )
    
    return data

# FUNCTION: generate_gausian_torus_knot

def generate_gausian_torus_knot ( data, p, q, R, s, sigma, data_count ):
    
    # local variables.
    
    pi = math.pi
    mu = 0.0
        
    # Generate data.
        
    ti = 2.0 * pi / data_count
    
    for t in np.arange ( 0.0, 2.0 * pi, ti ):

        # Compute radial coeficient.        

        a = R + math.cos ( q * t ) + 4.0
        
        # Cmopute torus.        
        
        x =  s[0] * ( a * math.cos ( p * t ) - 3.0 * math.cos ( ( p - q ) * t ) )
        y =  s[1] * ( a * math.sin ( p * t ) - 3.0 * math.sin ( ( p - q ) * t ) )
        z =  s[2] * ( 4.0 * math.sin ( q * t ) )
        
        # Compute random displacement.
        
        xn = np.random.normal ( mu, sigma )
        yn = np.random.normal ( mu, sigma )
        zn = np.random.normal ( mu, sigma )
        
        x += xn
        y += yn
        z += zn
        
        # Add point to data set.        
        
        data.append ( [ x, y, z ] ) 

    # Normalize data.

    max_abs_scaler  = preprocessing.MaxAbsScaler()
    data_normalized = max_abs_scaler.fit_transform ( data )
        
    return data_normalized

# FUNCTION: transform_data_displace

def transform_data_displace ( data, xd, yd, zd ):
    
    # Get data count.
    
    data_count = len ( data )    
    
    # Compute data transformation.
    
    for i in range ( data_count ):
        
        data[i][0] += xd
        data[i][1] += yd
        data[i][2] += zd
    
    return data

# FUNCTION: transform_data_rotate

def transform_data_rotate ( data, pitch, yaw, roll ):
    
    # local constants.
    
    pi = math.pi
    
    # Get data count.
    
    data_count = len ( data )   
    
    # Convert angles to radians.
    
    pitch = 2.0 * pi * pitch
    yaw   = 2.0 * pi * yaw
    roll  = 2.0 * pi * roll
    
    # Compute data transformation.
    
    for i in range ( data_count ):
        
        # Read data.
        
        x = data[i][0]
        y = data[i][1]
        z = data[i][2]
        
        # Compute pitch (x).
        
        yt =  y * math.cos ( pitch ) + z * math.sin ( pitch )
        zt = -y * math.sin ( pitch ) + z * math.cos ( pitch )
        y  = yt
        z  = zt

        # Compute yaw (y).

        xt =  x * math.cos ( yaw ) + z * math.sin ( yaw )
        zt = -x * math.sin ( yaw ) + z * math.cos ( yaw )
        x  = xt
        z  = zt

        # Compute roll (z).

        xt =  x * math.cos ( roll ) + y * math.sin ( roll )
        yt = -x * math.sin ( roll ) + y * math.cos ( roll )        
        x  = xt
        y  = yt
        
        # Update data.        
        
        data[i][0] = x
        data[i][1] = y
        data[i][2] = z
    
    return data
