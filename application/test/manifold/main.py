# libray imports

import winsound
import time

import numpy as np

from sklearn import manifold
from sklearn import preprocessing

# application imports.

from data_generator     import generate_test_data
from data_visualization import plot_data_2d, plot_data_3d

# FUNCTION: sound.

def sound ( f, p, n  ):
    for i in range ( n ):
        winsound.Beep ( f, p )
    

# Function: Main program function.    
    
def main ():
    
    # Initialize program.
    
    time_start = time.time() 
    print ( '\nInitialize Program.\n' )          
    sound ( 10000, 80, 1 )
    
    # Configure data parameters.
    
    embedding_enabled = True
    
    # Generate data.    
    
    data   = []
    labels = [] 
       
    data, labels = generate_test_data ( data, labels )
    
    # Plot data
    
    if ( len ( data ) > 0 ):
        
        # Plot untransformed data
        
        np_data   = np.array ( data   )
        np_lables = np.array ( labels )
        plot_data_3d ( np_data, np_lables )
        
        # Learn manifolds.
        
        if embedding_enabled:
            
            print ( '\nComputing dimentionality reduction embedding.\n' )
            
            perplexity   = 20
            n_components = 2
            
            embedding = manifold.TSNE ( n_components = n_components, perplexity = perplexity, init = 'pca', random_state = 0  )
            
            x      = data
            x_tSNE = embedding.fit_transform ( x )
            
            # Convert data format to Numpy array.
            
            np_data = np.array ( x_tSNE )
            
            # Normalize data.

            max_abs_scaler  = preprocessing.MaxAbsScaler()
            data_normalized = max_abs_scaler.fit_transform ( np_data )            
            
            if n_components == 2:
                plot_data_2d ( np_data, np_lables, markersize = 2, alpha = 1.0, auto_limit_enabled = False )
                
            elif n_components == 3:
                plot_data_3d ( data_normalized, np_lables )
    
    else:
        print ( 'WARNING: No data.\n' )
        
    
    # Print program data.
    
    time_stop    = time.time()
    elapsed_time = time_stop - time_start
    print ( 'Time.Start:   ' + time.strftime ( "%H:%M:%S", time.gmtime ( time_start ) ) )
    print ( 'Time.Stop:    ' + time.strftime ( "%H:%M:%S", time.gmtime ( time_stop ) ) )
    print ( 'Time.Elapsed: ' + time.strftime ( "%H:%M:%S", time.gmtime ( elapsed_time ) ) )
    print ( 'Data Count:   ' + str ( len ( data ) ) )    
    sound ( 12000, 80, 2 )
    
# Program entry point.    
    
main ()