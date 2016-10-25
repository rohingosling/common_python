# libray imports

import winsound
import time

import numpy as np

from sklearn import manifold
from sklearn import preprocessing

# application imports.

from data_generator      import generate_test_data
from data_visualization  import plot_data_2d, plot_data_3d
from sklearn.ensemble    import GradientBoostingClassifier
from sklearn             import cross_validation, metrics
from sklearn.grid_search import GridSearchCV 

# FUNCTION: sound.

def sound ( f, p, n  ):
    for i in range ( n ):
        winsound.Beep ( f, p )

# FUNCTION: Compute dimentionality reduction embeddling.

def compute_dimentionality_reduction_embedding ( x, y ):
    
    print ( '\nComputing dimentionality reduction embedding.\n' )
            
    perplexity   = 20
    n_components = 3
    
    embedding = manifold.TSNE ( n_components = n_components, perplexity = perplexity, init = 'pca', random_state = 0  )    
    x_tSNE    = embedding.fit_transform ( x )
   
    # Normalize data.

    max_abs_scaler    = preprocessing.MaxAbsScaler()
    x_tSNE_normalized = max_abs_scaler.fit_transform ( x_tSNE )            
    
    if n_components == 2:
        plot_data_2d ( x_tSNE_normalized, y, markersize = 2, alpha = 1.0, auto_limit_enabled = False )
        
    elif n_components == 3:
        plot_data_3d ( x_tSNE_normalized, y )
        
    return x_tSNE_normalized

# FUNCTION: fit the model.

def modelfit ( model, x, y, performCV = True, printFeatureImportance = True, cv_folds = 5 ):
    
    # Fit the algorithm on the data
    
    model.fit ( x, y[:,0] )    

# FUNCTION: Main program function.    
    
def main ():
    
    # Initialize program.
    
    time_start = time.time() 
    print ( '\nInitialize Program.' )          
    sound ( 10000, 80, 1 )
    
    # Configure data parameters.
    
    training_enabled       = True
    normalise_input_data   = False
    
    # Generate data.    
    
    data   = []
    labels = [] 
       
    data, labels = generate_test_data ( data, labels )
    
    # Normalise data.
    
    if normalise_input_data:
        max_abs_scaler  = preprocessing.MaxAbsScaler()
        data            = max_abs_scaler.fit_transform ( data )
    
    # Convert data to Numpy format.
    
    xt = np.array ( data   )
    yt = np.array ( labels )    
    
    # Only begin training sequence, if there is data.
    
    if ( len ( data ) > 0 ):
        
        # Plot untransformed data
        
        plot_data_3d ( xt, yt )
        
        # Begin training sequence..
        
        if training_enabled:
            
            # learn t-SNE embedding.
            
            compute_dimentionality_reduction_embedding ( xt, yt )            
            
            #model = GradientBoostingClassifier ()
            
            #modelfit ( model, xt, yt )
            
    
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