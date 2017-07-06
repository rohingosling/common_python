#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Library imports.

import winsound
import time
import os

import pandas as pd
import numpy  as np

from sklearn.decomposition  import PCA
from sklearn.neural_network import MLPClassifier
from sklearn                import cross_validation, metrics


# application imports.

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# GLOBAL CONSTANTS
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Text constants

C_TEXT_SYSTEM       = 'SYSTEM'
C_TEXT_APPLICATION  = 'APPLICATION'
C_TEXT_MODEL        = 'MODEL'

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: initialize application
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def initialize ():
    
    console_log ( C_TEXT_SYSTEM, 'Initialize Program.', True, True, True )
    
    # Disable warnings.
    # - We do this specificaly to disable TensorFlow warnings.
    
    os.environ [ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'
    
    # Initialize random number generaor.
    
    np.random.seed ( 0 )
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------    
# FUNCTION: report_time
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def console_report_time ( time_start, time_stop, elapsed_time ):

    # Local constants.
    
    TIME_FORMAT  = '%H:%M:%S'
    TIME_START   = 'Time.Start:   '
    TIME_STOP    = 'Time.Stop:    '
    TIME_ELAPSED = 'Time.Elapsed: '
    
    # Sound parameters.
    
    frequency = 12000
    period    = 50
    repeat    = 2
        
    # Report time        
        
    print ( TIME_START   + time.strftime ( TIME_FORMAT, time.gmtime ( time_start ) ) )
    print ( TIME_STOP    + time.strftime ( TIME_FORMAT, time.gmtime ( time_stop ) ) )
    print ( TIME_ELAPSED + time.strftime ( TIME_FORMAT, time.gmtime ( elapsed_time ) ) )
        
    sound ( frequency, period, repeat )

#---------------------------------------------------------------------------------------------------------------------------------------------------------------    
# FUNCTION: console_log
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def console_log ( text_category, text, new_line_before, new_line_after, sound_enabled = False ):
    
    # Local constants.
    
    TEXT_CATEGORY_BRACKET_OPEN  = '['
    TEXT_CATEGORY_BRACKET_CLOSE = '] '
    
    # Local variables.
    
    frequency = 100
    period    = 50
    repeat    = 1
    
    # write text to console.
    
    if new_line_before:
        console_new_line ()
        
    if text_category != '':
        text_category_string = TEXT_CATEGORY_BRACKET_OPEN + text_category + TEXT_CATEGORY_BRACKET_CLOSE
        print ( text_category_string + text )
        
    if new_line_after:
        console_new_line ()
    
    if sound_enabled:
        sound ( frequency, period, repeat )


#-----------------------------------------------------------------------------
# Plot model data to console.
#-----------------------------------------------------------------------------

def report_training_results ( model, t, t_predictions, t_prediction_probabilities ):
    
    CONSOLE_ALIGN_KEY = '{0:.<24}'        
        
    accuracy = metrics.accuracy_score ( t.values, t_predictions              ) * 100
    auc      = metrics.roc_auc_score  ( t,        t_prediction_probabilities )
    logloss  = metrics.log_loss       ( t,        t_prediction_probabilities )
        
    console_log ( C_TEXT_SYSTEM, CONSOLE_ALIGN_KEY.format ( 'Accuracy' ) + ' = ' + '{:.3f}'.format ( accuracy ), False, False, True )
    console_log ( C_TEXT_SYSTEM, CONSOLE_ALIGN_KEY.format ( 'AUC' )      + ' = ' + '{:.6f}'.format ( auc ),      False, False, True )
    console_log ( C_TEXT_SYSTEM, CONSOLE_ALIGN_KEY.format ( 'logloss' )  + ' = ' + '{:.6f}'.format ( logloss ),  False, False, True )


#---------------------------------------------------------------------------------------------------------------------------------------------------------------    
# FUNCTION: console_new_line
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def console_new_line ():
    print ( '' )
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------    
# FUNCTION: sound.
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def sound ( f, p, n  ):
    for i in range ( n ):
        winsound.Beep ( f, p )

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: main
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def main ():
    
    # Initialize program.
    
    initialize ();  
    

    # Srart clock.    
    
    time_start = time.time()


    # Initialize local variables.

    path_data             = '../data/'
    file_name_training    = 'numerai_training_data.csv'
    file_name_live        = 'numerai_tournament_data.csv'
    file_name_predictions = 'predictions.csv'
    

    # Load data
    
    console_log ( C_TEXT_MODEL, 'Loading training data, "' + path_data + file_name_training + '"', False, False, True )
    console_log ( C_TEXT_MODEL, 'Loading live data, "' + path_data + file_name_live + '"', False, False, True )
    
    training_data   = pd.read_csv ( path_data + file_name_training, header = 0 )
    prediction_data = pd.read_csv ( path_data + file_name_live,     header = 0 )
    

    # Format the loaded CSV data into numpy arrays

    console_log ( C_TEXT_MODEL, 'Formating data from CSV to numpy array.', False, False, True )
    
    features     = [ f for f in list ( training_data ) if "feature" in f ]    
    
    X            = training_data   [ features ]
    Y            = training_data   [ "target" ]    
    
    x_prediction = prediction_data [ features ]    
    ids          = prediction_data [ "id" ]
    
    
    # Select features.
    
    console_log ( C_TEXT_MODEL, 'Selecting features.', False, False, True )
    
    feature_count  = len ( X.columns )
    transformation = PCA ( n_components = feature_count )
    
    transformation.fit ( X )
    Xt = transformation.transform ( X )
    
    
    # Configure Neural network.

    model = MLPClassifier (
        hidden_layer_sizes = ( 7, 1 ),
        activation         = 'tanh',
        solver             = 'adam',
        learning_rate      = 'adaptive',
        momentum           = 0.9,
        alpha              = 0.00001,
        random_state       = 1
    )
    
    
    # Train model.
    
    console_log ( C_TEXT_MODEL, 'Training model.', False, False, True )    
    
    model.fit ( Xt, Y )
    
    
    # Report results.
    
    console_new_line ()
    console_log ( C_TEXT_SYSTEM, 'Training results.', False, False, True )
    
    t_predictions              = model.predict       ( Xt )
    t_prediction_probabilities = model.predict_proba ( Xt ) [ :, 1 ]
    
    report_training_results ( model, Y, t_predictions, t_prediction_probabilities )
    
    
    # Use trained model to predict production targets.
        
    console_log ( C_TEXT_MODEL, 'Predicting live data.', True, False, True )        
    
    xt_prediction = transformation.transform ( x_prediction )
    y_prediction  = model.predict_proba ( xt_prediction )
    results       = y_prediction [ :, 1 ]
    results_df    = pd.DataFrame ( data = { 'probability' : results } )
    predictions   = pd.DataFrame ( ids ).join ( results_df )
    
    
    # Save prediction data.
    
    console_log ( C_TEXT_MODEL, 'Saving live predictions, "' + path_data + file_name_predictions + '"', False, False, True )

    predictions.to_csv ( path_data + file_name_predictions, index = False )
    
   
    # Stop clock.
    
    time_stop    = time.time()
    elapsed_time = time_stop - time_start
   

    # Report time..
    
    console_new_line ()
    console_report_time ( time_start, time_stop, elapsed_time )
    console_new_line ()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Program entry point.    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    
if __name__ == '__main__':
    main()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------