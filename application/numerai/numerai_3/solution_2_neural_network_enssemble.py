#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Library imports.

import winsound
import time
import os

import pandas as pd
import numpy  as np

from sklearn.decomposition   import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network  import MLPClassifier
from sklearn                 import metrics

# Application imports.

from utility import console_log, console_new_line, sound, console_report_elapsed_time, report_training_results

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# GLOBAL CONSTANTS
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# CSV Header data

C_CSV_TARGET      = 'target'
C_CSV_FEATURE     = 'feature'
C_CSV_ID          = 'id'
C_CSV_PROBABILITY = 'probability'

# Text constants

C_TEXT_SYSTEM       = '[SYSTEM] '
C_TEXT_APPLICATION  = '[APPLICATION] '
C_TEXT_MODEL        = '[MODEL] '
C_TEXT_REPORTING    = '[REPORTING] '
C_TEXT_TAB          = '| '

# Sound constants

C_FREQUENCY_1 = 100
C_FREQUENCY_2 = 11000

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: initialize application
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def initialize ():
    
    console_log ( C_TEXT_SYSTEM + 'Initialize Program.', indent = 0, lines_before = 1, frequency = C_FREQUENCY_1 )
    
    # Disable warnings.
    # - We do this specificaly to disable TensorFlow warnings.
    
    os.environ [ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'
    
    # Initialize random number generaor.
    
    np.random.seed ( 0 )
    

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: load_training_data
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_training_data ( filename ):

    # Load data.    
    
    console_log ( C_TEXT_MODEL + 'Loading training data, "' + filename + '".', indent = 0, lines_before = 1, frequency = C_FREQUENCY_1 )
    
    training_data = pd.read_csv ( filename, header = 0 )
    
    # Format the loaded CSV data into numpy arrays.
    
    features = [ f for f in list ( training_data ) if C_CSV_FEATURE in f ]
    
    x = training_data [ features ]
    y = training_data [ C_CSV_TARGET ]

    # Return data vectors.

    return x, y


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: load_application_data
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_application_data ( filename ):
    
    console_log ( C_TEXT_MODEL + 'Loading application data, "' + filename + '".', indent = 0, lines_before = 1, frequency = C_FREQUENCY_1 )
    
    application_data = pd.read_csv ( filename, header = 0 )
    
    # Format the loaded CSV data into numpy arrays.
    
    features = [ f for f in list ( application_data ) if C_CSV_FEATURE in f ]
    
    i = application_data [ C_CSV_ID ]
    x = application_data [ features ]    
    
    # Return data vectors.
    
    return i, x

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: select_features
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def select_features ( x ):
    
    console_log ( C_TEXT_MODEL + 'Selecting features.', indent = 0, frequency = C_FREQUENCY_1 )
    
    feature_count  = len ( x.columns )
    transformation = PCA ( n_components = feature_count )
    
    transformation.fit ( x )
    x_transformed = transformation.transform ( x )
    
    return x_transformed, transformation

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: train_model
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def train_model ( x, y, random_state ):
        
    console_log ( C_TEXT_MODEL + 'Training model.', indent = 0, frequency = C_FREQUENCY_1 )
    
    # Configure model parameters.

    model = MLPClassifier (
        hidden_layer_sizes = ( 7, 1 ),
        activation         = 'tanh',
        solver             = 'adam',
        learning_rate      = 'adaptive',
        momentum           = 0.9,
        alpha              = 0.00001,
        random_state       = random_state
    )
    
    # Train the model.
        
    model.fit ( x, y )
    
    # REturn the trained model.
    
    return model
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: predict_application
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def predict_application ( x, transformation, model ):
    
    console_log ( C_TEXT_MODEL + 'Predicting application data.', indent = 0, frequency = C_FREQUENCY_1 )    
    
    # Predict application results.
    
    x_transformed = transformation.transform ( x )
    y             = model.predict_proba      ( x_transformed )
    
    # Return predictions.    
    
    return y



#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: save_application_predictions
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def save_application_predictions ( filename, y_prediction, x_id ):
    
    console_log ( C_TEXT_MODEL + 'Saving application predictions, "' + filename + '".', indent = 0, frequency = C_FREQUENCY_1 )    
    
    # Format prediction results as Pandas dataframe.    
    
    results                = y_prediction [ :, 1 ]
    results_dataframe      = pd.DataFrame ( data = { C_CSV_PROBABILITY : results } )
    y_prediction_dataframe = pd.DataFrame ( x_id ).join ( results_dataframe )
    
    # Save data.
    
    y_prediction_dataframe.to_csv ( filename, index = False )

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: main
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def main ():
    
    # Initialize program.
    
    initialize ();  

    # Srart clock.    
    
    time_start = time.time()

    # Initialize local variables.

    file_path             = '../data/'
    file_name_training    = 'numerai_training_data.csv'
    file_name_application = 'numerai_tournament_data.csv'
    file_name_predictions = 'predictions.csv'
    
    # Train model
    
    x_train, y_train                    = load_training_data ( file_path + file_name_training )
    x_train_transformed, transformation = select_features ( x_train )    
    model1                              = train_model ( x_train_transformed, y_train, 1 )
    model2                              = train_model ( x_train_transformed, y_train, 2 )
    model3                              = train_model ( x_train_transformed, y_train, 3 )
    model4                              = train_model ( x_train_transformed, y_train, 4 )
    model5                              = train_model ( x_train_transformed, y_train, 5 )
        
    # Report results.
    
    report_training_results ( model1, x_train_transformed, y_train )
    report_training_results ( model2, x_train_transformed, y_train )
    report_training_results ( model3, x_train_transformed, y_train )
    report_training_results ( model4, x_train_transformed, y_train )
    report_training_results ( model5, x_train_transformed, y_train )
        
    # Load application data.
    
    x_id, x_application = load_application_data ( file_path + file_name_application )
    y_prediction1        = predict_application ( x_application, transformation, model1 )
    y_prediction2        = predict_application ( x_application, transformation, model2 )
    y_prediction3        = predict_application ( x_application, transformation, model3 )
    y_prediction4        = predict_application ( x_application, transformation, model4 )
    y_prediction5        = predict_application ( x_application, transformation, model5 )
    
    y_prediction = ( y_prediction1 + y_prediction2 + y_prediction3 + y_prediction4 + y_prediction5 ) / 5.0            
    
    # Save prediction data.

    save_application_predictions ( file_path + file_name_predictions, y_prediction, x_id )
      
    # Stop clock.
    
    time_stop    = time.time()
    elapsed_time = time_stop - time_start
   
    # Report time..
    
    console_new_line ()
    console_report_elapsed_time ( time_start, time_stop, elapsed_time )
    

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Program entry point.    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    
if __name__ == '__main__':
    main()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------