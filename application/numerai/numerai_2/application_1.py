#/////////////////////////////////////////////////////////////////////////////
#
# PROGRAM: Numerai - Solution 3
# VERSION: 3.0
# AUTHOR:  Rohin Gosling
#
#/////////////////////////////////////////////////////////////////////////////

#/////////////////////////////////////////////////////////////////////////////
# Import libraries.
#/////////////////////////////////////////////////////////////////////////////

# Platform imports.

import winsound
import time
import datetime

import pandas           as pd
import numpy            as np
import xgboost          as xgb
import matplotlib.pylab as plt

from matplotlib.pylab    import rcParams
from xgboost.sklearn     import XGBClassifier
from sklearn             import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

# Application inports

from numerai_constants  import Constant
from data_visualization import plot_feature_rank

# Platform configuration.

rcParams [ 'figure.figsize' ] = 12, 4

#/////////////////////////////////////////////////////////////////////////////
# GLobal Variables
#/////////////////////////////////////////////////////////////////////////////

time_event = time.time()

#/////////////////////////////////////////////////////////////////////////////
# Functions.
#/////////////////////////////////////////////////////////////////////////////

#-----------------------------------------------------------------------------
# FUNCTION: run
#
# DESCRIPTION:
#
# - This is the main function for this file.
#-----------------------------------------------------------------------------

def main ():
    
    # Initialize application.    
    
    log ( 'Time.Start:. ' + str ( datetime.datetime.now() ) )
    log ( 'Initializing Program.' )
    
    # Train model.

    log ( 'Initiate Training Sequence.' )
    
    model = train_and_test ()
    
    # Apply model.
    
    log ( 'Initiate Application Sequence.' )    
    
    apply_model ( model )
    
    # Shut down application.
    
    log ( 'Program shutting down.' )
    log ( 'Time.Stop:. ' + str ( datetime.datetime.now() ) )
    
    #debug_show_sample_data ( training_data, cols_features, col_target, row_count = 3, precision = 16 )    

#-----------------------------------------------------------------------------
# FUNCTION: Apply model.
#-----------------------------------------------------------------------------

def apply_model ( model ):
    
    # Load application data.
    
    application_data_file_name = Constant.Numerai.DataFile.PATH + Constant.Numerai.DataFile.APPLICATION    
    x, application_data        = load_application_data ( application_data_file_name )
    
    # Apply model.
    
    y = model.predict_proba ( x )   
    
    # Save predictions.
    
    save_application_results ( application_data, y )
    
#-----------------------------------------------------------------------------
# FUNCTION: Parameter searc.
#-----------------------------------------------------------------------------
    
def parameter_search ():
    
    # Parameter search parameters.
    
    row_count = 100
    
    # Load training data.
    
    training_data_file_name = Constant.Numerai.DataFile.PATH + Constant.Numerai.DataFile.TRAINING
    cols_features           = [ Constant.Numerai.CSV.FEATURE + str ( index ) for index in range ( 1, 22 ) ]
    col_target              = Constant.Numerai.CSV.TARGET
    training_data           = load_training_data ( training_data_file_name, row_count )
    
    

#-----------------------------------------------------------------------------
# FUNCTION: Train and test model.
#-----------------------------------------------------------------------------

def train_and_test ():
    
    # Load training data.
    
    training_data_file_name = Constant.Numerai.DataFile.PATH + Constant.Numerai.DataFile.TRAINING
    cols_features           = [ Constant.Numerai.CSV.FEATURE + str ( index ) for index in range ( 1, 22 ) ]
    col_target              = Constant.Numerai.CSV.TARGET
    training_data           = load_training_data ( training_data_file_name )
    
    # Initialize parameters.
    
    search_divisor = 5
    
    row_count_min     = 100
    row_count_max     = 100
    row_count_divisor = search_divisor
    row_count_stride  = ( row_count_max - row_count_min ) / row_count_divisor
    
    learning_rate_min     = 0.001
    learning_rate_max     = 1.0
    learning_rate_divisor = search_divisor
    learning_rate_stride  = ( learning_rate_max - learning_rate_min ) / learning_rate_divisor
    
    n_estimator_min     = 1
    n_estimator_max     = 5000
    n_estimator_divisor = learning_rate_divisor
    n_estimator_stride  = ( n_estimator_max - n_estimator_min ) / n_estimator_divisor
    
        
    # Train and test model.
        
    model = XGBClassifier (
        learning_rate    = Constant.Model.LEARNING_RATE,
        n_estimators     = Constant.Model.N_ESTIMATORS,
        max_depth        = Constant.Model.MAX_DEPTH,
        min_child_weight = Constant.Model.MIN_CHILD_WEIGHT,
        gamma            = Constant.Model.GAMMA,
        subsample        = Constant.Model.SUBSAMPLE,
        colsample_bytree = Constant.Model.COLSAMPLE_BYTREE,
        objective        = Constant.Model.OBJECTIVE,
        scale_pos_weight = Constant.Model.SCALE_POS_WEIGHT,        
        seed             = Constant.Model.SEED     
    )
    
    model = train_model ( model, training_data, cols_features, col_target, cross_validation_enabled = True )
        
    # Return trained model.
    
    return model


#-----------------------------------------------------------------------------
# FUNCTION: Train model.
#-----------------------------------------------------------------------------

def train_model (
        model, 
        data, 
        features,
        target,
        cross_validation_enabled    = True, 
        cross_validation_fold_count = 4, 
        early_stopping_count        = 50
    ):
        
    log ( 'Training model, ' + str ( model.__class__ ) + '.', indent = 1 )
    verbose_eval = 'None'
    
    # Retreive feature and target vectors.
    
    x = data [ features ]
    y = data [ target   ]
        
    # Initialize corss validation.
    
    if cross_validation_enabled:
        
        log ( 'Configure cross validation.', indent = 2 )
        
        xgb_parameters = model.get_xgb_params ()
        xgb_data       = xgb.DMatrix          ( data = x.values, label = y.values )
        
        cross_validation_result = xgb.cv (
        
            xgb_parameters, 
            xgb_data, 
            num_boost_round       = model.get_params() [ 'n_estimators' ],
            nfold                 = cross_validation_fold_count,
            metrics               = Constant.Model.METRIC,
            early_stopping_rounds = early_stopping_count,
            verbose_eval          = verbose_eval
        )
        
        model.set_params ( n_estimators = cross_validation_result.shape [ 0 ] )

        # Compute elepsed time for cross validation.

        time_now               = time.time()
        elapsed_time_formatted = time_to_string ( time_now - time_event )
        log ( 'Cross validation: elapsed_time = ' + elapsed_time_formatted, indent = 3 )
            
    # Fit the model to the data, using the parameters discovered through cross validation.
            
    log ( 'Fitting model.', indent = 2 )
    
    model.fit ( x, y, eval_metric = Constant.Model.METRIC )
        
    # Predict training set.
        
    y_predictions              = model.predict       ( x )
    y_prediction_probabilities = model.predict_proba ( x ) [ :, 1 ]
        
    # Print model report.
        
    console_report ( model, y, y_predictions, y_prediction_probabilities )
                    
    feature_rank = pd.Series ( model.booster().get_fscore()).sort_values ( ascending = False )
        
    #feature_rank.plot ( kind = 'bar', title = 'Feature Rank' )
    #plt.ylabel        ( 'Feature Rank' )
    
    plot_feature_rank ( feature_rank )
    
    # Return trained model.
    
    return model

#-----------------------------------------------------------------------------
# FUNCTION: Time string.
#-----------------------------------------------------------------------------

def time_to_string ( time_sample ):
    
    hours, remainder = divmod ( time_sample, 3600 )
    minutes, seconds = divmod ( remainder,   60   )
    time_string      = '{:0>2}:{:0>2}:{:0>6.3f}'.format ( int ( hours ), int ( minutes ), seconds )    
    
    return time_string


#-----------------------------------------------------------------------------
# FUNCTION: log message to console.
#-----------------------------------------------------------------------------

def log ( message, indent = 0, newline = True, frequency = 0 ):
    
    # Loacal constants.
    
    INDENT_STRING = '| '
    
    # Compile time string.
    
    time_now               = time.time()
    elapsed_time_formatted = time_to_string ( time_now - time_event )
    
    # compile indentation string.
    
    indent_string = INDENT_STRING * indent
    
    # Print message.    
    
    if newline:
        print ( "[" + elapsed_time_formatted + "] " + indent_string + message )
    else:
        print ( "[" + elapsed_time_formatted + "] " + indent_string + message, end = '' )
        
    # Play sound.
        
    if frequency == 0:
        winsound.Beep ( Constant.Sound.EVENT_FREQUENCY, Constant.Sound.EVENT_PERIOD )
    else:
        winsound.Beep ( frequency, 70 )

#-----------------------------------------------------------------------------
# FUNCTION: Load training data.
#-----------------------------------------------------------------------------

def load_training_data ( file_name, row_count = -1 ):
    
    log ( 'Loading training data: ' + '"' + file_name + '"', indent = 1 )
    winsound.Beep ( Constant.Sound.START_FREQUENCY, Constant.Sound.START_PERIOD )
    
    # Load training data.
        
    training_data = pd.read_csv ( file_name )    
    
    # Reduce training data for configuration testing.
    
    if row_count > 0:    
        log ( 'Traning Data: training_data_limit_enabled = TRUE', indent = 2 )
        training_data = training_data.head ( row_count - 1 )
    
    log ( 'Traning Data: row_count = ' + str ( len ( training_data.index ) + 1 ), indent = 2 )
    #winsound.Beep ( Constant.Sound.STOP_FREQUENCY, Constant.Sound.STOP_PERIOD )
    
    return training_data

#-----------------------------------------------------------------------------
# Load application data.
#-----------------------------------------------------------------------------

def load_application_data ( file_name ):
    
    log ( 'Loading application data: ' + '"' + file_name + '"', indent = 1 )
    winsound.Beep ( Constant.Sound.START_FREQUENCY, Constant.Sound.START_PERIOD )
    
    # Load application data from file.

    application_data = pd.read_csv ( file_name )
    
    # Prepare data for execution. y_application = f ( x_application )
    # - Input vector = x_application
    # - Output vector = y_application ...To be allocated after model execution.
    
    x_application = application_data.drop ( Constant.Numerai.CSV.ID, axis = 1 )
    
    log ( 'Application Data: row_count = ' + str ( len ( application_data.index ) + 1 ), indent = 1 )
    #winsound.Beep ( Constant.Sound.STOP_FREQUENCY, Constant.Sound.STOP_PERIOD )
    
    return x_application, application_data

#-----------------------------------------------------------------------------
# Save application results.
#-----------------------------------------------------------------------------

def save_application_results ( data_application, y_application, file_name = Constant.Numerai.DataFile.PREDICTION ):

    #data_application [ Constant.Numerai.CSV.PROBABILITY ] = y_application
    data_application [ Constant.Numerai.CSV.PROBABILITY ] = y_application [ :, 1 ]
    
    # Save the results to file.    
    
    data_application.to_csv (
        Constant.Numerai.DataFile.PATH + file_name, 
        columns = ( Constant.Numerai.CSV.ID, Constant.Numerai.CSV.PROBABILITY ), 
        index   = None
    )

#-----------------------------------------------------------------------------
# Plot model data to console.
#-----------------------------------------------------------------------------

def console_report ( model, y, y_predictions, y_prediction_probabilities ):
    
    FREQUENCY         = 200
    CONSOLE_ALIGN_KEY = '{0:.<24}'
    INDENT_HEADER     = 2
    INDENT_DATA       = 3
        
    accuracy = metrics.accuracy_score ( y.values, y_predictions              ) * 100
    auc      = metrics.roc_auc_score  ( y,        y_prediction_probabilities )
    logloss  = metrics.log_loss       ( y,        y_prediction_probabilities )
    
    log ( 'Model Parameters:',   indent = INDENT_HEADER )
    
    for key in model.get_params():             
        parameter_str  = CONSOLE_ALIGN_KEY.format ( key )
        parameter_str += ' = '
        parameter_str += str ( model.get_params() [ key ] )
        log ( parameter_str, indent = INDENT_DATA, frequency = FREQUENCY )    
        
    log ( 'Training Results:', indent = INDENT_HEADER )
    log ( CONSOLE_ALIGN_KEY.format ( 'Accuracy' ) + ' = ' + '{:.3f}'.format ( accuracy ), indent = INDENT_DATA, frequency = FREQUENCY )
    log ( CONSOLE_ALIGN_KEY.format ( 'AUC' )      + ' = ' + '{:.6f}'.format ( auc ),      indent = INDENT_DATA, frequency = FREQUENCY )
    log ( CONSOLE_ALIGN_KEY.format ( 'logloss' )  + ' = ' + '{:.6f}'.format ( logloss ),  indent = INDENT_DATA, frequency = FREQUENCY )
    


#-----------------------------------------------------------------------------
# DEBUG FUNCTION: Show sample data
#-----------------------------------------------------------------------------


def debug_show_sample_data ( data, features, target, row_count = 3, precision = 16):
        
    # Display transposed list of first few rows.
    
    format_string = '{:,.' + str ( precision ) + 'f}'

    print ( 'FEATURES:')
    pd.options.display.float_format = format_string.format       
    print ( data [ features ].head ( row_count ).transpose() )
    
    print ( '\nTARGETS:')        
    print ( data [ target ].head ( row_count ).transpose() )



#//////////////////////////////////////////////////////////////////////////
# Program entry point.
#/////////////////////////////////////////////////////////////////////////////

if __name__ == "__main__":

    print ()        
    main () 
   
#/////////////////////////////////////////////////////////////////////////////
