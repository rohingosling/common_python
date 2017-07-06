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
import sys

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

from optimize_model     import optimize_model_parameters

from utility            import print_data_frame
from utility            import print_model_parameters
from utility            import log
from utility            import report_training_results

# Platform configuration.

rcParams [ 'figure.figsize' ] = 12, 4

#/////////////////////////////////////////////////////////////////////////////
# Functions.
#/////////////////////////////////////////////////////////////////////////////

#-----------------------------------------------------------------------------
# FUNCTION: initialize_model
#-----------------------------------------------------------------------------

def initialize_model ():
    
    log ( 'Initialize model: XGBClassifier.', indent = 0 )
    
    # Initialize model
    
    model = XGBClassifier (
    
        learning_rate    = Constant.Model.LEARNING_RATE,
        n_estimators     = Constant.Model.N_ESTIMATORS,
        max_depth        = Constant.Model.MAX_DEPTH,
        min_child_weight = Constant.Model.MIN_CHILD_WEIGHT,
        gamma            = Constant.Model.GAMMA,
        subsample        = Constant.Model.SUBSAMPLE,
        colsample_bytree = Constant.Model.COLSAMPLE_BYTREE,
        reg_alpha        = Constant.Model.REG_ALPHA,
        reg_lambda       = Constant.Model.REG_LAMBDA,
        objective        = Constant.Model.OBJECTIVE,
        scale_pos_weight = Constant.Model.SCALE_POS_WEIGHT,        
        seed             = Constant.Model.SEED     
    )
    
    # Print model parameters.
    
    print_model_parameters ( model, 1 )
    
    # Return initialized model.
    
    return model

#-----------------------------------------------------------------------------
# FUNCTION: Train model.
#-----------------------------------------------------------------------------

def train_model ( model, x, t ):
    
    # Local constants.
    
    INDENT = 1
    
    # Begin training sequence.
    
    log ( 'TRAINING SEQUENCE:' )    
    
    if Constant.Application.TRAINING_ENABLED:
        
        log ( 'Training model.', indent = INDENT+1 )
        
        # Train the model using current model parameters.
    
        model.fit ( x, t, eval_metric = Constant.Model.METRIC )
            
        # Compute training restuls.
            
        log ( 'Computing training results.', indent = INDENT+1 )
            
        t_predictions              = model.predict       ( x )
        t_prediction_probabilities = model.predict_proba ( x ) [ :, 1 ]
        
        # Print model report.
        
        report_training_results ( model, t, t_predictions, t_prediction_probabilities )
        feature_rank = pd.Series ( model.booster().get_fscore()).sort_values ( ascending = False )
        
        # Test Model.
    
        accuracy, auc, logloss = test_model ( model, x, t)
        
        # Plot feature ranking.
        
        if Constant.Application.PLOT_FEATURE_RANK_ENABLED:
            
            log ( 'REPORTING: Plotting feature rank.', indent = INDENT+1 )
            
            plot_feature_rank ( feature_rank )
            
        else:
            log ( 'Feature rank plot: DISABLED', indent = INDENT+1 )
        
    else:
        
        log ( 'Traning: DISABLED', indent = 1 )

    # Return trained model
    
    return model

#-----------------------------------------------------------------------------
# FUNCTION: test model..
#-----------------------------------------------------------------------------

def test_model ( model, x, t ):
    
    log ( 'Testing trained model.', indent = 1 )
    
    accuracy = 0
    auc      = 0
    logloss  = 0
    
    if Constant.Application.TEST_MODEL:
        
        pass        
        
    else:
        
        log ( 'Test Model: DISABLED', indent = 2 )        
    
    return accuracy, auc, logloss

#-----------------------------------------------------------------------------
# FUNCTION: apply_model
#-----------------------------------------------------------------------------

def apply_model ( model ):
    
    log ( 'APPLICATION SEQUENCE:' )
    
    if Constant.Application.MODEL_APPLICATION_ENABLED:
        
        # Compile file names.
        
        application_file_name = Constant.Numerai.DataFile.PATH + Constant.Numerai.DataFile.APPLICATION    
        prediction_file_name  = Constant.Numerai.DataFile.PATH + Constant.Numerai.DataFile.PREDICTION 
        
        # Load application data.    
        
        i, x = load_application_data ( application_file_name )
    
        # Apply model.
    
        y = predict ( model, x )
        
        # Save results.
        
        save_prediction_data ( prediction_file_name, i, y )
        
    else:
        
        log ( 'Application: DISABLED', indent = 1 )
        
        

#-----------------------------------------------------------------------------
# FUNCTION: test model..
#-----------------------------------------------------------------------------

def predict ( model, x ):
    
    log ( 'Applying model to production data.', indent = 1 )
    
    winsound.Beep ( Constant.Sound.START_FREQUENCY, Constant.Sound.START_PERIOD )
    
    y = model.predict_proba ( x )   
    
    return y

#-----------------------------------------------------------------------------
# FUNCTION: Load training data.
#-----------------------------------------------------------------------------

def load_training_data ( file_name, row_count = -1 ):
    
    log ( 'Loading training data: ' + '"' + file_name + '"', indent = 1 )
        
    # Load training data.
        
    training_data = pd.read_csv ( file_name )    
    
    # Reduce training data samples for configuration testing.
    
    if row_count > 0:    
        log ( 'Traning Data: training_data_limit_enabled = TRUE', indent = 2 )
        training_data = training_data.head ( row_count - 1 )
    
    log ( 'Traning Data: row_count = ' + str ( len ( training_data.index ) + 1 ), indent = 2 )
    
    # Seperate training data columns into features x, and target/s t.
    
    col_features = [ f for f in list ( training_data ) if Constant.Numerai.CSV.FEATURE in f ]    
    col_target   = Constant.Numerai.CSV.TARGET
    
    x = training_data [ col_features ]
    t = training_data [ col_target   ]
    
    return x, t

#-----------------------------------------------------------------------------
# Load application data.
#-----------------------------------------------------------------------------

def load_application_data ( file_name ):
    
    log ( 'Loading application data: ' + '"' + file_name + '"', indent = 1 )
        
    # Load application data from file.

    application_data = pd.read_csv ( file_name )
    
    # Prepare data for execution. y_application = f ( x_application )
    # - Input vector = x_application
    # - Output vector = y_application ...To be allocated after model execution.
    
    features = [ f for f in list ( application_data ) if Constant.Numerai.CSV.FEATURE in f ]
    i        = application_data [ [ Constant.Numerai.CSV.ID ] ]
    x        = application_data [ features ]
    
    log ( 'Application Data: row_count = ' + str ( len ( application_data.index ) + 1 ), indent = 2 )
    
    return i, x

#-----------------------------------------------------------------------------
# Save application results.
#-----------------------------------------------------------------------------

def save_prediction_data ( file_name, i, y ):
    
    log ( 'Saving application data: ' + '"' + file_name + '"', indent = 1 )
    
    # Isolate propability of 1.0.

    p = pd.DataFrame ( y ) [ 1 ]
    
    # Create prediction data table.        
    
    prediction_data         = pd.concat ( [ i, p ], axis=1 )
    prediction_data.columns = [ Constant.Numerai.CSV.ID, Constant.Numerai.CSV.PROBABILITY ]
        
    # Save the results to file.    
    
    prediction_data.to_csv ( Constant.Numerai.DataFile.PATH + file_name, index = None )
        
    
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


#/////////////////////////////////////////////////////////////////////////////
# Program entry point.
#/////////////////////////////////////////////////////////////////////////////

def main():
    
    # Initialize application.    
    
    log ( 'PROGRAM.START: ' + str ( datetime.datetime.now() ) )
    winsound.Beep ( Constant.Sound.START_FREQUENCY, Constant.Sound.START_PERIOD )

    # Initialize loal variables.
    
    training_file_name = Constant.Numerai.DataFile.PATH + Constant.Numerai.DataFile.TRAINING 
    row_count          = Constant.Model.TRAINING_DATA_LIMIT
        
    # Train and optimize model.
    
    model = initialize_model ()    
    x, t  = load_training_data        ( training_file_name, row_count )     
    model = optimize_model_parameters ( model, x, t )
    model = train_model               ( model, x, t )
    
    # Apply model.    
    
    apply_model ( model )
    
    # Shut down application.
        
    log ( 'PROGRAM.STOP: ' + str ( datetime.datetime.now() ) )
    print ('')
    winsound.Beep ( Constant.Sound.STOP_FREQUENCY, Constant.Sound.STOP_PERIOD )    
    
    #debug_show_sample_data ( training_data, cols_features, col_target, row_count = 3, precision = 16 )

if __name__ == "__main__":

     main()
   
#/////////////////////////////////////////////////////////////////////////////
