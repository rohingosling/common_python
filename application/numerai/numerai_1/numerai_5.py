#/////////////////////////////////////////////////////////////////////////////
# Import libraries.
#/////////////////////////////////////////////////////////////////////////////

import random
import os

import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np

from sklearn.metrics      import log_loss
from sklearn.ensemble     import GradientBoostingRegressor


#/////////////////////////////////////////////////////////////////////////////
# Constants
#/////////////////////////////////////////////////////////////////////////////

# Strings.

C_NUMERAI = "[NUMERAI]: "

# CSV file column names.

C_CSV_TRAINING_TARGET         = "target"
C_CSV_APPLICATION_ID          = "t_id"
C_CSV_APPLICATION_PROBABILITY = "probability"

# Data file path and file names.

FILE_PATH        = "../data/"
FILE_TRAINING    = FILE_PATH + "numerai_training_data.csv"
FILE_APPLICATION = FILE_PATH + "numerai_tournament_data.csv"
FILE_PREDICTION  = FILE_PATH + "predictions.csv"

# Training settings.

C_TRAINING_MODEL_COUNT = 1

# Algorythms: GradientBoostingRegressor

C_GBR_LEARNING_RATE = 0.01        # Best: 0.01
C_GBR_MAX_FEATURES  = 21           # Best: 14
C_GBR_N_ESTIMATORS  = 256          # Best: 256
C_GBR_MAX_DEPTH     = 8            # Best: 5
C_GBR_WARM_START    = False        # Best: False
#C_GBR_LOSS          = "quantile"
C_GBR_LOSS          = "huber"
C_GBR_ALPHA         = 0.1
C_GBR_VERBOSE       = 0

# Reporting settings.

C_REPORT_MODEL_PARAMETERS_ENABLED    = True
C_REPORT_FIGURE_FEATURE_RANK_ENABLED = True


#/////////////////////////////////////////////////////////////////////////////
# Functions.
#/////////////////////////////////////////////////////////////////////////////

#-----------------------------------------------------------------------------
# Main Program.
#-----------------------------------------------------------------------------

def main ():
    
    new_line ()
    log ( "PROGRAM: " + os.path.basename(__file__) )
    
    #-------------------------------------------------------------------------
    # Train model.
    #-------------------------------------------------------------------------
        
    # Load training data.
    
    log ( "TRAINING:" )
    log ( "- Loading training data: " + " \"" + FILE_TRAINING + "\""  )    
    
    x_train, y_train = load_training_data ( FILE_TRAINING )
    
    # Train model on training data.
    
    log ( "- Training model." )
    
    model = train_best_model ( x_train, y_train, C_TRAINING_MODEL_COUNT )
    
    
    #-------------------------------------------------------------------------
    # Apply model.
    #-------------------------------------------------------------------------
        
    # Load application data.
    
    log ( "APPLICATION:" )
    log ( "- Loading application data: " + " \"" + FILE_APPLICATION + "\""  )    
    
    x_application, data_application = load_application_data ( FILE_APPLICATION )
    
    # Apply model to application data.
    
    log ( "- Predicting results." )   
    
    y_application = model.predict ( x_application )
    
    
    #-------------------------------------------------------------------------
    # Save results.
    #-------------------------------------------------------------------------
    
    log ( "- Saving results: " + " \"" + FILE_PREDICTION + "\""  )
    
    save_application_results ( data_application, y_application )
    
    
    #-----------------------------------------------------------------------------
    # Analysis and Reporting.
    #-----------------------------------------------------------------------------
    
    log ( "REPORTING:" )
    
    console_report ( model, x_train, y_train)
    
    plot_data ( model )
    

#-----------------------------------------------------------------------------
# Console logging functions.
#-----------------------------------------------------------------------------

def log ( message, newline = True ):
    if newline:
        print ( C_NUMERAI + message )
    else:
        print ( C_NUMERAI + message, end="" )

#-----------------------------------------------------------------------------

def new_line():
    print ( "" )


#-----------------------------------------------------------------------------
# Load training data.
#-----------------------------------------------------------------------------

def load_training_data ( FILE_TRAINING ):
    
    # Load training data from file.

    data_training = pd.read_csv ( FILE_TRAINING )
    
    # Prepare data for training. y_train = f ( x_train )
    # - Input vector = x_train
    # - Output vector = y_train
    
    x_train = data_training.drop ( C_CSV_TRAINING_TARGET, axis = 1 )
    y_train = data_training.target.values                           
    
    return x_train, y_train 


#-----------------------------------------------------------------------------
# Load application data.
#-----------------------------------------------------------------------------

def load_application_data ( FILE_APPLICATION ):
    
    # Load application data from file.

    data_application = pd.read_csv ( FILE_APPLICATION )
    
    # Prepare data for execution. y_application = f ( x_application )
    # - Input vector = x_application
    # - Output vector = y_application ...To be allocated after model execution.
    
    x_application = data_application.drop ( C_CSV_APPLICATION_ID, axis = 1 ) 
    
    return x_application, data_application


#-----------------------------------------------------------------------------
# Save application results.
#-----------------------------------------------------------------------------

def save_application_results ( data_application, y_application ):

    data_application [ C_CSV_APPLICATION_PROBABILITY ] = y_application
    #data_application [ C_CSV_APPLICATION_PROBABILITY ] = y_application [ :, 1 ]
    
    # Save the results to file.    
    
    data_application.to_csv (
        FILE_PREDICTION, 
        columns = ( C_CSV_APPLICATION_ID, C_CSV_APPLICATION_PROBABILITY ), 
        index   = None
    )

#-----------------------------------------------------------------------------
# Train model.
#-----------------------------------------------------------------------------

def train_model ( x_train, y_train ):
    
    model = GradientBoostingRegressor (                
                #loss          = C_GBR_LOSS,
                #alpha         = C_GBR_ALPHA,
                learning_rate = C_GBR_LEARNING_RATE,
                max_features  = C_GBR_MAX_FEATURES,
                n_estimators  = C_GBR_N_ESTIMATORS,
                max_depth     = C_GBR_MAX_DEPTH,
                random_state  = random.randint ( 0, 1000 ),
                warm_start    = C_GBR_WARM_START,
                verbose       = C_GBR_VERBOSE
            )
            
    model.fit ( x_train, y_train )

    return model


#-----------------------------------------------------------------------------
# Train best model out of N.
#-----------------------------------------------------------------------------

def train_best_model ( x_train, y_train, count ):
    
    training_cycle_count   = count
    log_loss_training_best = 1.0
    
    for training_cycle_index in range ( 0, training_cycle_count ):
        
        log ( "- - Model " + str ( training_cycle_index+1 ) + "/" + str ( training_cycle_count ) + ": ", newline = False )    
        
        # Train model.
        
        model = train_model ( x_train, y_train )
        
        # Test model and compute log loss.
        
        log_loss_traning = compute_log_loss ( model, x_train, y_train )
        
        # Update best model.
        
        if log_loss_traning < log_loss_training_best:
            log_loss_training_best = log_loss_traning
            model_best             = model
        
        # Report results.
    
        print ( "log_loss = " + "{0:.5f}".format ( log_loss_traning ) )
        
    # Update best model.
    
    model = model_best
    
    return model


#-----------------------------------------------------------------------------
# Compute log loss.
#-----------------------------------------------------------------------------

def compute_log_loss ( model, x_train, y_train ):
    
    y_true = y_train
    y_pred = model.predict ( x_train )

    return log_loss ( y_true, y_pred )


#-----------------------------------------------------------------------------
# Plot data.
#-----------------------------------------------------------------------------

def console_report ( model, x_train, y_train ):
    
    # Collect data to report on.
    
    best_log_loss = compute_log_loss ( model, x_train, y_train )

    # Print reporting and analysis data.
    
    log ( "- Best log loss = " + "{0:.5f}".format ( best_log_loss ) )
    
    if C_REPORT_MODEL_PARAMETERS_ENABLED:
        log ( "- MODEL:\n\n" + str ( model ) )


#-----------------------------------------------------------------------------
# Plot data.
#-----------------------------------------------------------------------------

def plot_data ( model ):
    
    if C_REPORT_FIGURE_FEATURE_RANK_ENABLED:
        
        # Collect data to plot.
        
        feature_count = 21
        #indices       = range ( 0, feature_count     )
        indices       = np.argsort ( model.feature_importances_ )
        bar_width     = 0.75
        
        # Plot the feature importances of the forest
        
        plt.bar (
            np.arange ( feature_count ),
            model.feature_importances_ [ indices ],
            bar_width,
            color = 'grey',
            align = 'center'
        )
        
        plt.title  ( "Feature Rank" )
        plt.ylabel ( "Relative Feature Rank" )
        plt.xlabel ( "Feature" )
        plt.xticks ( np.arange ( feature_count ) + bar_width/2.0, indices )
        
        plt.show ()
        
        
#/////////////////////////////////////////////////////////////////////////////
# Program entry point.
#/////////////////////////////////////////////////////////////////////////////

main()







