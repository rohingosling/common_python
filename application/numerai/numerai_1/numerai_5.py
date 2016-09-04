#/////////////////////////////////////////////////////////////////////////////
# Import libraries.
#/////////////////////////////////////////////////////////////////////////////

import random
import os
#import time
import datetime

import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np

from sklearn.metrics      import log_loss
from sklearn.ensemble     import GradientBoostingRegressor


#/////////////////////////////////////////////////////////////////////////////
# Constants
#/////////////////////////////////////////////////////////////////////////////

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

C_TRAINING_MODEL_COUNT = 4

# Algorythms: GradientBoostingRegressor

C_GBR_MAX_FEATURES      = 9
C_GBR_MAX_DEPTH         = 3
C_GBR_N_ESTIMATORS      = 4096
C_GBR_LEARNING_RATE     = 0.001
C_GBR_WARM_START        = False
C_GBR_SUBSAMPLE         = 1.0
C_GBR_MIN_SAMPLES_SPLIT = 2
C_GBR_VERBOSE           = 0

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

    C_INDENT = "  "

    new_line ()
    log ( "PROGRAM: " + os.path.basename(__file__) )    
    
    #-------------------------------------------------------------------------
    # Train model.
    #-------------------------------------------------------------------------
        
    # Load training data.
    
    log ( "TRAINING:" )
    log ( C_INDENT + "Loading training data: " + " \"" + FILE_TRAINING + "\""  )    
    
    x_train, y_train = load_training_data ( FILE_TRAINING )
    
    # Train model on training data.
    
    log ( C_INDENT + "Training model." )
    
    model, training_time = train_best_model ( x_train, y_train, C_TRAINING_MODEL_COUNT )
    
    
    #-------------------------------------------------------------------------
    # Apply model.
    #-------------------------------------------------------------------------
        
    # Load application data.
    
    log ( "APPLICATION:" )
    log ( C_INDENT + "Loading application data: " + " \"" + FILE_APPLICATION + "\""  )    
    
    x_application, data_application = load_application_data ( FILE_APPLICATION )
    
    # Apply model to application data.
    
    log ( C_INDENT + "Predicting results." )   
    
    y_application = model.predict ( x_application )
    
    
    #-------------------------------------------------------------------------
    # Save results.
    #-------------------------------------------------------------------------
    
    log ( C_INDENT + "Saving results: " + " \"" + FILE_PREDICTION + "\""  )
    
    save_application_results ( data_application, y_application )
    
    
    #-------------------------------------------------------------------------
    # Analysis and Reporting.
    #-------------------------------------------------------------------------
    
    log ( "REPORTING:" )
    
    console_report ( model, x_train, y_train, training_time )
    
    plot_data ( model )
   

    #-------------------------------------------------------------------------
    # Analysis and Reporting.
    #-------------------------------------------------------------------------
    
    new_line ()
    log ( "End." )

#-----------------------------------------------------------------------------
# Console logging functions.
#-----------------------------------------------------------------------------

def log ( message, newline = True ):
    
    # Compile time string.
    
    ts = datetime.datetime.now().strftime ( "%H:%M:%S" )    

    # Print message.    
    
    if newline:
        print ( "[" + ts + "]: " + message )
    else:
        print ( "[" + ts + "]: " + message, end="" )

#-----------------------------------------------------------------------------

def new_line():
    print ( "" )

#-----------------------------------------------------------------------------    

def time_to_string ( time ):        
    
        ms_str   = str ( time )[-7:]
        ms       = round ( float (ms_str), 3 )
        ms_str   = "{0:.3f}".format ( ms )
        time_str = str ( time )[:-6] + ms_str[2:]

        return time_str


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
                max_features      = C_GBR_MAX_FEATURES,
                #min_samples_split = C_GBR_MIN_SAMPLES_SPLIT,
                n_estimators      = C_GBR_N_ESTIMATORS,
                max_depth         = C_GBR_MAX_DEPTH,
                learning_rate     = C_GBR_LEARNING_RATE,
                #subsample         = C_GBR_SUBSAMPLE,
                random_state      = random.randint ( 0, 1000 ),
                warm_start        = C_GBR_WARM_START,
                verbose           = C_GBR_VERBOSE
            )
            
    model.fit ( x_train, y_train )

    return model


#-----------------------------------------------------------------------------
# Train best model out of N.
#-----------------------------------------------------------------------------

def train_best_model ( x_train, y_train, count ):
    
    C_INDENT = "    "    
    
    training_cycle_count   = count
    log_loss_training_best = 1.0
    totaL_training_time    = datetime.timedelta ( 0 )
    
    log ( C_INDENT + "MODEL_INDEX\t   LOG_LOSS\t   TRAINING_TIME" )    
    
    for training_cycle_index in range ( 0, training_cycle_count ):
        
        log ( C_INDENT + str ( training_cycle_index+1 ) + "/" + str ( training_cycle_count ), newline = False )    
        
        # Start clock
        
        clock_start = datetime.datetime.now()        
        
        # Train model.
        
        model = train_model ( x_train, y_train )
        
        # Test model and compute log loss.
        
        log_loss_traning = compute_log_loss ( model, x_train, y_train )
                
        # Update best model.
        
        if log_loss_traning < log_loss_training_best:
            log_loss_training_best = log_loss_traning
            model_best             = model
            
        # stop clock, and compute elapsed time.
            
        clock_stop           = datetime.datetime.now() 
        elapsed_time         = clock_stop - clock_start
        totaL_training_time += elapsed_time       
        
        # Report results.
                
        s =  "\t\t" + "{0:.5f}".format ( log_loss_traning )        
        s += "\t\t" + time_to_string ( elapsed_time )
    
        print ( s )
        
    # Update best model.
    
    if (log_loss_traning < 1.0 ):
        model = model_best    
    
    return model, totaL_training_time


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

def console_report ( model, x_train, y_train, training_time ):
    
    C_INDENT = "  "
    
    # Collect data to report on.
    
    best_log_loss = compute_log_loss ( model, x_train, y_train )

    # Print reporting and analysis data.
    
    log ( C_INDENT + "Best log loss = " + "{0:.5f}".format ( best_log_loss ) )
    log ( C_INDENT + "Training time = " + time_to_string ( training_time ) )
    
    if C_REPORT_MODEL_PARAMETERS_ENABLED:
        
        log ( C_INDENT + "MODEL:\n" )        
        
        model_parameters = model.get_params()
        
        for key in model_parameters:
            
            log_str  = "%24s = " % key
            log_str += str ( model_parameters [ key ] )
            print ( log_str )    


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







