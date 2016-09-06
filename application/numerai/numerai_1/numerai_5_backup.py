#/////////////////////////////////////////////////////////////////////////////
# Import libraries.
#/////////////////////////////////////////////////////////////////////////////

import random
import os
import datetime
import sys

import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np

from sklearn.metrics  import log_loss
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics  import accuracy_score


#/////////////////////////////////////////////////////////////////////////////
# Constants
#/////////////////////////////////////////////////////////////////////////////

# CSV file column names.

C_CSV_TRAINING_TARGET         = "target"
C_CSV_APPLICATION_ID          = "t_id"
C_CSV_APPLICATION_PROBABILITY = "probability"

# Data file path and file names.

C_FILE_PATH        = "../data/"
C_FILE_TRAINING    = C_FILE_PATH + "numerai_training_data.csv"
C_FILE_APPLICATION = C_FILE_PATH + "numerai_tournament_data.csv"
C_FILE_PREDICTION  = C_FILE_PATH + "predictions.csv"

# Program file name.

C_PROGRAM_FILE_NAME = os.path.basename ( __file__ )  

# Log file path and file name.

C_LOG_FILE_PATH = ""
C_LOG_FILE_NAME = C_PROGRAM_FILE_NAME [ :-2 ] + "log"

# Training settings.

C_TRAINING_MODEL_COUNT = 2

# Algorythms: GradientBoostingRegressor

C_GBR_MAX_FEATURES      = 21
C_GBR_MAX_DEPTH         = 2
C_GBR_N_ESTIMATORS      = 8
C_GBR_LEARNING_RATE     = 0.1
C_GBR_WARM_START        = False
C_GBR_SUBSAMPLE         = 1.0
C_GBR_MIN_SAMPLES_SPLIT = 2
C_GBR_VERBOSE           = 0


# Reporting settings.

C_REPORT_MODEL_PARAMETERS_ENABLED    = True
C_REPORT_FIGURE_FEATURE_RANK_ENABLED = True
C_LOG_FILE_ENABLED                   = True

#/////////////////////////////////////////////////////////////////////////////
# Data structures
#/////////////////////////////////////////////////////////////////////////////

class Model ( object ):
    def __init__(self):
        self.algorythm     = GradientBoostingRegressor()
        self.log_loss      = 1.0
        self.accuracy      = 1.0
        self.training_time = datetime.timedelta ()

#/////////////////////////////////////////////////////////////////////////////
# Functions.
#/////////////////////////////////////////////////////////////////////////////

#-----------------------------------------------------------------------------
# Main Program.
#-----------------------------------------------------------------------------

def main ():

    C_INDENT = "  "

    new_line ()
    log ( "PROGRAM: " + C_PROGRAM_FILE_NAME )    
    
    #-------------------------------------------------------------------------
    # Train model.
    #-------------------------------------------------------------------------
        
    # Load training data.
    
    log ( "TRAINING:" )
    log ( C_INDENT + "Loading training data: " + " \"" + C_FILE_TRAINING + "\""  )    
    
    x_train, y_train = load_training_data ( C_FILE_TRAINING )
    
    # Train model on training data.
    
    log ( C_INDENT + "Training model." )
    
    model = train_best_model ( x_train, y_train, C_TRAINING_MODEL_COUNT )
    
    
    #-------------------------------------------------------------------------
    # Apply model.
    #-------------------------------------------------------------------------
        
    # Load application data.
    
    log ( "APPLICATION:" )
    log ( C_INDENT + "Loading application data: " + " \"" + C_FILE_APPLICATION + "\""  )    
    
    x_application, data_application = load_application_data ( C_FILE_APPLICATION )
    
    # Apply model to application data.
    
    log ( C_INDENT + "Predicting results." )   
    
    y_application = model.algorythm.predict ( x_application )
    
    
    #-------------------------------------------------------------------------
    # Save results.
    #-------------------------------------------------------------------------
    
    log ( C_INDENT + "Saving results: " + " \"" + C_FILE_PREDICTION + "\""  )
    
    save_application_results ( data_application, y_application )
    
    
    #-------------------------------------------------------------------------
    # Analysis and Reporting.
    #-------------------------------------------------------------------------
    
    log ( "REPORTING:" )
    
    console_report ( model, x_train, y_train )
    
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
        C_FILE_PREDICTION, 
        columns = ( C_CSV_APPLICATION_ID, C_CSV_APPLICATION_PROBABILITY ), 
        index   = None
    )

#-----------------------------------------------------------------------------
# Train model.
#-----------------------------------------------------------------------------

def train_new_model ( x_train, y_train ):
    
    # Local constants.
    
    C_RANDOM_MIN = 0
    C_RANDOM_MAX = 1000
    
    # Initialize model.
    
    algorythm = GradientBoostingRegressor (                                                
                max_features      = C_GBR_MAX_FEATURES,
                min_samples_split = C_GBR_MIN_SAMPLES_SPLIT,
                n_estimators      = C_GBR_N_ESTIMATORS,
                max_depth         = C_GBR_MAX_DEPTH,
                learning_rate     = C_GBR_LEARNING_RATE,
                subsample         = C_GBR_SUBSAMPLE,
                random_state      = random.randint ( C_RANDOM_MIN, C_RANDOM_MAX ),
                warm_start        = C_GBR_WARM_START,
                verbose           = C_GBR_VERBOSE
            )
            
    # Start clock
        
    clock_start = datetime.datetime.now() 
            
    # Train model.
            
    algorythm.fit ( x_train, y_train )
    
    # Test model and compute performance characteristics.
        
    training_log_loss, tranining_accuracy = compute_training_performance ( algorythm, x_train, y_train )
    
    # Stop clock.    
    
    clock_stop   = datetime.datetime.now () 
    elapsed_time = clock_stop - clock_start
    
    # Compile training record and return training results.
    
    model               = Model()
    model.model         = model    
    model.log_loss      = training_log_loss
    model.accuracy      = tranining_accuracy
    model.training_time = elapsed_time

    return model


#-----------------------------------------------------------------------------
# Train best model out of N.
#-----------------------------------------------------------------------------

def train_best_model ( x_train, y_train, count ):
    
    # Local constants.    
    
    C_LOG_LOSS_MAX          = sys.maxsize
    C_INDENT                = "    "
    C_TABLE_HEADER          = "MODEL_INDEX\tLOG_LOSS\tACCURACY\tTRAINING_TIME"
    C_TABLE_HORIZONTAL_LINE = "-------------\t-------------\t-------------\t-------------" 
    
    # Local variables.
    
    training_record_best          = TrainingRecord ()
    training_record_best.log_loss = C_LOG_LOSS_MAX
    
    # Start clock
        
    clock_start = datetime.datetime.now() 
    
    # Begin training sequence.
                     
    log ( C_INDENT + C_TABLE_HORIZONTAL_LINE )    
    log ( C_INDENT + C_TABLE_HEADER )    
    log ( C_INDENT + C_TABLE_HORIZONTAL_LINE )    
    
    for training_cycle in range ( 0, count ):
        
        log ( C_INDENT + str ( training_cycle + 1 ) + "/" + str ( count ), newline = False )    
        
        # Train model.
        
        training_record = train_new_model ( x_train, y_train )
                        
        # Update best model.
        
        if training_record.log_loss < training_record_best.log_loss:            
            training_record_best = training_record
        
        # Report this training cycles' results.
                
        s =  "\t\t" + "{0:.5f}".format ( training_record.log_loss )        
        s =  "\t\t" + "{0:.1f}".format ( training_record.accuracy )        
        s += "\t\t" + time_to_string   ( training_record.training_time )    
        print ( s )
    
    log ( C_INDENT + C_TABLE_HORIZONTAL_LINE )    
    
    # Update best model.
      
    training_record = training_record_best
            
    # Stop clock.    
    
    clock_stop   = datetime.datetime.now () 
    elapsed_time = clock_stop - clock_start
        
    # Return training record.
        
    training_record.training_time = elapsed_time

    return training_record


#-----------------------------------------------------------------------------
# Compute log loss.
#-----------------------------------------------------------------------------

def compute_training_performance ( model, x_train, y_train ):
  
    # Retrieve comparison criteria.
  
    y_true = y_train
    y_pred = model.predict ( x_train )
    
    # Compute logorythmic loss and accuracy.

    tranining_log_loss = log_loss       ( y_true, y_pred )   
    tranining_accuracy = accuracy_score ( y_true, y_pred ) * 100.0    
    
    # Return results.

    return tranining_log_loss, tranining_accuracy


#-----------------------------------------------------------------------------
# Plot data.
#-----------------------------------------------------------------------------

def console_report ( input_model, x_train, y_train, training_time ):
    
    C_INDENT = "  "
    
    # Collect data to report on.
    
    model = TrainingRecord ( input_model )
    
    best_log_loss         = model.log_loss
    average_training_time = training_time / C_TRAINING_MODEL_COUNT

    # Print reporting and analysis data.
    
    log ( C_INDENT + "Best log loss = " + "{0:.5f}".format ( best_log_loss ) )
    log ( C_INDENT + "Training time = " + time_to_string ( average_training_time ) )
    
    if C_REPORT_MODEL_PARAMETERS_ENABLED:
        
        log ( C_INDENT + "MODEL:\n" )        
        
        model_parameters = model.get_params()
        
        for key in model_parameters:
            
            parameter_str  = "%24s = " % key
            parameter_str += str ( model_parameters [ key ] )
            print ( parameter_str )
    
    # Write results to log file.
        
    if C_LOG_FILE_ENABLED:
        
        write_model_to_log_file ( model, C_LOG_FILE_NAME + C_LOG_FILE_PATH )
        

#-----------------------------------------------------------------------------
# Save current parameters and results to parameter log file.
#-----------------------------------------------------------------------------

def write_model_to_log_file ( model, log_file ):
    
    # Retieve data to save.
    
    model_parameters = model.get_params()
    
        
    # Write results to log file.
    
    log_file = open ( log_file, "a" )
    log_str = ""
    for key in model_parameters:
        log_str += str ( model_parameters [ key ] ) + ","
    log_str += time_to_string ( average_training_time ) + ","
    log_str += "{0:.5f}".format ( best_log_loss ) + "\n"
    log_file.write ( log_str )
    log_file.close()
    
            


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







