#/////////////////////////////////////////////////////////////////////////////
# Import libraries.
#/////////////////////////////////////////////////////////////////////////////

# Python imports.

import random
import os
import datetime
import sys
import copy

import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np

from sklearn.metrics  import log_loss
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics  import accuracy_score

# Application imports.

from numerai_constants import Constant


#/////////////////////////////////////////////////////////////////////////////
# Constants
#/////////////////////////////////////////////////////////////////////////////

pass

#/////////////////////////////////////////////////////////////////////////////
# Data structures
#/////////////////////////////////////////////////////////////////////////////

# Model

class Model ( object ):
    
    def __init__ ( self ):
    
        self.algorythm     = GradientBoostingRegressor ()
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

    new_line ()
    log ( "PROGRAM: " + Constant.PROGRAM_FILE_NAME )    
    
    #-------------------------------------------------------------------------
    # Train model.
    #-------------------------------------------------------------------------
        
    # Load training data.
    
    log ( "TRAINING:" )
    log ( Constant.INDENT + "Loading training data: " + " \"" + Constant.DATA_FILE_TRAINING + "\""  )    
    
    x_train, y_train = load_training_data ( Constant.DATA_FILE_TRAINING )
    
    # Train model on training data.
    
    log ( Constant.INDENT + "Training " + str ( Constant.TRAINING_MODEL_COUNT ) + " model/s." )
    
    model = train_best_model ( x_train, y_train, Constant.TRAINING_MODEL_COUNT )
    
    
    #-------------------------------------------------------------------------
    # Apply model.
    #-------------------------------------------------------------------------
        
    # Load application data.
    
    log ( "APPLICATION:" )
    log ( Constant.INDENT + "Loading application data: " + " \"" + Constant.DATA_FILE_APPLICATION + "\""  )    
    
    x_application, data_application = load_application_data ( Constant.DATA_FILE_APPLICATION )
    
    # Apply model to application data.
    
    log ( Constant.INDENT + "Predicting results." )   
    
    y_application = model.algorythm.predict ( x_application )
    
    
    #-------------------------------------------------------------------------
    # Save results.
    #-------------------------------------------------------------------------
    
    log ( Constant.INDENT + "Saving results: " + " \"" + Constant.DATA_FILE_PREDICTION + "\""  )
    
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

def load_training_data ( data_file_training ):
    
    # Load training data from file.

    data_training = pd.read_csv ( data_file_training )
    
    # Prepare data for training. y_train = f ( x_train )
    # - Input vector = x_train
    # - Output vector = y_train
    
    x_train = data_training.drop ( Constant.CSV_TRAINING_TARGET, axis = 1 )
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
    
    x_application = data_application.drop ( Constant.CSV_APPLICATION_ID, axis = 1 ) 
    
    return x_application, data_application


#-----------------------------------------------------------------------------
# Save application results.
#-----------------------------------------------------------------------------

def save_application_results ( data_application, y_application ):

    data_application [ Constant.CSV_APPLICATION_PROBABILITY ] = y_application
    #data_application [ Constant.CSV_APPLICATION_PROBABILITY ] = y_application [ :, 1 ]
    
    # Save the results to file.    
    
    data_application.to_csv (
        Constant.DATA_FILE_PREDICTION, 
        columns = ( Constant.CSV_APPLICATION_ID, Constant.CSV_APPLICATION_PROBABILITY ), 
        index   = None
    )

#-----------------------------------------------------------------------------
# Train model.
#-----------------------------------------------------------------------------

def train_new_model ( x_train, y_train ):
    
    # Local constants.
    
    Constant.RANDOM_MIN = 0
    Constant.RANDOM_MAX = 1000
    
    # Initialize model.
    
    algorythm = GradientBoostingRegressor (                                                
                max_features      = Constant.GBR_MAX_FEATURES,
                #min_samples_split = Constant.GBR_MIN_SAMPLES_SPLIT,
                n_estimators      = Constant.GBR_N_ESTIMATORS,
                max_depth         = Constant.GBR_MAX_DEPTH,
                learning_rate     = Constant.GBR_LEARNING_RATE,
                #subsample         = Constant.GBR_SUBSAMPLE,
                random_state      = random.randint ( Constant.RANDOM_MIN, Constant.RANDOM_MAX ),
                warm_start        = Constant.GBR_WARM_START,
                verbose           = Constant.GBR_VERBOSE
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
    model.algorythm     = algorythm    
    model.log_loss      = training_log_loss
    model.accuracy      = tranining_accuracy
    model.training_time = elapsed_time

    return model


#-----------------------------------------------------------------------------
# Train best model out of N.
#-----------------------------------------------------------------------------

def train_best_model ( x_train, y_train, count ):
    
    # Local constants.    
    
    Constant.LOG_LOSS_MAX          = sys.maxsize
    Constant.INDENT                = "    "
    Constant.TABLE_HEADER          = "MODEL_INDEX\t   LOG_LOSS\t   ACCURACY\t   TRAINING_TIME"
    
    # Local variables.
    
    model_best          = Model ()
    model_best.log_loss = Constant.LOG_LOSS_MAX
    
    # Start clock
        
    clock_start = datetime.datetime.now() 
    
    # Begin training sequence.
                         
    log ( Constant.INDENT + Constant.TABLE_HEADER )        
    
    for training_cycle in range ( 0, count ):
        
        log ( Constant.INDENT + str ( training_cycle + 1 ) + "/" + str ( count ), newline = False )    
        
        # Train model.
        
        model = train_new_model ( x_train, y_train )
                        
        # Update best model.
        
        if model.log_loss < model_best.log_loss:            
            model_best = copy.copy ( model )
        
        # Report this training cycles' results.
                
        s =  "\t\t" + "{0:.5f}".format ( model.log_loss )        
        s += "\t\t"   + "{0:.3f}".format ( model.accuracy ) + " %"        
        s += "\t\t" + time_to_string   ( model.training_time )    
        print ( s )
    
    # Update best model.
      
    model = copy.copy ( model_best )
            
    # Stop clock.    
    
    clock_stop   = datetime.datetime.now () 
    elapsed_time = clock_stop - clock_start
        
    # Return training record.
        
    model.training_time = elapsed_time

    return model


#-----------------------------------------------------------------------------
# Compute log loss.
#-----------------------------------------------------------------------------

def compute_training_performance ( algorythm, x_train, y_train ):
  
    # Retrieve comparison criteria.
  
    y_true = y_train
    y_pred = algorythm.predict ( x_train ) 
    
    # Convert continuous probability predictions, to binary integer predictions.
    
    y_pred_binary = [ round ( prediction ) for prediction in y_pred ]
    
    # Compute logorythmic loss and accuracy.

    tranining_log_loss = log_loss ( y_true, y_pred )   
    tranining_accuracy = accuracy_score ( y_true, y_pred_binary ) * 100.0    
    
    # Return results.

    return tranining_log_loss, tranining_accuracy


#-----------------------------------------------------------------------------
# Plot data.
#-----------------------------------------------------------------------------

def console_report ( input_model, x_train, y_train ):
    
    # Local constants.
    
    Constant.INDENT = "  "
    
    # Collect data to report on.
    
    model = Model ()
    model = copy.copy ( input_model )
    
    training_time         = model.training_time    
    best_log_loss         = model.log_loss
    accuracy              = model.accuracy
    average_training_time = training_time / Constant.TRAINING_MODEL_COUNT

    # Print reporting and analysis data.
    
    log ( Constant.INDENT + "Best log loss = " + "{0:.5f}".format ( best_log_loss ) )
    log ( Constant.INDENT + "Best accuracy = " + "{0:.1f}".format ( accuracy ) )
    log ( Constant.INDENT + "Training time = " + time_to_string ( average_training_time ) )
    
    if Constant.REPORT_MODEL_PARAMETERS_ENABLED:
        
        log ( Constant.INDENT + "MODEL:\n" )        
        
        model_parameters = model.algorythm.get_params()
        
        for key in model_parameters:
            
            parameter_str  = "%24s = " % key
            parameter_str += str ( model_parameters [ key ] )
            print ( parameter_str )
    
    # Write results to log file.
        
    if Constant.LOG_FILE_ENABLED:
        
        write_model_to_log_file ( model, Constant.LOG_FILE_NAME + Constant.LOG_FILE_PATH )
        

#-----------------------------------------------------------------------------
# Save current parameters and results to parameter log file.
#-----------------------------------------------------------------------------

def write_model_to_log_file ( input_model, log_file ):
    
    # Retieve data to save.
    
    model = Model ()
    model = copy.copy ( input_model )
    
    algorythm_parameters = model.algorythm.get_params()    
        
    # Write results to log file.
    
    log_file = open ( log_file, "a" )
    log_str = ""
    for key in algorythm_parameters:
        log_str += str ( algorythm_parameters [ key ] ) + ","
    log_str += time_to_string ( model.training_time ) + ","
    log_str += "{0:.5f}".format ( model.log_loss ) + "\n"
    log_file.write ( log_str )
    log_file.close()
    
            


#-----------------------------------------------------------------------------
# Plot data.
#-----------------------------------------------------------------------------

def plot_data ( input_model ):
    
    model = Model ()
    model = copy.copy ( input_model )
    
    if Constant.REPORT_FIGURE_FEATURE_RANK_ENABLED:
        
        # Collect data to plot.
        
        feature_count = 21
        #indices       = range ( 0, feature_count     )
        indices       = np.argsort ( model.algorythm.feature_importances_ )
        bar_width     = 0.75
        
        # Plot the feature importances of the forest
        
        plt.bar (
            np.arange ( feature_count ),
            model.algorythm.feature_importances_ [ indices ],
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

