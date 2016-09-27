#/////////////////////////////////////////////////////////////////////////////
# Import libraries.
#/////////////////////////////////////////////////////////////////////////////

# Python imports.

import random
import os
import datetime
import sys
import copy
import winsound

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

C_PROGRAM_NAME         = "Numerai Model-5"
C_PROGRAM_VERSION      = "1.0"
C_PROGRAM_TITLE_STRING = C_PROGRAM_NAME + " ( Version " + str ( C_PROGRAM_VERSION ) + " )"

#/////////////////////////////////////////////////////////////////////////////
# Data structures
#/////////////////////////////////////////////////////////////////////////////

# Model

class Model ( object ):
    
    def __init__ ( self ):
    
        self.algorithm      = GradientBoostingRegressor ()
        self.log_loss       = 1.0
        self.accuracy       = 1.0
        self.training_time  = datetime.timedelta ( 0 )                
        self.algorithm_name = str ( self.algorithm.__class__ ) [ 8 : len ( str ( self.algorithm.__class__ ) ) - 2 ]         

#/////////////////////////////////////////////////////////////////////////////
# Functions.
#/////////////////////////////////////////////////////////////////////////////

#-----------------------------------------------------------------------------
# Main Program.
#-----------------------------------------------------------------------------

def main ():

    new_line ()
    log ( "PROGRAM: " + C_PROGRAM_TITLE_STRING )    
    
    #-------------------------------------------------------------------------
    # Train model.
    #-------------------------------------------------------------------------
        
    # Load training data.
    
    log ( "TRAINING:" )
    log ( Constant.INDENT + "Loading training data: " + " \"" + Constant.DataFile.PATH + Constant.DataFile.TRAINING + "\""  )    
    
    x_train, y_train = load_training_data ( Constant.DataFile.PATH + Constant.DataFile.TRAINING )
    
    # Train model on training data.
    
    log ( Constant.INDENT + "Training " + str ( Constant.TRAINING_MODEL_COUNT ) + " model/s." )
    
    model = train_best_model ( x_train, y_train, Constant.TRAINING_MODEL_COUNT )
    
    
    #-------------------------------------------------------------------------
    # Apply model.
    #-------------------------------------------------------------------------
        
    # Load application data.
    
    log ( "APPLICATION:" )
    log ( Constant.INDENT + "Loading application data: " + " \"" + Constant.DataFile.PATH + Constant.DataFile.APPLICATION + "\""  )    
    
    x_application, data_application = load_application_data ( Constant.DataFile.PATH + Constant.DataFile.APPLICATION )
    
    # Apply model to application data.
    
    log ( Constant.INDENT + "Predicting results." )   
                         
    y_application = model.algorithm.predict ( x_application )
    
    
    #-------------------------------------------------------------------------
    # Save results.
    #-------------------------------------------------------------------------
    
    log ( Constant.INDENT + "Saving results: " + " \"" + Constant.DataFile.PATH + Constant.DataFile.PREDICTION + "\""  )
    
    save_application_results ( data_application, y_application )

    
    
    #-------------------------------------------------------------------------
    # Analysis and Reporting.
    #-------------------------------------------------------------------------
    
    log ( "REPORTING:" )
    
    console_report ( model, x_train, y_train )
    
    plot_data ( model )
   

    #-------------------------------------------------------------------------
    # Shut-down program.
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
        
    winsound.Beep ( 12000, 70 )

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

def table_row ( data_list, col_width ):
    
    data             = list ( data_list )    
    table_col_format = '{0: <' + str (col_width) + '}'

    s = ""    
    for value in data:
        s += table_col_format.format ( str ( value ) )
        
    return s
        
    
#-----------------------------------------------------------------------------
# Load training data.
#-----------------------------------------------------------------------------

def load_training_data ( data_file_training ):
    
    # Load training data from file.

    data_training = pd.read_csv ( data_file_training )
    
    # Prepare data for training. y_train = f ( x_train )
    # - Input vector = x_train
    # - Output vector = y_train
    
    x_train = data_training.drop ( Constant.CSV.Header.TRAINING_TARGET, axis = 1 )
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
    
    x_application = data_application.drop ( Constant.CSV.Header.APPLICATION_ID, axis = 1 ) 
    
    return x_application, data_application


#-----------------------------------------------------------------------------
# Save application results.
#-----------------------------------------------------------------------------

def save_application_results ( data_application, y_application, file_name = Constant.DataFile.PREDICTION ):

    data_application [ Constant.CSV.Header.APPLICATION_PROBABILITY ] = y_application
    #data_application [ Constant.CSV_APPLICATION_PROBABILITY ] = y_application [ :, 1 ]
    
    # Save the results to file.    
    
    data_application.to_csv (
        Constant.DataFile.PATH + file_name, 
        columns = ( Constant.CSV.Header.APPLICATION_ID, Constant.CSV.Header.APPLICATION_PROBABILITY ), 
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
    
    model = Model()
    
    model.algorithm = GradientBoostingRegressor (                                                
                          max_features      = Constant.GBR_MAX_FEATURES,
                          min_samples_split = Constant.GBR_MIN_SAMPLES_SPLIT,
                          n_estimators      = Constant.GBR_N_ESTIMATORS,
                          max_depth         = Constant.GBR_MAX_DEPTH,
                          learning_rate     = Constant.GBR_LEARNING_RATE,
                          subsample         = Constant.GBR_SUBSAMPLE,
                          random_state      = random.randint ( Constant.RANDOM_MIN, Constant.RANDOM_MAX ),
                          warm_start        = Constant.GBR_WARM_START,
                          verbose           = Constant.GBR_VERBOSE
                      )
            
    # Start clock
   
    clock_start = datetime.datetime.now() 
            
    # Train model.
            
    model.algorithm.fit ( x_train, y_train )
    
    # Test model and compute performance characteristics.
        
    model.log_loss, model.accuracy = compute_training_performance ( model.algorithm, x_train, y_train )
    
    # Stop clock.    
    
    clock_stop          = datetime.datetime.now () 
    model.training_time = clock_stop - clock_start

    return model


#-----------------------------------------------------------------------------
# Train best model out of N.
#-----------------------------------------------------------------------------

def train_best_model ( x_train, y_train, count ):
    
    # Local constants.    
    
    LOG_LOSS_MAX     = sys.maxsize
    TABLE_INDENT     = 2 * Constant.INDENT
    TABLE_COL_WIDTH  = 14
        
    # Local variables.
    
    model_best          = Model ()
    model_best.log_loss = LOG_LOSS_MAX
    table_row_header    = [ 'MODEL_INDEX', 'LOG_LOSS', 'ACCURACY', 'TRAINING_TIME' ]
    table_row_data      = []
    
    # Start clock
        
    clock_start = datetime.datetime.now() 
    
    # Begin training sequence.
                         
    log ( TABLE_INDENT + table_row ( table_row_header, TABLE_COL_WIDTH ) )        
    
    for training_cycle in range ( 0, count ):
        
        # Train model.
        
        model = train_new_model ( x_train, y_train )
                        
        # Update best model.
        
        if model.log_loss < model_best.log_loss:            
            model_best = copy.copy ( model )
            
        # Report this training cycles' results.
        
        table_row_data.clear  ()
        table_row_data.append ( str ( training_cycle + 1 ) + "/" + str ( count ) )
        table_row_data.append ( "{0:.5f}".format ( model.log_loss )              )
        table_row_data.append ( "{0:.3f}".format ( model.accuracy ) + " %"       )
        table_row_data.append ( time_to_string   ( model.training_time )         )
        
        log ( TABLE_INDENT + table_row ( table_row_data, TABLE_COL_WIDTH ) )
    
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

def compute_training_performance ( algorithm, x_train, y_train ):
  
    # Retrieve comparison criteria.
  
    y_true = y_train
    y_pred = algorithm.predict ( x_train ) 
    
    # Convert continuous probability predictions, to binary integer predictions.
    
    y_pred_binary = [ round ( prediction ) for prediction in y_pred ]
    
    # Compute logarithmic loss and accuracy.

    tranining_log_loss = log_loss ( y_true, y_pred )   
    tranining_accuracy = accuracy_score ( y_true, y_pred_binary ) * 100.0    
    
    # Return results.

    return tranining_log_loss, tranining_accuracy


#-----------------------------------------------------------------------------
# Plot data.
#-----------------------------------------------------------------------------

def console_report ( input_model, x_train, y_train ):
    
    # Collect data to report on.
    
    model = Model ()
    model = copy.copy ( input_model )
    
    algorithm     = model.algorithm_name
    training_time = model.training_time    
    best_log_loss = model.log_loss
    accuracy      = model.accuracy

    # Print reporting and analysis data.
    
    log ( Constant.INDENT + "Algorithm     = " + algorithm )
    log ( Constant.INDENT + "Best log loss = " + "{0:.5f}".format ( best_log_loss ) )
    log ( Constant.INDENT + "Best accuracy = " + "{0:.1f}".format ( accuracy ) + " %" )
    log ( Constant.INDENT + "Training time = " + time_to_string   ( training_time ) )
    
    if Constant.REPORT_MODEL_PARAMETERS_ENABLED:
        
        log ( Constant.INDENT + "MODEL:\n" )        
        
        model_parameters = model.algorithm.get_params()
        
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
    
    # Local constants.
    
    DELIMITER    = ','
    NEW_LINE     = '\n'
    EMPTY_STRING = ''
    
    # Retieve data to save.
    
    # Retrieve model data.    
    
    model                = Model ()
    model                = copy.copy ( input_model )    
    algorithm_parameters = model.algorithm.get_params()   
    
    # Get time and date info.
    
    save_time_date = datetime.datetime.now  ()
    save_time      = datetime.datetime.time ( save_time_date )
    save_date      = datetime.datetime.date ( save_time_date )
    
    # Compile filename.
    
    if log_file == '-':
        log_file = model.algorithm_name.replace ( '.', '_' ) + ".log.csv"
    
    # Check to see of hte log file exists.
    
    if os.path.exists ( log_file ):
        file_exists = True
    else:
        file_exists = False
        
    # Open file.
        
    log_file = open ( log_file, Constant.File.APPEND )
    log_str  = EMPTY_STRING
        
    # Open file, or create file if it does not yet exist.
    
    if not file_exists:
        
        # Create a new file and add the CSV header.
        
        log_str += "date" + DELIMITER
        log_str += "time" + DELIMITER
        log_str += "training_time" + DELIMITER
        log_str += "log_loss" + DELIMITER
        log_str += "accuracy" + DELIMITER
        
        for key in algorithm_parameters:
            log_str += str ( key ) + DELIMITER
            
        log_str += "program_name" + DELIMITER
        log_str += "program_version" + DELIMITER
        log_str += "algorythm" + NEW_LINE
        
    # Write CSV data to file.
        
    log_str += str ( save_date ) + DELIMITER 
    log_str += str ( save_time ) + DELIMITER
    log_str += time_to_string ( model.training_time ) + DELIMITER
    log_str += "{0:.5f}".format ( model.log_loss ) + DELIMITER
    log_str += "{0:.1f}".format ( model.accuracy ) + DELIMITER
        
    for key in algorithm_parameters:
        log_str += str ( algorithm_parameters [ key ] ) + DELIMITER
            
    log_str += C_PROGRAM_NAME + DELIMITER
    log_str += C_PROGRAM_VERSION + DELIMITER
    log_str += str ( model.algorithm_name ) + NEW_LINE
    
    # Wrie CSV data and close file.

    log_file.write ( log_str )    
    log_file.close()
    
#-----------------------------------------------------------------------------
# Plot data.
#-----------------------------------------------------------------------------

def plot_data ( input_model ):
    
    # Local variables.
    
    font_size_text  = 6
    font_size_axies = 5
    color_bar       = '0.75'
    
    # Plot data.
    
    model = Model ()
    model = copy.copy ( input_model )
    
    if Constant.REPORT_FIGURE_FEATURE_RANK_ENABLED:
        
        # Collect data to plot.
        
        feature_count = 21
        #indices       = range ( 0, feature_count     )
        indices       = np.argsort ( model.algorithm.feature_importances_ )        
        y_unit        = 0.01
        index_max     = model.algorithm.feature_importances_.max()
        bar_width     = 0.66 * ( 1.0 / float ( feature_count ) )
        
        # Plot the feature importances of the forest
        
        plt.bar (
            left      = np.arange ( feature_count ) / ( feature_count - 1 ),
            height    = model.algorithm.feature_importances_ [ indices ],
            width     = bar_width,
            color     = color_bar,
            edgecolor ='none',
            align     = 'center'
        )
        
        plt.title  ( "Feature Rank",          fontsize = font_size_text )
        plt.ylabel ( "Relative Feature Rank", fontsize = font_size_text )
        plt.xlabel ( "Feature",               fontsize = font_size_text )
        plt.xticks ( np.arange ( feature_count ) / ( feature_count - 1.0 ), indices, fontsize = font_size_axies )        
        plt.yticks ( np.arange ( 0.0, index_max + y_unit, y_unit ), np.arange ( 0.0, index_max + y_unit, y_unit ), fontsize = font_size_axies )
        
        plt.xlim ( [ -bar_width, 1.0 + bar_width ] )
        
        #plt.grid ( True )
        #plt.axes().set_aspect ( 'equal' )        
        
        plt.show ()
        
        
#/////////////////////////////////////////////////////////////////////////////
# Program entry point.
#/////////////////////////////////////////////////////////////////////////////

main()

