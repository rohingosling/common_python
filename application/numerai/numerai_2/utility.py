# Utility functions.

import winsound
import time

from numerai_constants import Constant
from sklearn           import cross_validation, metrics

#/////////////////////////////////////////////////////////////////////////////
# GLobal Variables
#/////////////////////////////////////////////////////////////////////////////

time_event = time.time()

#/////////////////////////////////////////////////////////////////////////////
# Functions.
#/////////////////////////////////////////////////////////////////////////////

#-----------------------------------------------------------------------------
# FUNCTION: log message to console.
#-----------------------------------------------------------------------------

def log ( message, indent = 0, newline = True, frequency = 0 ):
    
    # Loacal constants.
    
    INDENT_STRING = Constant.Text.INDENT
    
    # Compile time string.
    
    time_now               = time.time()
    elapsed_time_formatted = time_to_string ( time_now - time_event )
    
    # Compile indentation string.
    
    indent_string = INDENT_STRING * indent
    
    # Compile string line prefix.
    
    line_prefix = '[' + elapsed_time_formatted + '] '
    
    # Print message.    
    
    if newline:
        print ( line_prefix + indent_string + message )
    else:
        print ( line_prefix + indent_string + message, end = '' )
        
    # Play sound.
        
    if frequency == 0:
        winsound.Beep ( Constant.Sound.EVENT_FREQUENCY, Constant.Sound.EVENT_PERIOD )
    else:
        winsound.Beep ( frequency, 70 )

#-----------------------------------------------------------------------------
# FUNCTION: Time string.
#-----------------------------------------------------------------------------

def time_to_string ( time_sample ):
    
    hours, remainder = divmod ( time_sample, 3600 )
    minutes, seconds = divmod ( remainder,   60   )
    time_string      = '{:0>2}:{:0>2}:{:0>6.3f}'.format ( int ( hours ), int ( minutes ), seconds )    
    
    return time_string

#-----------------------------------------------------------------------------
# FUNCTION: Print model parameters.
#-----------------------------------------------------------------------------

def print_model_parameters ( model, indent ):

    for key in model.get_params():                   
        parameter_str  = '{0:.<24}'.format ( key )
        parameter_str += ' = '
        parameter_str += str ( model.get_params() [ key ] )
        log ( parameter_str, indent = indent )


#-----------------------------------------------------------------------------
# FUNCTION: Print Pandas data frame.
#-----------------------------------------------------------------------------

def print_data_frame ( data, indent ):

    # print header.
    
    r       = data.columns[0]
    n       = data.columns[1]
    t       = data.columns[2]
    logloss = data.columns[3]
    
    s = '{:<8}{:<8}{:<16}{:<16}'.format ( r, n, t, logloss )
    log ( s, indent )
    
    # Print data.
    
    for row in data.values:
        r       = row[0]
        n       = row[1] 
        t       = row[2]
        logloss = row[3]
        s = '{:<8}{:<8}{:<16}{:<16.6f}'.format ( r, int(n), t, logloss )
        log ( s, indent )

#-----------------------------------------------------------------------------
# Plot model data to console.
#-----------------------------------------------------------------------------

def report_training_results ( model, t, t_predictions, t_prediction_probabilities ):
    
    CONSOLE_ALIGN_KEY = '{0:.<24}'
    INDENT_HEADER     = 2
    INDENT_DATA       = 3
        
    accuracy = metrics.accuracy_score ( t.values, t_predictions              ) * 100
    auc      = metrics.roc_auc_score  ( t,        t_prediction_probabilities )
    logloss  = metrics.log_loss       ( t,        t_prediction_probabilities )
        
    log ( 'Training Results:', indent = INDENT_HEADER )
    log ( CONSOLE_ALIGN_KEY.format ( 'Accuracy' ) + ' = ' + '{:.3f}'.format ( accuracy ), indent = INDENT_DATA )
    log ( CONSOLE_ALIGN_KEY.format ( 'AUC' )      + ' = ' + '{:.6f}'.format ( auc ),      indent = INDENT_DATA )
    log ( CONSOLE_ALIGN_KEY.format ( 'logloss' )  + ' = ' + '{:.6f}'.format ( logloss ),  indent = INDENT_DATA )
