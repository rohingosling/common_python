#-----------------------------------------------------------------------------
# Library: Utility
# Version: 1.1
# Author:  Rohin Gosling
#
# Description:
# 
#   General purpose utility functions.
#
#-----------------------------------------------------------------------------

import winsound
import time

from sklearn import cross_validation, metrics

#/////////////////////////////////////////////////////////////////////////////
# GLobal Variables
#/////////////////////////////////////////////////////////////////////////////

time_event = time.time()

#/////////////////////////////////////////////////////////////////////////////
# Functions.
#/////////////////////////////////////////////////////////////////////////////

#-----------------------------------------------------------------------------
# Function: console_log
#
# Description:
#
#   Logs a formatted message to the console.
#
# Parameters:
#
#   message:
#     Text string to write to the console.
#
#   indent:
#     Number of character spaces to indent by.
#
#   newline:
#     Omit charage return and new line if False.
#     Add a regular cariage return and new line after the string if True.
#
#   lines_before:
#     Number of lines to leave before wiring message to the console.
#
#   lines_after:
#     Number of lines to leave before wiring message to the console.
#
#   frequency:
#     Frequency in Hz, of the sound to play when a message is writen to
#     the console.
#
#-----------------------------------------------------------------------------

def console_log ( message, indent = 0, new_line = True, lines_before = 0, lines_after = 0, frequency = 0 ):
    
    # Loacal constants.
    
    INDENT_STRING = ' ' 
    SOUND_PERIOD  = 70
    
    # Compile time string.
    
    time_now               = time.time()
    elapsed_time_formatted = time_to_string ( time_now - time_event )
    
    # Compile indentation string.
    
    indent_string = INDENT_STRING * indent
    
    # Compile string line prefix.
    
    line_prefix = '[' + elapsed_time_formatted + '] '
    
    # Leave lines before
    
    for i in range ( 0, lines_before ):
        console_new_line ()
    
    # Print message.    
    
    if new_line:
        print ( line_prefix + indent_string + message )
    else:
        print ( line_prefix + indent_string + message, end = '' )
        
    # Leave lines after.        
    
    for i in range ( 0, lines_after ):
        console_new_line ()    
    
    # Play sound.
        
    if frequency > 0:
        winsound.Beep ( frequency, SOUND_PERIOD )

#---------------------------------------------------------------------------------------------------------------------------------------------------------------    
# FUNCTION: report_time
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def console_report_elapsed_time ( time_start, time_stop, elapsed_time ):

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
# FUNCTION: console_new_line
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def console_new_line ():
    print ( '' )

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

#-----------------------------------------------------------------------------
# Plot model data to console.
#-----------------------------------------------------------------------------

def report_training_results ( model, x, y ):
    
    # Local constants.    
    
    CONSOLE_ALIGN_KEY = '{0:.<24}'
    C_FREQUENCY       = 100
    C_TEXT_MODEL      = '[REPORTING] '

    # Compute metrics.

    y_predictions              = model.predict       ( x )
    y_prediction_probabilities = model.predict_proba ( x ) [ :, 1 ]
        
    accuracy = metrics.accuracy_score ( y.values, y_predictions              ) * 100
    auc      = metrics.roc_auc_score  ( y,        y_prediction_probabilities )
    logloss  = metrics.log_loss       ( y,        y_prediction_probabilities )
    
    # Compile display strings.
    
    message_accuracy = C_TEXT_MODEL + CONSOLE_ALIGN_KEY.format ( 'Accuracy' ) + ' = ' + '{:.3f}'.format ( accuracy )
    message_auc      = C_TEXT_MODEL + CONSOLE_ALIGN_KEY.format ( 'AUC' )      + ' = ' + '{:.6f}'.format ( auc )
    message_logloss  = C_TEXT_MODEL + CONSOLE_ALIGN_KEY.format ( 'logloss' )  + ' = ' + '{:.6f}'.format ( logloss )
    
    # Write display strings to the console.
    
    console_new_line()
    
    console_log ( str ( message_accuracy ), indent = 0, frequency = C_FREQUENCY )
    console_log ( str ( message_auc ),      indent = 0, frequency = C_FREQUENCY )
    console_log ( str ( message_logloss ),  indent = 0, frequency = C_FREQUENCY )

#---------------------------------------------------------------------------------------------------------------------------------------------------------------    
# FUNCTION: sound.
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def sound ( f, p, n  ):
    for i in range ( n ):
        winsound.Beep ( f, p )
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------