#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Library imports.

import winsound
import time
import os

import numpy      as np
import tensorflow as tf

# application imports.

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# GLOBAL CONSTANTS
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Text constants

C_TEXT_SYSTEM      = 'SYSTEM'
C_TEXT_APPLICATION = 'APPLICATION'

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: initialize application
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def initialize ():
        
    console_log ( C_TEXT_SYSTEM, 'Initialize Program.', True, False, True )
    sound ( 11000, 50, 1 )
    
    # Disable warnings.
    # - We do this specificaly to disable TensorFlow warnings.
    
    os.environ [ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------    
# FUNCTION: report_time
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def console_report_time ( time_start, time_stop, elapsed_time ):

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
# FUNCTION: console_log
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def console_log ( text_category, text, new_line_before, new_line_after, sound_enabled = False ):
    
    # Local constants.
    
    TEXT_CATEGORY_BRACKET_OPEN  = '['
    TEXT_CATEGORY_BRACKET_CLOSE = '] '
    
    # Local variables.
    
    frequency = 100
    period    = 50
    repeat    = 1
    
    # write text to console.
    
    if new_line_before:
        console_new_line ()
        
    if text_category != '':
        text_category_string = TEXT_CATEGORY_BRACKET_OPEN + text_category + TEXT_CATEGORY_BRACKET_CLOSE
        print ( text_category_string + text )
        
    if new_line_after:
        console_new_line ()
    
    if sound_enabled:
        sound ( frequency, period, repeat )

#---------------------------------------------------------------------------------------------------------------------------------------------------------------    
# FUNCTION: console_new_line
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def console_new_line ():
    print ( '' )
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------    
# FUNCTION: sound.
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def sound ( f, p, n  ):
    for i in range ( n ):
        winsound.Beep ( f, p )

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: main
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def main ():
    
    # Initialize program.
    
    initialize ();  
    
    # Srart clock.    
    
    time_start = time.time()     
        
    # Initialize TensorFlow data.
        
    console_log ( C_TEXT_SYSTEM, 'Initialize TensorFlow data.', False, False, True )
    
    node0 = tf.constant ( 1.0 )
    node1 = tf.constant ( 2.0 )
    node2 = tf.add ( node0, node1 )
        
    node_list = [ node0, node1, node2 ]
        
    # Initialize TensoreFlow session.
    
    console_log ( C_TEXT_SYSTEM, 'Initialize and Execute TensorFlow Session.', False, False, True )
    
    sess = tf.Session()
    sess.run ( node_list )
    
    # Report results.
    
    console_log ( C_TEXT_SYSTEM, 'Report Results.', False, False, True )
    
    console_new_line ()
    for node in node_list:
        node_text = 'Node: ' + str ( node )
        console_log ( C_TEXT_APPLICATION, node_text, False, False, True )
    console_new_line ()
    
    # Close TensorFlow session.
    
    console_log ( C_TEXT_SYSTEM, 'Close TensorFlow session.', False, True, True )    
    
    sess.close()
    
    # Stop clock.
    
    time_stop    = time.time()
    elapsed_time = time_stop - time_start
    
    # Report time..
    
    console_report_time ( time_start, time_stop, elapsed_time )
    console_new_line ()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Program entry point.    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    
if __name__ == '__main__':
    main()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------