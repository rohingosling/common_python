#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Library imports.

import winsound
import time
import os

import numpy as np

# application imports.

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: initialize application
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def initialize ():
    
    print ( '\nInitialize Program.' )
    print ( '' )
    sound ( 11000, 50, 1 )
    
    
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
# FUNCTION: sound.
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def sound ( f, p, n  ):
    for i in range ( n ):
        winsound.Beep ( f, p )

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: main
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def main ():
    
    # Initialize constants.
    
    # Initialize program.
    
    initialize ();  
    
    # Srart clock.    
    
    time_start = time.time()     
        
    # Execute tests.
        
    TEXT_SYSTEMS = '[SYSTEM] '
        
    print ( TEXT_SYSTEMS + 'Initialize TensorFlow data.' )
    
    node1 = tf.constant ( 1.0 )
    node2 = tf.constant ( 2.0 )
    node3 = tf.add ( node1, node2 )
    
    x = [ node1, node2, node3 ]
    
    print ( TEXT_SYSTEMS + 'Initialize and Execute TensorFlow Session.' )
    
    sess = tf.Session()
    sess.run ( x )
    
    print ( TEXT_SYSTEMS + 'Report Results.' )    
    
    print ( '' )
    for node in x:
        print ( 'Node : ' + str ( node ) )    
    print ( '' )
    
    sess.close()
    
    # Stop clock.
    
    time_stop    = time.time()
    elapsed_time = time_stop - time_start
    
    # Report results.
    
    console_report_time ( time_start, time_stop, elapsed_time )
    print ( '' )


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Program entry point.    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    
main ()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------