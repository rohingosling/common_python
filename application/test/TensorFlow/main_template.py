# libray imports

import winsound
import time
import numpy as np

# application imports.

# FUNCTION: sound.

def sound ( f, p, n  ):
    for i in range ( n ):
        winsound.Beep ( f, p )

# FUNCTION: main

def main ():
    
    # Initialize program.
    
    time_start = time.time() 
    print ( '\nInitialize Program.' )          
    sound ( 11000, 50, 1 )
    
    # Execut tests.
    
    pass
    
    # Print program data.
    
    time_stop    = time.time()
    elapsed_time = time_stop - time_start
    print ( 'Time.Start:   ' + time.strftime ( "%H:%M:%S", time.gmtime ( time_start ) ) )
    print ( 'Time.Stop:    ' + time.strftime ( "%H:%M:%S", time.gmtime ( time_stop ) ) )
    print ( 'Time.Elapsed: ' + time.strftime ( "%H:%M:%S", time.gmtime ( elapsed_time ) ) )
    # print ( 'Data Count:   ' + str ( len ( data ) ) )    
    sound ( 12000, 50, 2 )
    
    
# Program entry point.    
    
main ()