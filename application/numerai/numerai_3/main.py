import os
import time

import numpy as np

from numerai_constants               import Constant
from utility                         import console_log, console_new_line, console_report_elapsed_time
from solution_3_neural_network_array import Application

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION: initialize application
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def initialize ():
    
    console_log ( Constant.Text.SYSTEM + 'Initialize Program.', lines_before = 1, frequency = Constant.Sound.LOG_FREQUENCY )
    
    # Disable warnings.
    # - We do this specificaly to disable TensorFlow warnings.
    
    if Constant.System.TENSOR_FLOW_WARNINGS_ENABLED == False:
        os.environ [ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'
    
    # Initialize random number generaor.
    
    np.random.seed ( 0 )

#/////////////////////////////////////////////////////////////////////////////
# Program entry point.
#/////////////////////////////////////////////////////////////////////////////

def main():
    
    # Initialize program.
    
    initialize ();  

    # Srart clock.    
    
    time_start = time.time()

    # Run application
    
    application = Application ()
    application.run()
      
    # Stop clock.
    
    time_stop    = time.time()
    elapsed_time = time_stop - time_start
   
    # Report time..
    
    console_new_line ()
    console_report_elapsed_time ( time_start, time_stop, elapsed_time )
    

if __name__ == "__main__":

     main()