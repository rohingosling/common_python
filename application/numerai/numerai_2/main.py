import time

from utility import console_log, console_new_line

#/////////////////////////////////////////////////////////////////////////////
# Program entry point.
#/////////////////////////////////////////////////////////////////////////////

def main():
    
    # Initialize application.    
    
    console_log ( 'PROGRAM.START: ' + str ( time.now() ), indent = 0, lines_before = 1, frequency = 11000 )    

    
    # Shut down application.
    
    console_log ( 'PROGRAM.STOP: ' + str ( time.now() ), indent = 0, lines_before = 1, frequency = 11000 )            
    console_new_line ()    
    
    #debug_show_sample_data ( training_data, cols_features, col_target, row_count = 3, precision = 16 )

if __name__ == "__main__":

     main()