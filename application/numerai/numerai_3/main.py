import winsound
import datetime

from numerai_constants import Constant
from application_3     import initialize_model
from application_3     import load_training_data
from application_3     import train_model
from application_3     import apply_model
from optimize_model    import optimize_model_parameters
from utility           import log


#/////////////////////////////////////////////////////////////////////////////
# Program entry point.
#/////////////////////////////////////////////////////////////////////////////

def main():
    
    # Initialize application.    
    
    log ( 'PROGRAM.START: ' + str ( datetime.datetime.now() ) )
    winsound.Beep ( Constant.Sound.START_FREQUENCY, Constant.Sound.START_PERIOD )

    # Initialize loal variables.
    
    training_file_name    = Constant.Numerai.DataFile.PATH + Constant.Numerai.DataFile.TRAINING    
    row_count             = Constant.Model.TRAINING_DATA_LIMIT
        
    # Train and optimize model.
    
    model = initialize_model ()    
    x, t  = load_training_data        ( training_file_name, row_count )     
    model = optimize_model_parameters ( model, x, t )
    model = train_model               ( model, x, t )
    
    # Apply model.    
    
    apply_model ( model )
    
    # Shut down application.
        
    log ( 'PROGRAM.STOP: ' + str ( datetime.datetime.now() ) )
    print ('')
    winsound.Beep ( Constant.Sound.STOP_FREQUENCY, Constant.Sound.STOP_PERIOD )    
    
    #debug_show_sample_data ( training_data, cols_features, col_target, row_count = 3, precision = 16 )

if __name__ == "__main__":

     main()

