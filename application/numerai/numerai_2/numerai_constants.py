#/////////////////////////////////////////////////////////////////////////////
# Constants
#/////////////////////////////////////////////////////////////////////////////
    
class Constant ( object ):

    # Text formating constans.

    class Text ( object):
        
        INDENT = "-   "
        
    class Sound ( object ):
        
        EVENT_PERIOD    = 70
        EVENT_FREQUENCY = 10000
        
        START_PERIOD    = 200
        START_FREQUENCY = 12000
        
        STOP_PERIOD    = 200
        STOP_FREQUENCY = 8000
    
    # File arguments.

    class File ( object):    
        READ              = 'r'
        WRITE             = 'w'
        APPEND            = 'a'
        READ_WRITE        = 'r+'
        BINARY_READ       = 'rb'
        BINARY_WRITE      = 'wb'
        BINARY_APPEND     = 'ab'
        BINARY_READ_WRITE = 'r+b'
    
    # Numerai specific constants
    
    class Numerai ( object ):
        
        # Data file path and file names.
        
        class DataFile ( object ):    
            PATH        = "../data/"
            TRAINING    = "numerai_training_data.csv"
            APPLICATION = "numerai_tournament_data.csv"
            PREDICTION  = "predictions.csv"
        
        # CSV file constants.        
        
        class CSV ( object ):
            FEATURE     = 'feature'
            TARGET      = 'target'
            ID          = 't_id'
            PROBABILITY = 'probability'


    # Log file constants.    
    
    class LogFile ( object ):    
        
        # Log file path and file name.
        
        LOG_FILE_PATH    = ''
        #LOG_FILE_NAME    = "numerai.log.csv"
        LOG_FILE_NAME    = "-"
        LOG_FILE_ENABLED = True
        
    # Application Controls
        
    class Application ( object ):
        
        TRAINING_ENABLED = True
        
    
    # Model parameters        
    
    class Model ( object ):
        
        # Training control parameters.
        
        #TRAINING_DATA_LIMIT = 100 
        TRAINING_DATA_LIMIT = -1
        METRIC              = 'logloss'
        
        # XGBClassifier
        
        LEARNING_RATE    = 0.005
        N_ESTIMATORS     = 2000
        MAX_DEPTH        = 3
        MIN_CHILD_WEIGHT = 1
        GAMMA            = 0
        SUBSAMPLE        = 0.25
        COLSAMPLE_BYTREE = 0.25
        OBJECTIVE        = 'binary:logistic'
        SCALE_POS_WEIGHT = 1
        SEED             = 1
        
        # Model parameters.
        
        # learning_rate estimator_count max_depth elapsed_time numerai_logloss
        # 0.5           5               2         00:01:38.696 0.69126
        # 0.1           49              2         00:05:19.353 0.69133
        # 0.05          118             2         00:10:30.526 0.69127
        # 0.01          153             2         00:12:09.226 0.69167
        # 0.005         843             2         01:47:05.741 0.69132
        # 0.001         3830            2         07:23:20.548 0.69126
        # -             -               -         -            -         
        # 0.005         657             4         01:18:54.599 0.69120
        # -             -               -         -            - 
        # 0.005         898             3         01:38:12.810 0.69118
        
        # Reporting settings.
        
        REPORT_FIGURE_FEATURE_RANK_ENABLED = True
        
     
    	
	
	
	
	
	
	