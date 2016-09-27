#/////////////////////////////////////////////////////////////////////////////
# Library imports.
#/////////////////////////////////////////////////////////////////////////////

import os

#/////////////////////////////////////////////////////////////////////////////
# Constants
#/////////////////////////////////////////////////////////////////////////////
    
class Constant ( object ):

    # Console output.
    
    INDENT = "-   "
    
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
    
    # CSV file column names.
    
    class CSV ( object ):
        
        class Header ( object ):    
            TRAINING_TARGET         = "target"
            APPLICATION_ID          = "t_id"
            APPLICATION_PROBABILITY = "probability"        
    
    # Data file path and file names.
    
    class DataFile ( object ):    
        PATH        = "../data/"
        TRAINING    = "numerai_training_data.csv"
        APPLICATION = "numerai_tournament_data.csv"
        PREDICTION  = "predictions.csv"
    
    # Log file path and file name.
    
    LOG_FILE_PATH    = ''
    #LOG_FILE_NAME    = "numerai.log.csv"
    LOG_FILE_NAME    = "-"
    LOG_FILE_ENABLED = True
    
    # Training settings.
    
    TRAINING_MODEL_COUNT = 4
    
    # Algorithms: GradientBoostingRegressor
    
    GBR_MAX_FEATURES      = 4
    GBR_MAX_DEPTH         = 3
    GBR_N_ESTIMATORS      = 400
    GBR_LEARNING_RATE     = 0.009
    GBR_WARM_START        = False
    GBR_SUBSAMPLE         = 1.0
    GBR_MIN_SAMPLES_SPLIT = 2
    GBR_VERBOSE           = 0
    
    # Reporting settings.
    
    REPORT_MODEL_PARAMETERS_ENABLED    = True
    REPORT_FIGURE_FEATURE_RANK_ENABLED = True
	
	
	
	
	
	
	