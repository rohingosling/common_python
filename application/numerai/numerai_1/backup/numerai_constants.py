#/////////////////////////////////////////////////////////////////////////////
# Library imports.
#/////////////////////////////////////////////////////////////////////////////

import os

#/////////////////////////////////////////////////////////////////////////////
# Constants
#/////////////////////////////////////////////////////////////////////////////
    
class Constant ( object ):

    # Console output.
    
    INDENT = "  "
    
    # CSV file column names.
    
    CSV_TRAINING_TARGET         = "target"
    CSV_APPLICATION_ID          = "t_id"
    CSV_APPLICATION_PROBABILITY = "probability"
    
    # Data file path and file names.
    
    DATA_FILE_PATH        = "../data/"
    DATA_FILE_TRAINING    = DATA_FILE_PATH + "numerai_training_data.csv"
    DATA_FILE_APPLICATION = DATA_FILE_PATH + "numerai_tournament_data.csv"
    DATA_FILE_PREDICTION  = DATA_FILE_PATH + "predictions.csv"
    
    # Program file name.
    
    PROGRAM_FILE_NAME = os.path.basename ( __file__ )  
    
    # Log file path and file name.
    
    LOG_FILE_PATH    = ""
    LOG_FILE_NAME    = PROGRAM_FILE_NAME [ :-2 ] + "log"
    LOG_FILE_ENABLED = True
    
    # Training settings.
    
    TRAINING_MODEL_COUNT = 4
    
    # Algorithms: GradientBoostingRegressor
    
    GBR_MAX_FEATURES      = 9
    GBR_MAX_DEPTH         = 128
    GBR_N_ESTIMATORS      = 6
    GBR_LEARNING_RATE     = 0.001
    GBR_WARM_START        = False
    GBR_SUBSAMPLE         = 1.0
    GBR_MIN_SAMPLES_SPLIT = 2
    GBR_VERBOSE           = 0
    
    # Reporting settings.
    
    REPORT_MODEL_PARAMETERS_ENABLED    = True
    REPORT_FIGURE_FEATURE_RANK_ENABLED = True
