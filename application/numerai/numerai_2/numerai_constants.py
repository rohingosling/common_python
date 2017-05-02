#/////////////////////////////////////////////////////////////////////////////
# Constants
#/////////////////////////////////////////////////////////////////////////////
    
class Constant ( object ):

    # Text formating constans.

    class Text ( object):
        
        INDENT = "|   "
        
    class Sound ( object ):
        
        EVENT_PERIOD    = 70
        EVENT_FREQUENCY = 13000
        
        START_PERIOD    = 70
        START_FREQUENCY = 10000
        
        STOP_PERIOD    = 70
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

    # Application Controls
        
    class Application ( object ):
        
        # Parameter optimization.        
        
        PARAMETER_OPTIMIZATION_ENABELED    = False
        
        OPTIMIZE_ESTIMATOR_COUNT           = True
        OPTIMIZE_TREE_PARAMETERS           = True
        OPTIMIZE_REGULARIZATION_PARAMETERS = True
        
        # Training
		
        TRAINING_ENABLED  = True        
        FIT_MODEL_ENBALED = True
        
        # Test
        
        TEST_MODEL = False
        
        # Reporting
        
        TRAINING_REPORT_ENABLED   = False
        PLOT_FEATURE_RANK_ENABLED = False
		
        # Application
		
        MODEL_APPLICATION_ENABLED = True
        
    
    # Model parameters        
    
    class Model ( object ):
        
        # Training control parameters.
        
        TRAINING_DATA_LIMIT = 1000 
        #TRAINING_DATA_LIMIT = -1
        
        # Cross validation.
                
        CROSS_VALIDATION_FOLD_COUNT = 5
        EARLY_STOPPING_COUNT        = 50
        METRIC                      = 'logloss'
        
        # XGBClassifier
        
        LEARNING_RATE    = 0.07
        N_ESTIMATORS     = 207
        MAX_DEPTH        = 1
        MIN_CHILD_WEIGHT = 1
        GAMMA            = 0.0
        SUBSAMPLE        = 0.28
        COLSAMPLE_BYTREE = 0.05
        REG_ALPHA        = 6.0
        REG_LAMBDA       = 0.6
        OBJECTIVE        = 'binary:logistic'
        SCALE_POS_WEIGHT = 1
        SEED             = 1
        
        # Grid Search
        
        class GridSearch ( object ):
            
            class Tree ( object ):
                
                SCORING = 'log_loss'
                CV      = 3           # 2 <= f <= 5
                VERBOSE = 2           # 0 <= v <= 10
        
        
        
        
     
    	
	
	
	
	
	
	