#/////////////////////////////////////////////////////////////////////////////
# Constants
#/////////////////////////////////////////////////////////////////////////////
    
class Constant ( object ):

    # System constants.

    class System ( object ):
        SUPPRESS_TENSOR_FLOW_WARNINGS = False

    # Text formating constans.

    class Text ( object):
        
        SYSTEM       = '[SYSTEM] '
        APPLICATION  = '[APPLICATION] '
        MODEL        = '[MODEL] '
        REPORTING    = '[REPORTING] '
        INDENT       = 4
    
    # Sound constants.
    
    class Sound ( object ):
        
        LOG_PERIOD    = 70
        LOG_FREQUENCY = 100        
        
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
        
        class File ( object ):    
            PATH        = '../data/'
            TRAINING    = 'numerai_training_data.csv'
            LIVE        = 'numerai_tournament_data.csv'
            PREDICTION  = 'predictions.csv'
        
        # CSV file constants.        
        
        class CSV ( object ):
            FEATURE     = 'feature'
            TARGET      = 'target'
            ID          = 'id'
            PROBABILITY = 'probability'

    # Application Controls
        
    class Application ( object ):
        
        pass
    
    # Model parameters        
    
    class Model ( object ):
        
        pass