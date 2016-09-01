#-----------------------------------------------------------------------------
# Import libraries.
#-----------------------------------------------------------------------------

import pandas as pd
import os

from sklearn.metrics      import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge


#-----------------------------------------------------------------------------
# Constants
#-----------------------------------------------------------------------------

C_NUMERAI = "[NUMERAI]: "


#-----------------------------------------------------------------------------
# Functions.
#-----------------------------------------------------------------------------

# Console logging functions.

def log ( message ):
    print ( C_NUMERAI + message )
    
def new_line():
    print ( "" )
    
# Function: Training functions.
    
def train_model ( alpha, x_train, y_train ):
    
    model = Ridge (
                alpha     = alpha,
                normalize = True
            )

    model.fit ( x_train, y_train )
    
    return model
    
# Function: Calculate log loss.

def compute_log_loss ( x_train, y_train ):
    
    y_true = y_train
    y_pred = model.predict ( x_train )
    
    return log_loss ( y_true, y_pred )


#-----------------------------------------------------------------------------
# Initialize Program
#-----------------------------------------------------------------------------

new_line ()
log ( "PROGRAM: " + os.path.basename ( __file__ ) )
log ( "Initializing." )

# Initialize file names.

file_path        = "../data/"
file_training    = file_path + "numerai_training_data.csv"
file_application = file_path + "numerai_tournament_data.csv"
file_prediction  = file_path + "predictions.csv"


#-----------------------------------------------------------------------------
# Load training data.
#-----------------------------------------------------------------------------

log ( "Loading data." )

data_training    = pd.read_csv ( file_training )
data_application = pd.read_csv ( file_application )

# Prepare data for training.

y_train       = data_training.target.values
x_train       = data_training.drop    ( 'target', axis = 1 )
x_application = data_application.drop ( 't_id',   axis = 1 )


#-----------------------------------------------------------------------------
# Train model.
#-----------------------------------------------------------------------------

log ( "Training model." )

# Train model.

training_cycle_count = 1
alpha_step           = 0.0
alpha                = 0.01

for training_cycle in range ( 0, training_cycle_count ):

    # Train model.

    model = train_model ( alpha, x_train, y_train )
    
    # Test model.    
    
    log_loss_traning = compute_log_loss ( x_train, y_train )  
    
    # Report results.    
    
    log ( "Training Cycle " + str ( training_cycle ) + ": " + "alpha = " + str ( round ( alpha, 3 ) ) + ", log_loss = " + "{0:.5f}".format ( log_loss_traning ) )
    
    # Update variables.
    
    alpha += alpha_step


#-----------------------------------------------------------------------------
# Apply model.
#-----------------------------------------------------------------------------

log ( "Predicting results." )

y_application = model.predict ( x_application )


#-----------------------------------------------------------------------------
# Save results.
#-----------------------------------------------------------------------------

log ( "Saving results." )

data_application [ 'probability' ] = y_application #y_application [ :, 0 ]

data_application.to_csv (
    file_prediction, 
    columns = ( 't_id', 'probability' ), 
    index   = None
)


#-----------------------------------------------------------------------------
# Analysis and Reporting.
#-----------------------------------------------------------------------------

log ( "Reporting." )
#log ( "log_loss = " + "{0:.5f}".format ( log_loss_traning ) )
log ( "MODEL:\n\n" + str ( model ) )


