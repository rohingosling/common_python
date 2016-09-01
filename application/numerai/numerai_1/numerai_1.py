#-----------------------------------------------------------------------------
# Import libraries.
#-----------------------------------------------------------------------------

import pandas as pd
import os

from sklearn.linear_model import LogisticRegression


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


#-----------------------------------------------------------------------------
# Initialize Program
#-----------------------------------------------------------------------------

new_line ()
log ( "PROGRAM: " + os.path.basename(__file__) )
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

model = LogisticRegression ()
model.fit ( x_train, y_train )


#-----------------------------------------------------------------------------
# Apply model.
#-----------------------------------------------------------------------------

log ( "Predicting results." )

y_application = model.predict_proba ( x_application )


#-----------------------------------------------------------------------------
# Save results.
#-----------------------------------------------------------------------------

log ( "Saving results." )

data_application [ 'probability' ] = y_application [ :, 1 ]

data_application.to_csv (
    file_prediction, 
    columns = ( 't_id', 'probability' ), 
    index   = None
)


#-----------------------------------------------------------------------------
# Analysis and Reporting.
#-----------------------------------------------------------------------------

log ( "Reporting." )
log ( "MODEL:\n\n" + str ( model ) )



