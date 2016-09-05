#-----------------------------------------------------------------------------
# Import libraries.
#-----------------------------------------------------------------------------
    
import random
import os

import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import log_loss
from sklearn.ensemble     import GradientBoostingRegressor


#-----------------------------------------------------------------------------
# Constants
#-----------------------------------------------------------------------------

# Strings.

C_NUMERAI = "[NUMERAI]: "

# Training settings.

C_TRAINING_MODEL_COUNT = 4

# Algorythms: GradientBoostingRegressor

C_GBR_LEARNING_RATE = 0.01      # Best: 0.01
C_GBR_MAX_FEATURES  = 21        # Best: 14
C_GBR_N_ESTIMATORS  = 256       # Best: 256
C_GBR_MAX_DEPTH     = 5         # Best: 5
C_GBR_WARM_START    = False     # Best: False


#-----------------------------------------------------------------------------
# Functions.
#-----------------------------------------------------------------------------

# Console logging functions.

def log ( message, newline = True ):
    if newline:
        print ( C_NUMERAI + message )
    else:
        print ( C_NUMERAI + message, end="" )
    
def new_line():
    print ( "" )

# Train model.

def train_model ( x_train, y_train ):
    
    model = GradientBoostingRegressor (
                #loss          = "quantile",
                #loss          = "huber",
                #alpha         = 0.01,
                learning_rate = C_GBR_LEARNING_RATE,
                max_features  = C_GBR_MAX_FEATURES,
                n_estimators  = C_GBR_N_ESTIMATORS,
                max_depth     = C_GBR_MAX_DEPTH,
                random_state  = random.randint ( 0, 1000 ),
                warm_start    = C_GBR_WARM_START
            )
            
    model.fit ( x_train, y_train )

    return model

# Train best model out of N.

def train_best_model ( x_train, y_train, count ):
    
    training_cycle_count   = count
    log_loss_training_best = 1.0
    
    for training_cycle_index in range ( 0, training_cycle_count ):
        
        log ( "- Model " + str ( training_cycle_index+1 ) + "/" + str ( training_cycle_count ) + ": ", newline = False )    
        
        # Train model.
        
        model = train_model ( x_train, y_train )
        
        # Test model and calculate log loss.
        
        log_loss_traning = compute_log_loss ( model, x_train, y_train )
        
        # Update best model.
        
        if log_loss_traning < log_loss_training_best:
            log_loss_training_best = log_loss_traning
            model_best             = model
        
        # Report results.
    
        print ( "log_loss = " + "{0:.5f}".format ( log_loss_traning ) )
        
    # Update best model.
    
    model = model_best
    
    return model

# Compute log loss.

def compute_log_loss ( model, x_train, y_train ):
    
    y_true = y_train
    y_pred = model.predict ( x_train )

    return log_loss ( y_true, y_pred )

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

log ( "Loading data." + " \"" + file_training + "\"" )

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

# Train models.

model = train_best_model ( x_train, y_train, C_TRAINING_MODEL_COUNT )

#-----------------------------------------------------------------------------
# Apply model.
#-----------------------------------------------------------------------------

log ( "Predicting results." )

y_application = model.predict ( x_application )


#-----------------------------------------------------------------------------
# Save results.
#-----------------------------------------------------------------------------

log ( "Saving results." + " \"" + file_prediction + "\""  )

data_application [ 'probability' ] = y_application # y_application [ :, 0 ]

data_application.to_csv (
    file_prediction, 
    columns = ( 't_id', 'probability' ), 
    index   = None
)


#-----------------------------------------------------------------------------
# Analysis and Reporting.
#-----------------------------------------------------------------------------

log ( "Reporting." )

# Collect data to report on.

best_log_loss = compute_log_loss ( model, x_train, y_train )
feature_rank  = model.feature_importances_  

# Print reporting and analysis data.

log ( "- Best log loss = " + "{0:.5f}".format ( best_log_loss ) )
log ( "- MODEL:\n\n" + str ( model ) )


#-----------------------------------------------------------------------------
# Plots
#-----------------------------------------------------------------------------

# Collect data to plot.


feature_count = 21
#indices       = range ( 0, feature_count     )
indices       = np.argsort ( model.feature_importances_ )
features      = range ( 1, feature_count + 1 )
bar_width     = 0.8

# Plot the feature importances of the forest

plt.bar (
    np.arange ( feature_count ),
    model.feature_importances_ [ indices ],
    bar_width,
    color = 'grey',
    align = 'center'
)

plt.title  ( "Feature Rank" )
plt.ylabel ( "Relative Rank" )
plt.xlabel ( "Feature" )
plt.xticks ( np.arange ( feature_count ) + bar_width/2.0, indices )

plt.show ()




