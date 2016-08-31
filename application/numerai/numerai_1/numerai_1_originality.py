# Import libraries.

import pandas as pd
import random

from sklearn.feature_selection import RFE
from sklearn.linear_model      import LogisticRegression
from sklearn.linear_model      import RandomizedLogisticRegression
from sklearn.metrics           import log_loss
from sklearn.cross_validation  import KFold

# Constants

C_NUMERAI = "[NUMERAI]: "

# Start program

print ( "|" )

# Initialize file names.

train_file  = '../data/numerai_training_data.csv'
test_file   = '../data/numerai_tournament_data.csv'
output_file = '../data/predictions.csv'


print ( C_NUMERAI + "Loading data..." )

# Load training data.

train = pd.read_csv ( train_file )
test  = pd.read_csv ( test_file )


# Prepare data for training.

y_train = train.target.values
x_train = train.drop ( 'target', axis = 1 )
x_test  = test.drop  ( 't_id',   axis = 1 )

# TRain model.

print ( C_NUMERAI + "Training model..." )

for count in range (0,2):
    
    print ( C_NUMERAI + "Training model (" + str ( count ) + ") ", end="" )     
    
    model = LogisticRegression (random_state = 23)
    rfe   = RFE ( model, 7 )
    rfe   = rfe.fit ( x_train, y_train )
    
    y_true = y_train
    y_pred = rfe.predict_proba ( x_train )

    training_score    = rfe.score ( x_train, y_train ) * 100
    traninig_log_loss = log_loss ( y_true, y_pred )
    print ( "- Training Score: " + "{0:.2f}%".format ( training_score ), end=""  )
    print ( ", Logloss: " + "{0:.5f}".format ( traninig_log_loss ) )
    

# Execute model.

print ( C_NUMERAI + "Predicting results..." )

p = rfe.predict_proba ( x_test )


# Save results.

print ( C_NUMERAI + "Saving results..." )

test [ 'probability' ] = p [ :, 1 ]

test.to_csv ( output_file, columns = ( 't_id', 'probability' ), index = None )

# Analysis report

print ( C_NUMERAI + "Reporting..." )
print ( C_NUMERAI + "RFE Ranking: " + str ( rfe.ranking_ ) )
