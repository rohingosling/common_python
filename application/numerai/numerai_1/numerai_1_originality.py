# Import libraries.

import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.linear_model      import LogisticRegression

# Constants

C_NUMERAI = "[NUMERAI]: "

# Start program

print ("" )
print ( "| Numerai Model" )
print ( "| Cersion 1.0" )
print ( "| 2016-08-31" )
print ( "| Rohin Gosling" )
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


print ( C_NUMERAI + "Training model..." )

# create the RFE model and select 3 attributes

model = LogisticRegression ( verbose = 2 )

# create the RFE model and select 3 attributes

rfe = RFE ( model, 3 )
rfe = rfe.fit ( x_train, y_train )

# Execute model.

print ( "" )
print ( C_NUMERAI + "Predicting results..." )

p = rfe.predict_proba ( x_test )


# Save results.

print ( C_NUMERAI + "Saving results..." )

test [ 'probability' ] = p [ :, 1 ]

test.to_csv ( output_file, columns = ( 't_id', 'probability' ), index = None )

# Analysis report

print ( C_NUMERAI + "Reporting...\n" )

# summarize the selection of the attributes

print ( "RFE Ranking: " + str ( rfe.ranking_ ) )