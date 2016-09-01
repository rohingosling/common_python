# Import libraries.

import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.linear_model      import LogisticRegression

# Start program

print ( "\nNumerai (version 1.0)\n" )

# Initialize file names.

train_file  = '../data/numerai_training_data.csv'
test_file   = '../data/numerai_tournament_data.csv'
output_file = '../data/predictions.csv'


print ( "Loading data..." )

# Load training data.

train = pd.read_csv ( train_file )
test  = pd.read_csv ( test_file )


# Prepare data for training.

y_train = train.target.values
x_train = train.drop ( 'target', axis = 1 )
x_test  = test.drop  ( 't_id',   axis = 1 )


print ( "Training model..." )

# create the RFE model and select 3 attributes

model = LogisticRegression()

# create the RFE model and select 3 attributes

rfe = RFE ( model, 7 )
rfe = rfe.fit ( x_train, y_train )

# Execute model.

print ( "Predicting results..." )

p = rfe.predict_proba ( x_test )


# Save results.

print ( "Saving results..." )

test [ 'probability' ] = p [ :, 1 ]

test.to_csv ( output_file, columns = ( 't_id', 'probability' ), index = None )

# Analysis report

print ( "Reporting...\n" )

# summarize the selection of the attributes

print ( "RFE Support: " + str ( rfe.support_ ) )
print ( "RFE Ranking: " + str ( rfe.ranking_ ) )