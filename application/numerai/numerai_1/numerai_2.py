# Import libraries.

import pandas as pd

from sklearn                   import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.decomposition     import PCA

# constants

C_NUMERAI = "[NUMERAI]: "

# Start program

print ( "\n" + C_NUMERAI + "Numerai (version 1.0)\n" )

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


# Train model.

print ( C_NUMERAI + "Training model..." )

pca = PCA ( n_components = 2 )
pca.fit ( x_train )
xt_train = pca.transform ( x_train )
xt_test  = pca.transform ( x_test )

model = LogisticRegression ()
model.fit ( xt_train, y_train )

# Execute model.

print ( C_NUMERAI + "Predicting results..." )

p = model.predict_proba ( xt_test )

# Save results.

print ( C_NUMERAI + "Saving results..." )

test [ 'probability' ] = p [ :, 1 ]

test.to_csv ( output_file, columns = ( 't_id', 'probability' ), index = None )

# Analysis report

print ( C_NUMERAI + "Reporting..." )

# summarize the selection of the attributes

print ( C_NUMERAI + "Model: \n" + str ( model ) + "\n" )
