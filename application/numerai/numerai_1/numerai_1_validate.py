# !/usr/bin/env python

"Load data, create the validation split, optionally scale data, train a linear model, evaluate"
"Code updated for march 2016 data"

import pandas as pd
import random

from sklearn.cross_validation import train_test_split
from sklearn.linear_model     import LogisticRegression as LR
from sklearn.metrics          import roc_auc_score      as AUC, log_loss
from sklearn                  import metrics
from sklearn.ensemble         import ExtraTreesClassifier

#

def train_and_evaluate( y_train, x_train, y_val, x_val ):

    model = ExtraTreesClassifier()
    model.fit ( x_train, y_train )

    p = model.predict_proba ( x_val )
   
    pt = model.predict_proba ( x_test )
    
	
    auc = AUC      ( y_val, p[:,1] )
    ll  = log_loss ( y_val, p[:,1] )
    
    return ( auc, ll, pt, model )
	
# Execute training sequence.

test_file   = '../data/numerai_tournament_data.csv'
output_file = '../data/predictions.csv'
input_file  = '../data/numerai_training_data.csv'
d           = pd.read_csv ( input_file )
test        = pd.read_csv ( test_file )

train, val = train_test_split( d, test_size = 3000 )

y_train = train.target.values
y_val   = val.target.values

x_train = train.drop ( 'target', axis = 1 )
x_val   = val.drop   ( 'target', axis = 1 )
x_test  = test.drop  ( 't_id',   axis = 1 )   

# train, predict, evaluate

print ("Initiatetraining sequence..\n" )

random.seed()

ll = 1.0

count = 0
while ( ( ll >= 0.69 ) and (count < 3) ):
    
    auc, ll, pt, model = train_and_evaluate ( y_train, x_train, y_val, x_val )

    print ( str(count) + ": " + "AUC: {:.2%}, log loss: {:.5} \n".format( auc, ll ) )
    
    count += 1

test [ 'probability' ] = pt [ :, 1 ]
test.to_csv ( output_file, columns = ( 't_id', 'probability' ), index = None )
print ( "Itteration Count = " + str(count))
print ( model.feature_importances_ )