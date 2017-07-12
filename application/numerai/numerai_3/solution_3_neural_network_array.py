# Import platform libraries.

import math
import random

import numpy   as np
import pandas  as pd

from sklearn.decomposition   import PCA
from sklearn.neural_network  import MLPClassifier
from sklearn                 import metrics

# Import application libraries.

from numerai_constants       import Constant
from utility                 import console_log

#------------------------------------------------------------------------------
# Class: Application
#------------------------------------------------------------------------------

class Application:
    
    #--------------------------------------------------------------------------
    # Function: Constructor
    #--------------------------------------------------------------------------

    def __init__ ( self ):
        
        # File data.        
        
        self.file_path             = Constant.Numerai.File.PATH
        self.file_name_training    = self.file_path + Constant.Numerai.File.TRAINING
        self.file_name_live        = self.file_path + Constant.Numerai.File.LIVE
        self.file_name_predictions = self.file_path + Constant.Numerai.File.PREDICTION
    
    #--------------------------------------------------------------------------
    # Function: run
    #--------------------------------------------------------------------------    
    
    def run ( self ):
        
        # Application start.
        
        console_log ( Constant.Text.APPLICATION + 'Application.Start.', indent = 0, frequency = Constant.Sound.LOG_FREQUENCY )
        
        # Train model.
        
        x_train, y_train                    = self.load_training_data ( self.file_name_training )
        transformation, x_train_transformed = self.select_features    ( x_train )
        model                               = self.train              ( x_train_transformed, y_train )        
        
        # Apply model
        
        id_live, x_live = self.load_live_data ( self.file_name_live )
        y_live    = self.compute_model  ( model, transformation, x_live )
        self.save_live_data ( self.file_name_predictions, id_live, y_live )
                
        # Application end.
        
        console_log ( Constant.Text.APPLICATION + 'Application.Stop.', indent = 0, frequency = Constant.Sound.LOG_FREQUENCY )
        
        self.report_model ( model, transformation, x_train_transformed, y_train )

    #--------------------------------------------------------------------------
    # Function: train
    #--------------------------------------------------------------------------
        
    def select_features ( self, x ):
    
        console_log ( Constant.Text.MODEL + 'Selecting features.', indent = Constant.Text.INDENT, frequency = Constant.Sound.LOG_FREQUENCY )
        
        # Configure aglorythm.        
        
        feature_count  = len ( x.columns )
        transformation = PCA ( n_components = feature_count )
        
        # Fit model.        
        
        transformation.fit ( x )
        
        # Apply model.
        
        x_transformed = transformation.transform ( x )
        
        # return to caller.
        
        return transformation, x_transformed
        
    #--------------------------------------------------------------------------
    # Function: train
    #--------------------------------------------------------------------------  
        
    def train ( self, x, y ):
        
        console_log ( Constant.Text.MODEL + 'Training model.', indent = Constant.Text.INDENT, frequency = Constant.Sound.LOG_FREQUENCY )
        
        # Compute random state.
    
        random_state_max = 2**32 - 1
        random_state     = math.floor ( random_state_max * random.random() )
        
        # Configure model parameters.
    
        model = MLPClassifier (
            hidden_layer_sizes = ( 7, 1 ),
            activation         = 'tanh',
            solver             = 'adam',
            learning_rate      = 'adaptive',
            momentum           = 0.9,
            alpha              = 0.00001,
            random_state       = random_state
        )
        
        # Train the model.
                    
        model.fit ( x, y )
        
        # Return to caller
        
        return model

    #--------------------------------------------------------------------------
    # Function: compute_model
    #--------------------------------------------------------------------------  
        
    def compute_model ( self, model, transformation, x ):
        
        console_log ( Constant.Text.MODEL + 'Computing model.', indent = Constant.Text.INDENT, frequency = Constant.Sound.LOG_FREQUENCY )
        
        # Compute model.
    
        x_transformed = transformation.transform ( x )
        y             = model.predict_proba      ( x_transformed )
        
        # Return to caller.
        
        return y        
                
    #--------------------------------------------------------------------------
    # Function: save_live_data
    #--------------------------------------------------------------------------  
       
    def save_live_data ( self, file_name, i, y ):
                
        # Load file.
        
        console_log ( Constant.Text.MODEL + 'Saving live predictions, "' + file_name + '".', indent = Constant.Text.INDENT, frequency = Constant.Sound.LOG_FREQUENCY )    
        
        # Format prediction results as Pandas dataframe.    
        
        results                = y [ :, 1 ]
        results_dataframe      = pd.DataFrame ( data = { Constant.Numerai.CSV.PROBABILITY : results } )
        y_prediction_dataframe = pd.DataFrame ( i ).join ( results_dataframe )
        
        # Save data.
        
        y_prediction_dataframe.to_csv ( file_name, index = False )
        
        
    #--------------------------------------------------------------------------
    # Function: load_training_data
    #--------------------------------------------------------------------------  
        
    def load_training_data ( self, file_name ):
                
        # Load data file.
        
        console_log ( Constant.Text.MODEL + 'Loading training data, "' + file_name + '".', indent = Constant.Text.INDENT, frequency = Constant.Sound.LOG_FREQUENCY )
        
        training_data = pd.read_csv ( file_name, header = 0 )
        
        # Format the loaded CSV data into numpy arrays.
        
        features = [ f for f in list ( training_data ) if Constant.Numerai.CSV.FEATURE in f ]
        
        x = training_data [ features ]
        y = training_data [ Constant.Numerai.CSV.TARGET ]
        
        # return to caller        
        
        return x, y


    #--------------------------------------------------------------------------
    # Function: load_validation_data
    #--------------------------------------------------------------------------  
    
    def load_validation_data ( self, file_name ):
        
        console_log ( Constant.Text.MODEL + 'Loading validation data, "' + file_name + '".', indent = Constant.Text.INDENT, frequency = Constant.Sound.LOG_FREQUENCY )

    #--------------------------------------------------------------------------
    # Function: load_test_data
    #--------------------------------------------------------------------------  
    
    def load_test_data ( self, file_name ):
        
        console_log ( Constant.Text.MODEL + 'Loading test data, "' + file_name + '".', indent = Constant.Text.INDENT, frequency = Constant.Sound.LOG_FREQUENCY )
 
    #--------------------------------------------------------------------------
    # Function: load_live_data
    #--------------------------------------------------------------------------  
       
    def load_live_data ( self, file_name ):
                
        # Load file.
        
        console_log ( Constant.Text.MODEL + 'Loading live data, "' + file_name + '".', indent = Constant.Text.INDENT, frequency = Constant.Sound.LOG_FREQUENCY )
        
        live_data = pd.read_csv ( file_name, header = 0 )
    
        # Format the loaded CSV data into numpy arrays.
            
        features = [ f for f in list ( live_data ) if Constant.Numerai.CSV.FEATURE in f ]
            
        i = live_data [ Constant.Numerai.CSV.ID ]       # id vector......
        x = live_data [ features ]                      # Feature tensor.
        
        # Return o caller.
        
        return i, x
    #-----------------------------------------------------------------------------
    # FUNCTION: Report results.
    #-----------------------------------------------------------------------------
    
    def report_model ( self, model, transformation, x, y ):
    
        print ( '' )
        print ( 'Learning Algorythm: ' + model.__class__.__name__ )
        self.print_model_parameters ( model )
        
        print ( '' )
        print ( 'Dimentionality Reduction Algorythm: ' + transformation.__class__.__name__ )
        self.print_model_parameters ( transformation )
        
        print ( '' )
        print ( 'Model Test Performance:' )
        self.report_model_results ( model, x, y )
        
    #-----------------------------------------------------------------------------
    # Display model test performance.
    #-----------------------------------------------------------------------------
    
    def report_model_results ( self, model, x, y ):
        
        CONSOLE_ALIGN_KEY = '{0:.<24}'
        
        # Test model.
        
        y_predictions              = model.predict       ( x )
        y_prediction_probabilities = model.predict_proba ( x ) [ :, 1 ]
        
        # Display test results.
            
        accuracy = metrics.accuracy_score ( y.values, y_predictions              ) * 100
        auc      = metrics.roc_auc_score  ( y,        y_prediction_probabilities )
        logloss  = metrics.log_loss       ( y,        y_prediction_probabilities )
        
        print ( CONSOLE_ALIGN_KEY.format ( '- Accuracy' ) + ' = ' + '{:.3f}'.format ( accuracy ) )
        print ( CONSOLE_ALIGN_KEY.format ( '- AUC' )      + ' = ' + '{:.6f}'.format ( auc ) )
        print ( CONSOLE_ALIGN_KEY.format ( '- logloss' )  + ' = ' + '{:.6f}'.format ( logloss ) )
    
    #-----------------------------------------------------------------------------
    # FUNCTION: Print model parameters.
    #-----------------------------------------------------------------------------
    
    def print_model_parameters ( self, model ):
    
        indent = '- '
        
        for key in model.get_params():  
                     
            parameter_str =  indent
            parameter_str += '{0:.<24}'.format ( key )
            parameter_str += ' = '
            parameter_str += str ( model.get_params() [ key ] )
                    
            print ( parameter_str )
