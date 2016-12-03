#/////////////////////////////////////////////////////////////////////////////
#
# PROGRAM: Numerai - Solution 3
# VERSION: 3.0
# AUTHOR:  Rohin Gosling
#
#/////////////////////////////////////////////////////////////////////////////

#/////////////////////////////////////////////////////////////////////////////
# Import libraries.
#/////////////////////////////////////////////////////////////////////////////

# Platform imports.

import winsound
import time
import datetime
import sys

import pandas           as pd
import numpy            as np
import xgboost          as xgb
import matplotlib.pylab as plt

from matplotlib.pylab    import rcParams
from xgboost.sklearn     import XGBClassifier
from sklearn             import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

# Application inports

from numerai_constants  import Constant
from data_visualization import plot_feature_rank

# Platform configuration.

rcParams [ 'figure.figsize' ] = 12, 4

#/////////////////////////////////////////////////////////////////////////////
# GLobal Variables
#/////////////////////////////////////////////////////////////////////////////

time_event = time.time()

#/////////////////////////////////////////////////////////////////////////////
# Functions.
#/////////////////////////////////////////////////////////////////////////////

#-----------------------------------------------------------------------------
# FUNCTION: initialize_model
#-----------------------------------------------------------------------------

def initialize_model ():
    
    log ( 'Initialize model: XGBClassifier.', indent = 0 )
    
    # Initialize model
    
    model = XGBClassifier (
    
        learning_rate    = Constant.Model.LEARNING_RATE,
        n_estimators     = Constant.Model.N_ESTIMATORS,
        max_depth        = Constant.Model.MAX_DEPTH,
        min_child_weight = Constant.Model.MIN_CHILD_WEIGHT,
        gamma            = Constant.Model.GAMMA,
        subsample        = Constant.Model.SUBSAMPLE,
        colsample_bytree = Constant.Model.COLSAMPLE_BYTREE,
<<<<<<< HEAD
=======
        reg_alpha        = Constant.Model.REG_ALPHA,
        reg_lambda       = Constant.Model.REG_LAMBDA,
>>>>>>> dev
        objective        = Constant.Model.OBJECTIVE,
        scale_pos_weight = Constant.Model.SCALE_POS_WEIGHT,        
        seed             = Constant.Model.SEED     
    )
<<<<<<< HEAD
    
    # Print model parameters.
    
    print_model_parameters ( model, 1 )
    
    # Return initialized model.
    
=======
    
    # Print model parameters.
    
    print_model_parameters ( model, 1 )
    
    # Return initialized model.
    
>>>>>>> dev
    return model

#-----------------------------------------------------------------------------
# FUNCTION: Train model.
#-----------------------------------------------------------------------------

def train_model ( model ):
    
    # Local variables.
    
    row_count = Constant.Model.TRAINING_DATA_LIMIT
    
    # Begin training sequence.
<<<<<<< HEAD
    
    log ( 'TRAINING SEQUENCE:' )    
    
    if Constant.Application.TRAINING_ENABLED:
    
        # Compile file names.
        
        training_file_name = Constant.Numerai.DataFile.PATH + Constant.Numerai.DataFile.TRAINING    
        
        # Load training data.
        
        x, t = load_training_data ( training_file_name, row_count ) 
        
        # Initialize, optimize and train mode.        
        
        model = optimize_model_parameters ( model, x, t )
        model = fit_model                 ( model, x, t )
    
        # Test Model.
    
=======
    
    log ( 'TRAINING SEQUENCE:' )    
    
    if Constant.Application.TRAINING_ENABLED:
    
        # Compile file names.
        
        training_file_name = Constant.Numerai.DataFile.PATH + Constant.Numerai.DataFile.TRAINING    
        
        # Load training data.
        
        x, t = load_training_data ( training_file_name, row_count ) 
        
        # Initialize, optimize and train mode.        
        
        model = optimize_model_parameters ( model, x, t )
        model = fit_model                 ( model, x, t )
    
        # Test Model.
    
>>>>>>> dev
        accuracy, auc, logloss = test_model ( model, x, t)
        
        # Report training results.
        
        report_model ( model, accuracy, auc, logloss )
        
    else:
        
        log ( 'Traning: DISABLED', indent = 1 )

    # Return trained model
    
    return model

#-----------------------------------------------------------------------------
# FUNCTION: Optimize model parameters.
#-----------------------------------------------------------------------------

def optimize_model_parameters ( model, x, t ):
    
    log ( 'Optimizing Parameters:', indent = 1 )
    
    if Constant.Application.PARAMETER_OPTIMIZATION_ENABELED:
        
        # Compute optimal estimator count.
    
        optimize_estimator_count ( model, x, t )
        
        # Compute optimal tree parameters.

        optimize_tree_parameters ( model, x, t )
        
        # Compute regularization parameters.
        
        optimize_regularization_parameters ( model, x, t )        
        
        # Print model parameters.
        
        print_model_parameters ( model, 2 )
        
    else:
        
        log ( 'Parameter Optimization: DISABLED', indent = 2 )
        
    
    return model

#-----------------------------------------------------------------------------
# FUNCTION: Grid searcvh estimator count.
#-----------------------------------------------------------------------------

def optimize_estimator_count ( model, x, t ):
    
    log ( 'Optimizing estimator count:', indent = 2 )

    if Constant.Application.OPTIMIZE_ESTIMATOR_COUNT:    

        learning_rate_range = [ 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.009, 0.001 ]

        data_columns = [ 'r', 'n', 't', 'logloss' ]        
        result_table = pd.DataFrame ( columns = data_columns )
        
        index = 0
        for r in learning_rate_range:
        
            estimator_count, logloss_test_mean, elapsed_time_formatted = compute_estimator_count ( model, x, t, r )
            
            data  = [ r, int(estimator_count), elapsed_time_formatted, logloss_test_mean ]        
                    
            result_table.loc [ index ] = data
            index += 1
            
        # Select optimal estimator count.
            
        logloss_min             = result_table [ 'logloss' ].argmin()
        row                     = result_table.loc [ logloss_min ]        
        optimal_learning_rate   = row [ 'r' ]
        optimal_estimator_count = int ( row [ 'n' ] )
        
        # Set iptimal learning rate and estimator count
        
        model.set_params ( learning_rate = optimal_learning_rate   )
        model.set_params ( n_estimators  = optimal_estimator_count )
            
        # Print results table.
            
        print_data_frame ( result_table, indent = 3 )
        
        log ( 'Optimal learning rate   = ' + str(optimal_learning_rate),   indent = 3 )
        log ( 'Optimal estimator count = ' + str(optimal_estimator_count), indent = 3 )
        
    else:
        
        log ( 'Estimator Count Optimization: DISABLED', indent = 4 )
        
    return model

#-----------------------------------------------------------------------------
# FUNCTION: Grid search rtree parameters.
#-----------------------------------------------------------------------------

def optimize_tree_parameters ( model, x, t ):
    
    # Local constants.
    
    MAX_DEPTH_AND_MIN_CHILD_WEIGHT_ENABLED   = True
    GAMMA_ENABLED                            = True
    SUB_SAMPLE_AND_COL_SAMPLE_BYTREE_ENABLED = True
    
    # Execute optimization sequence.

    log ( 'Optimizing tree parameters:', indent = 2 )

    if Constant.Application.OPTIMIZE_TREE_PARAMETERS:
        
        if MAX_DEPTH_AND_MIN_CHILD_WEIGHT_ENABLED:
            model = optimize_max_depth_and_min_child_weight ( model, x, t )
            
        if GAMMA_ENABLED:
            model = optimize_gamma                          ( model, x, t )
            
        if SUB_SAMPLE_AND_COL_SAMPLE_BYTREE_ENABLED:
            model = optimize_subsample_and_colsample_bytree ( model, x, t )
                
    else:
        log ( 'Tree Parameter Optimization: DISABLED', indent = 4 )

    return model


#-----------------------------------------------------------------------------
# FUNCTION: Grid search regularization parameters.
#-----------------------------------------------------------------------------

def optimize_regularization_parameters ( model, x, t ):

    # Local variables.
    
    indent = 3
    
    log ( 'Optimizing max depth and min child weight:', indent = indent )
<<<<<<< HEAD
            
    # Configure grid search.
    
    parameter_search_1 = {
        'reg_alpha'  : [ x for x in range ( 3, 7+1, 1 ) ],
        'reg_lambda' : [ (0.5-0.1)+(x*0.1) for x in range ( 1, 7+1, 1 ) ]
=======
       
    # Initialize grid search parameters.

    reg_alpha_locus              =  5.0
    reg_alpha_min_index          = -3
    reg_alpha_max_index          =  3    
    reg_alpha_index_stride_scale =  0.5
    reg_alpha_index_range        = range ( reg_alpha_min_index, reg_alpha_max_index + 1, 1 )
    reg_alpha_search_domain      = [ reg_alpha_locus + ( x * reg_alpha_index_stride_scale ) for x in reg_alpha_index_range ]
    
    reg_lambda_locus              =  1.0
    reg_lambda_min_index          = -5
    reg_lambda_max_index          =  1    
    reg_lambda_index_stride_scale =  0.1
    reg_lambda_index_range        = range ( reg_lambda_min_index, reg_lambda_max_index + 1, 1 )
    reg_lambda_search_domain      = [ reg_lambda_locus + ( x * reg_lambda_index_stride_scale ) for x in reg_lambda_index_range ]
       
    # Configure grid search.
    
    parameter_search_1 = {
        'reg_alpha'  : reg_alpha_search_domain,
        'reg_lambda' : reg_lambda_search_domain
>>>>>>> dev
    }
     
    # Perform grid search.
    
    grid_search_1 = GridSearchCV (
    
        estimator = XGBClassifier (
            learning_rate    = Constant.Model.LEARNING_RATE,
            n_estimators     = Constant.Model.N_ESTIMATORS,
            max_depth        = Constant.Model.MAX_DEPTH,
            min_child_weight = Constant.Model.MIN_CHILD_WEIGHT,
            gamma            = Constant.Model.GAMMA,
            subsample        = Constant.Model.SUBSAMPLE,
            colsample_bytree = Constant.Model.COLSAMPLE_BYTREE,
            objective        = Constant.Model.OBJECTIVE,
            scale_pos_weight = Constant.Model.SCALE_POS_WEIGHT,        
            seed             = Constant.Model.SEED     
        ),
        
        param_grid = parameter_search_1,
        scoring    = Constant.Model.GridSearch.Tree.SCORING,
        cv         = Constant.Model.GridSearch.Tree.CV,
        verbose    = Constant.Model.GridSearch.Tree.VERBOSE,
        n_jobs     = 1,
        iid        = False
    )
    
    grid_search_1.fit ( x, t )
    
    # Report results.
    
    for e in grid_search_1.grid_scores_:
        log ( str (e) , indent = indent+1 )
    
    reg_alpha  = grid_search_1.best_params_ [ 'reg_alpha'  ]
    reg_lambda = grid_search_1.best_params_ [ 'reg_lambda' ]
<<<<<<< HEAD
    
    log ( 'Optimal regularization alpha  = ' + str ( reg_alpha                  ), indent = indent+1 )
    log ( 'Optimal regularization lambda = ' + str ( reg_lambda                 ), indent = indent+1 )
    log ( 'Best Score                    = ' + str ( grid_search_1.best_score_  ), indent = indent+1 )
        
    # update model using optimized parameters.
    
    model.set_params ( reg_alpha  = reg_alpha )
    model.set_params ( reg_lambda = reg_lambda )
    
    # Return updated model.
    
    return model

=======
    
    log ( 'Optimal regularization alpha  = ' + str ( reg_alpha                  ), indent = indent+1 )
    log ( 'Optimal regularization lambda = ' + str ( reg_lambda                 ), indent = indent+1 )
    log ( 'Best Score                    = ' + str ( grid_search_1.best_score_  ), indent = indent+1 )
        
    # update model using optimized parameters.
    
    model.set_params ( reg_alpha  = reg_alpha )
    model.set_params ( reg_lambda = reg_lambda )
    
    # Return updated model.
    
    return model
>>>>>>> dev

#-----------------------------------------------------------------------------
# FUNCTION: Optimize min_child_weight and max_depth
#-----------------------------------------------------------------------------

<<<<<<< HEAD
=======
#-----------------------------------------------------------------------------
# FUNCTION: Optimize min_child_weight and max_depth
#-----------------------------------------------------------------------------

>>>>>>> dev
def optimize_max_depth_and_min_child_weight ( model, x, t ):
    
    # Local variables.
    
    indent = 3
    
    log ( 'Optimizing max depth and min child weight:', indent = indent )
    
    # Initialize ssearch parameters.

    grid_resolution = 3        
        
    max_depth_min    = 1
    max_depth_max    = 9  
    max_depth_stride = grid_resolution
    
    min_child_weight_min    = 1
    min_child_weight_max    = 9    
    min_child_weight_stride = grid_resolution
    
    # Configure grid search.
    
    parameter_search_1 = {
        'max_depth'        : list ( range ( max_depth_min,        max_depth_max,        max_depth_stride        ) ),
        'min_child_weight' : list ( range ( min_child_weight_min, min_child_weight_max, min_child_weight_stride ) )
    }
     
    # Perform grid search.
    
    grid_search_1 = GridSearchCV (
    
        estimator = XGBClassifier (
            learning_rate    = Constant.Model.LEARNING_RATE,
            n_estimators     = Constant.Model.N_ESTIMATORS,
            max_depth        = Constant.Model.MAX_DEPTH,
            min_child_weight = Constant.Model.MIN_CHILD_WEIGHT,
            gamma            = Constant.Model.GAMMA,
            subsample        = Constant.Model.SUBSAMPLE,
            colsample_bytree = Constant.Model.COLSAMPLE_BYTREE,
            objective        = Constant.Model.OBJECTIVE,
            scale_pos_weight = Constant.Model.SCALE_POS_WEIGHT,        
            seed             = Constant.Model.SEED     
        ),
        
        param_grid = parameter_search_1,
        scoring    = Constant.Model.GridSearch.Tree.SCORING,
        cv         = Constant.Model.GridSearch.Tree.CV,
        verbose    = Constant.Model.GridSearch.Tree.VERBOSE,
        n_jobs     = 1,
        iid        = False
    )
    
    grid_search_1.fit ( x, t )
    
    # Report results.
    
    for e in grid_search_1.grid_scores_:
        log ( str (e) , indent = indent+1 )
    
    min_child_weight = grid_search_1.best_params_ [ 'min_child_weight' ]
    max_depth        = grid_search_1.best_params_ [ 'max_depth'        ]
    
    log ( 'Optimal min child weight = ' + str ( min_child_weight           ), indent = indent+1 )
    log ( 'Optimal max depth        = ' + str ( max_depth                  ), indent = indent+1 )
    log ( 'Best Score               = ' + str ( grid_search_1.best_score_  ), indent = indent+1 )
        
    # update model using optimized parameters.
    
    model.set_params ( min_child_weight = min_child_weight )
    model.set_params ( max_depth        = max_depth )
    
    # Return updated model.
    
    return model

#-----------------------------------------------------------------------------
# FUNCTION: Optimize gamma.
#-----------------------------------------------------------------------------

def optimize_gamma ( model, x, t ):
    
    # Local variables.
    
    indent = 3
    
    log ( 'Optimizing gamma:', indent = indent )
    
    # Initialize search parameters.

    gamma_min    = 0.0
    gamma_max    = 0.5  
    gamma_stride = 0.1    
    gamma_range  = np.arange ( gamma_min, gamma_max + gamma_stride, gamma_stride )

    data_columns = [ 'gamma', 'time', 'logloss' ]        
    result_table = pd.DataFrame ( columns = data_columns )
    
    index = 0
    for gamma in gamma_range:
    
        logloss_test_mean, elapsed_time_formatted = compute_gamma ( model, x, t, gamma )
        
        data  = [ gamma, elapsed_time_formatted, logloss_test_mean ]        
                
        result_table.loc [ index ] = data
        index += 1
        
    # Select optimal gamma
            
    logloss_min       = result_table [ 'logloss' ].min()
    logloss_min_index = result_table [ 'logloss' ].argmin()
    row               = result_table.loc [ logloss_min_index ]        
    optimal_gamma     = row [ 'gamma' ]
    
    # Set iptimal learning rate and estimator count
    
    model.set_params ( gamma = optimal_gamma )
    
    # Print results table.
    
    log ( 'Optimal gamma = ' + str ( optimal_gamma ), indent = indent+1 )
    log ( 'Logloss       = ' + str ( logloss_min   ), indent = indent+1 )
<<<<<<< HEAD
=======

    # Return updated model.    
    
    return model

#-----------------------------------------------------------------------------
# FUNCTION: Optimize sub sample and column sample by tree.
#-----------------------------------------------------------------------------

def optimize_subsample_and_colsample_bytree ( model, x, t ):
    
    # Local variables.
    
    indent = 3
    
    log ( 'Optimizing sub-sample and column sample by tree:', indent = indent )
    
    # Initialize ssearch parameters.
    
    subsample_min    = 0.1
    subsample_max    = 0.4  
    subsample_stride = 0.01
    
    colsample_bytree_min    = 0.05
    colsample_bytree_max    = 0.1    
    colsample_bytree_stride = 0.01
    
    # Configure grid search.
    
    parameter_search_1 = {
        'subsample'        : np.arange ( subsample_min,        subsample_max,        subsample_stride        ),
        'colsample_bytree' : np.arange ( colsample_bytree_min, colsample_bytree_max, colsample_bytree_stride )
    }
     
    # Perform grid search.
    
    grid_search_1 = GridSearchCV (
    
        estimator = XGBClassifier (
            learning_rate    = Constant.Model.LEARNING_RATE,
            n_estimators     = Constant.Model.N_ESTIMATORS,
            max_depth        = Constant.Model.MAX_DEPTH,
            min_child_weight = Constant.Model.MIN_CHILD_WEIGHT,
            gamma            = Constant.Model.GAMMA,
            subsample        = Constant.Model.SUBSAMPLE,
            colsample_bytree = Constant.Model.COLSAMPLE_BYTREE,
            objective        = Constant.Model.OBJECTIVE,
            scale_pos_weight = Constant.Model.SCALE_POS_WEIGHT,        
            seed             = Constant.Model.SEED     
        ),
        
        param_grid = parameter_search_1,
        scoring    = Constant.Model.GridSearch.Tree.SCORING,
        cv         = Constant.Model.GridSearch.Tree.CV,
        verbose    = Constant.Model.GridSearch.Tree.VERBOSE,
        n_jobs     = 1,
        iid        = False
    )
    
    grid_search_1.fit ( x, t )
    
    # Report results.
    
    if False:
        for e in grid_search_1.grid_scores_:
            log ( str (e) , indent = indent+1 )
    
    subsample        = grid_search_1.best_params_ [ 'subsample'        ]
    colsample_bytree = grid_search_1.best_params_ [ 'colsample_bytree' ]
    
    log ( 'Optimal subsample        = ' + str ( subsample                  ), indent = indent+1 )
    log ( 'Optimal colsample_bytree = ' + str ( colsample_bytree           ), indent = indent+1 )
    log ( 'Best Score               = ' + str ( grid_search_1.best_score_  ), indent = indent+1 )
        
    # update model using optimized parameters.
    
    model.set_params ( subsample        = subsample )
    model.set_params ( colsample_bytree = colsample_bytree )
    
    # Return updated model.
    
    return model


>>>>>>> dev

    # Return updated model.    
    
    return model

#-----------------------------------------------------------------------------
<<<<<<< HEAD
# FUNCTION: Optimize regularization alpha.
#-----------------------------------------------------------------------------

def optimize_regularization_alpha ( model, x, t ):
    
    # Local variables.
    
    indent = 3
    
    log ( 'Optimizing regularization alpha:', indent = indent )
    
    # Initialize search parameters.

    reg_alpha_range  = [ 1*(10**i) for i in range (-1,5)]

    data_columns = [ 'reg_alpha', 'time', 'logloss' ]        
    result_table = pd.DataFrame ( columns = data_columns )
    
    index = 0
    for reg_alpha in reg_alpha_range:
    
        logloss_test_mean, elapsed_time_formatted = compute_regularization_alpha ( model, x, t, reg_alpha )
        
        data  = [ reg_alpha, elapsed_time_formatted, logloss_test_mean ]        
                
        result_table.loc [ index ] = data
        index += 1
        
    # Select optimal gamma
            
    logloss_min       = result_table [ 'logloss' ].min()
    logloss_min_index = result_table [ 'logloss' ].argmin()
    row               = result_table.loc [ logloss_min_index ]        
    optimal_reg_alpha = row [ 'reg_alpha' ]
    
    # Set iptimal learning rate and estimator count
    
    model.set_params ( reg_alpha = optimal_reg_alpha )
    
    # Print results table.
    
    log ( 'Optimal regularization alpha = ' + str ( optimal_reg_alpha ), indent = indent+1 )
    log ( 'Logloss                      = ' + str ( logloss_min       ), indent = indent+1 )

    # Return updated model.    
    
    return model

#-----------------------------------------------------------------------------
# FUNCTION: Optimize sub sample and column sample by tree.
#-----------------------------------------------------------------------------

def optimize_subsample_and_colsample_bytree ( model, x, t ):
    
    # Local variables.
    
    indent = 3
    
    log ( 'Optimizing sub-sample and column sample by tree:', indent = indent )
    
    # Initialize ssearch parameters.
    
    subsample_min    = 0.1
    subsample_max    = 0.4  
    subsample_stride = 0.01
    
    colsample_bytree_min    = 0.05
    colsample_bytree_max    = 0.1    
    colsample_bytree_stride = 0.01
    
    # Configure grid search.
    
    parameter_search_1 = {
        'subsample'        : np.arange ( subsample_min,        subsample_max,        subsample_stride        ),
        'colsample_bytree' : np.arange ( colsample_bytree_min, colsample_bytree_max, colsample_bytree_stride )
    }
     
    # Perform grid search.
    
    grid_search_1 = GridSearchCV (
    
        estimator = XGBClassifier (
            learning_rate    = Constant.Model.LEARNING_RATE,
            n_estimators     = Constant.Model.N_ESTIMATORS,
            max_depth        = Constant.Model.MAX_DEPTH,
            min_child_weight = Constant.Model.MIN_CHILD_WEIGHT,
            gamma            = Constant.Model.GAMMA,
            subsample        = Constant.Model.SUBSAMPLE,
            colsample_bytree = Constant.Model.COLSAMPLE_BYTREE,
            objective        = Constant.Model.OBJECTIVE,
            scale_pos_weight = Constant.Model.SCALE_POS_WEIGHT,        
            seed             = Constant.Model.SEED     
        ),
        
        param_grid = parameter_search_1,
        scoring    = Constant.Model.GridSearch.Tree.SCORING,
        cv         = Constant.Model.GridSearch.Tree.CV,
        verbose    = Constant.Model.GridSearch.Tree.VERBOSE,
        n_jobs     = 1,
        iid        = False
    )
    
    grid_search_1.fit ( x, t )
    
    # Report results.
    
    if False:
        for e in grid_search_1.grid_scores_:
            log ( str (e) , indent = indent+1 )
    
    subsample        = grid_search_1.best_params_ [ 'subsample'        ]
    colsample_bytree = grid_search_1.best_params_ [ 'colsample_bytree' ]
    
    log ( 'Optimal subsample        = ' + str ( subsample                  ), indent = indent+1 )
    log ( 'Optimal colsample_bytree = ' + str ( colsample_bytree           ), indent = indent+1 )
    log ( 'Best Score               = ' + str ( grid_search_1.best_score_  ), indent = indent+1 )
        
    # update model using optimized parameters.
    
    model.set_params ( subsample        = subsample )
    model.set_params ( colsample_bytree = colsample_bytree )
    
    # Return updated model.
    
    return model




#-----------------------------------------------------------------------------
# FUNCTION: Optimize tree count.
#-----------------------------------------------------------------------------

def compute_estimator_count ( model, x, t, r ):
    
    # Local Constants.
    
    ESTIMATOR_COUNT   = 0
    LOGLOSS_TEST_MEAN = 0
    
=======
# FUNCTION: Optimize tree count.
#-----------------------------------------------------------------------------

def compute_estimator_count ( model, x, t, r ):
    
    # Local Constants.
    
    ESTIMATOR_COUNT   = 0
    LOGLOSS_TEST_MEAN = 0
    
>>>>>>> dev
    # Local variables.
    
    verbose_eval = 'None'
    #verbose_eval = 20
    
    # Set learning rate.
    
    model.set_params ( learning_rate = r )
    
    # Compute estimator count.
    
    log ( 'Computing optimal estimator count.' + ' learning rate = ' + str(r), indent = 3 )
        
    xgb_parameters = model.get_xgb_params ()
    xgb_data       = xgb.DMatrix          ( data = x.values, label = t.values )
    
    cross_validation_result = xgb.cv (
    
        xgb_parameters, 
        xgb_data, 
        num_boost_round       = model.get_params() [ 'n_estimators' ],
        nfold                 = Constant.Model.CROSS_VALIDATION_FOLD_COUNT,
        metrics               = Constant.Model.METRIC,
        early_stopping_rounds = Constant.Model.EARLY_STOPPING_COUNT,
        verbose_eval          = verbose_eval
    )
    
    estimator_count   = cross_validation_result.shape [ ESTIMATOR_COUNT ]
    logloss_test_mean = cross_validation_result.loc [ estimator_count - 1 ][ LOGLOSS_TEST_MEAN ]
            
    # Compute elepsed time for cross validation.
    
    time_now               = time.time()
    elapsed_time_formatted = time_to_string ( time_now - time_event )
    
    log ( 'Cross validation: Estimator count   = ' + str ( estimator_count   ), indent = 4 )
    log ( 'Cross validation: Logloss test mean = ' + str ( logloss_test_mean ), indent = 4 )
    log ( 'Cross validation: elapsed_time      = ' + elapsed_time_formatted,    indent = 4 )
        
    return estimator_count, logloss_test_mean, elapsed_time_formatted 

#-----------------------------------------------------------------------------
# FUNCTION: Optimize tree count.
#-----------------------------------------------------------------------------

<<<<<<< HEAD
def compute_regularization_alpha ( model, x, t, reg_alpha ):
    
    # Local Constants.
    
    ESTIMATOR_COUNT_INDEX = 0
    
    # Local variables.
    
    #verbose_eval = 'None'
    verbose_eval = 1
    indent       = 4
    
    # Set learning rate.
    
    model.set_params ( reg_alpha = reg_alpha )
    
    # Compute estimator count.
    
    log ( 'Computing optimal regularization_alpha.' + ' reg_alpha = ' + str ( reg_alpha ), indent = indent )
        
    xgb_parameters = model.get_xgb_params ()
    xgb_data       = xgb.DMatrix          ( data = x.values, label = t.values )
    
    cross_validation_result = xgb.cv (
    
        xgb_parameters, 
        xgb_data, 
        num_boost_round       = model.get_params() [ 'n_estimators' ],
        nfold                 = Constant.Model.CROSS_VALIDATION_FOLD_COUNT,
        metrics               = Constant.Model.METRIC,
        early_stopping_rounds = Constant.Model.EARLY_STOPPING_COUNT,
        verbose_eval          = verbose_eval
    )
    
    estimator_count   = cross_validation_result.shape [ ESTIMATOR_COUNT_INDEX ]
    logloss_list      = np.array ( [ cross_validation_result.loc[i][0] for i in range ( 0, estimator_count ) ] )
    logloss_test_mean = logloss_list.min()
    
    # Compute elepsed time for cross validation.
    
    time_now               = time.time()
    elapsed_time_formatted = time_to_string ( time_now - time_event )
    
    log ( 'Cross validation: reg_alpha         = ' + str ( reg_alpha ),         indent = indent+1 )
    log ( 'Cross validation: Logloss test mean = ' + str ( logloss_test_mean ), indent = indent+1 )
    log ( 'Cross validation: elapsed_time      = ' + elapsed_time_formatted,    indent = indent+1 )
        
    return logloss_test_mean, elapsed_time_formatted

#-----------------------------------------------------------------------------
# FUNCTION: Optimize tree count.
#-----------------------------------------------------------------------------

def compute_gamma ( model, x, t, gamma ):
    
    # Local Constants.
    
    ESTIMATOR_COUNT = 0
    
    # Local variables.
    
=======
def compute_gamma ( model, x, t, gamma ):
    
    # Local Constants.
    
    ESTIMATOR_COUNT = 0
    
    # Local variables.
    
>>>>>>> dev
    verbose_eval = 'None'
    #verbose_eval = 20
    indent = 4
    
    # Set learning rate.
    
    model.set_params ( gamma = gamma )
    
    # Compute estimator count.
    
    log ( 'Computing optimal gamma.' + ' gamma = ' + str ( gamma ), indent = indent )
        
    xgb_parameters = model.get_xgb_params ()
    xgb_data       = xgb.DMatrix          ( data = x.values, label = t.values )
    
    cross_validation_result = xgb.cv (
    
        xgb_parameters, 
        xgb_data, 
        num_boost_round       = model.get_params() [ 'n_estimators' ],
        nfold                 = Constant.Model.CROSS_VALIDATION_FOLD_COUNT,
        metrics               = Constant.Model.METRIC,
        early_stopping_rounds = Constant.Model.EARLY_STOPPING_COUNT,
        verbose_eval          = verbose_eval
    )
    
    estimator_count   = cross_validation_result.shape [ ESTIMATOR_COUNT ]
    logloss_list      = np.array ( [ cross_validation_result.loc[i][0] for i in range ( 0, estimator_count ) ] )
    logloss_test_mean = logloss_list.min()
    
    # Compute elepsed time for cross validation.
    
    time_now               = time.time()
    elapsed_time_formatted = time_to_string ( time_now - time_event )
    
    log ( 'Cross validation: gamma             = ' + str ( gamma ),             indent = indent+1 )
    log ( 'Cross validation: Logloss test mean = ' + str ( logloss_test_mean ), indent = indent+1 )
    log ( 'Cross validation: elapsed_time      = ' + elapsed_time_formatted,    indent = indent+1 )
        
    return logloss_test_mean, elapsed_time_formatted

#-----------------------------------------------------------------------------
# FUNCTION: Fit model.
#-----------------------------------------------------------------------------

def fit_model ( model, x, t ):
    
    # Local constants.
    
    INDENT = 1    
    
    # Fit model.
    
    log ( 'Fitting model to training data.', indent = INDENT )
    
    if Constant.Application.FIT_MODEL_ENBALED:
        
        log ( 'Training model.', indent = INDENT+1 )
        
        # Train the model using current model parameters.
    
        model.fit ( x, t, eval_metric = Constant.Model.METRIC )
            
        # Predict training set.
            
        t_predictions              = model.predict       ( x )
        t_prediction_probabilities = model.predict_proba ( x ) [ :, 1 ]
        
        # Print model report.
        
        report_training_results ( model, t, t_predictions, t_prediction_probabilities )
        feature_rank = pd.Series ( model.booster().get_fscore()).sort_values ( ascending = False )
        
        # Plot feature ranking.
        
        if Constant.Application.PLOT_FEATURE_RANK_ENABLED:
            
            log ( 'REPORTING: Plotting feature rank.', indent = INDENT+1 )
            
            plot_feature_rank ( feature_rank )
            
        else:
            log ( 'Feature rank plot: DISABLED', indent = INDENT+1 )
		
    else:
        
        log ( 'Fit Model: DISABLED', indent = INDENT+1 )
    
    return model

#-----------------------------------------------------------------------------
# FUNCTION: test model..
#-----------------------------------------------------------------------------

def test_model ( model, x, t ):
    
    log ( 'Testing trained model.', indent = 1 )
    
    accuracy = 0
    auc      = 0
    logloss  = 0
    
    if Constant.Application.TEST_MODEL:
        
        pass        
        
    else:
        
        log ( 'Test Model: DISABLED', indent = 2 )        
    
    return accuracy, auc, logloss

#-----------------------------------------------------------------------------
# FUNCTION: Report model results.
#-----------------------------------------------------------------------------

def report_model ( model, accuracy, auc, logloss ):
    
    log ( 'Compiling training performance report.', indent = 1 )
    
    if Constant.Application.TRAINING_REPORT_ENABLED:
        
        pass
    
    else:
        log ( 'Traning Report: DISABLED', indent = 2 )        
        
    return

#-----------------------------------------------------------------------------
# FUNCTION: apply_model
#-----------------------------------------------------------------------------

def apply_model ( model ):
    
    log ( 'APPLICATION SEQUENCE:' )
    
    if Constant.Application.MODEL_APPLICATION_ENABLED:
        
        # Compile file names.
        
        application_file_name = Constant.Numerai.DataFile.PATH + Constant.Numerai.DataFile.APPLICATION    
        prediction_file_name  = Constant.Numerai.DataFile.PATH + Constant.Numerai.DataFile.PREDICTION 
        
        # Load application data.    
        
        i, x = load_application_data ( application_file_name )
    
        # Apply model.
    
        y = predict ( model, x )
        
        # Save results.
        
        save_prediction_data ( prediction_file_name, i, y )
        
    else:
        
        log ( 'Application: DISABLED', indent = 1 )
        
        

#-----------------------------------------------------------------------------
# FUNCTION: test model..
#-----------------------------------------------------------------------------

def predict ( model, x ):
    
    log ( 'Applying model to production data.', indent = 1 )
    
    winsound.Beep ( Constant.Sound.START_FREQUENCY, Constant.Sound.START_PERIOD )
    
    y = model.predict_proba ( x )   
    
    return y

#-----------------------------------------------------------------------------
# FUNCTION: Load training data.
#-----------------------------------------------------------------------------

def load_training_data ( file_name, row_count = -1 ):
    
    log ( 'Loading training data: ' + '"' + file_name + '"', indent = 1 )
        
    # Load training data.
        
    training_data = pd.read_csv ( file_name )    
    
    # Reduce training data samples for configuration testing.
    
    if row_count > 0:    
        log ( 'Traning Data: training_data_limit_enabled = TRUE', indent = 2 )
        training_data = training_data.head ( row_count - 1 )
    
    log ( 'Traning Data: row_count = ' + str ( len ( training_data.index ) + 1 ), indent = 2 )
    
    # Seperate training data columns into features x, and target/s t.
    
    index_feature_start = 1    
    index_feature_count = 22
    col_features        = [ Constant.Numerai.CSV.FEATURE + str ( index ) for index in range ( index_feature_start, index_feature_count ) ]
    col_target          = Constant.Numerai.CSV.TARGET
    
    x = training_data [ col_features ]
    t = training_data [ col_target   ]
    
    return x, t

#-----------------------------------------------------------------------------
# Load application data.
#-----------------------------------------------------------------------------

def load_application_data ( file_name ):
    
    log ( 'Loading application data: ' + '"' + file_name + '"', indent = 1 )
        
    # Load application data from file.

    application_data = pd.read_csv ( file_name )
    
    # Prepare data for execution. y_application = f ( x_application )
    # - Input vector = x_application
    # - Output vector = y_application ...To be allocated after model execution.
    
    i = application_data [ [ Constant.Numerai.CSV.ID ] ]
    x = application_data.drop ( Constant.Numerai.CSV.ID, axis = 1 )    
    
    log ( 'Application Data: row_count = ' + str ( len ( application_data.index ) + 1 ), indent = 2 )
    
    return i, x

#-----------------------------------------------------------------------------
# Save application results.
#-----------------------------------------------------------------------------

def save_prediction_data ( file_name, i, y ):
    
    log ( 'Saving application data: ' + '"' + file_name + '"', indent = 1 )
    
    # Isolate propability of 1.0.

    p = pd.DataFrame ( y ) [ 1 ]
    
    # Create prediction data table.        
    
    prediction_data         = pd.concat ( [ i, p ], axis=1 )
    prediction_data.columns = [ Constant.Numerai.CSV.ID, Constant.Numerai.CSV.PROBABILITY ]
        
    # Save the results to file.    
    
    prediction_data.to_csv ( Constant.Numerai.DataFile.PATH + file_name, index = None )
        
    
#-----------------------------------------------------------------------------
# DEBUG FUNCTION: Show sample data
#-----------------------------------------------------------------------------


def debug_show_sample_data ( data, features, target, row_count = 3, precision = 16):
        
    # Display transposed list of first few rows.
    
    format_string = '{:,.' + str ( precision ) + 'f}'

    print ( 'FEATURES:')
    pd.options.display.float_format = format_string.format       
    print ( data [ features ].head ( row_count ).transpose() )
    
    print ( '\nTARGETS:')        
    print ( data [ target ].head ( row_count ).transpose() )


#/////////////////////////////////////////////////////////////////////////////
# Utility Functions.
#/////////////////////////////////////////////////////////////////////////////

#-----------------------------------------------------------------------------
# FUNCTION: Time string.
#-----------------------------------------------------------------------------

def time_to_string ( time_sample ):
    
    hours, remainder = divmod ( time_sample, 3600 )
    minutes, seconds = divmod ( remainder,   60   )
    time_string      = '{:0>2}:{:0>2}:{:0>6.3f}'.format ( int ( hours ), int ( minutes ), seconds )    
    
    return time_string


#-----------------------------------------------------------------------------
# FUNCTION: Print Pandas data frame.
#-----------------------------------------------------------------------------

def print_data_frame ( data, indent ):
<<<<<<< HEAD

    # print header.
    
    r       = data.columns[0]
    n       = data.columns[1]
    t       = data.columns[2]
    logloss = data.columns[3]
    
    s = '{:<8}{:<8}{:<16}{:<16}'.format ( r, n, t, logloss )
    log ( s, indent )
    
    # Print data.
    
    for row in data.values:
        r       = row[0]
        n       = row[1] 
        t       = row[2]
        logloss = row[3]
        s = '{:<8}{:<8}{:<16}{:<16.6f}'.format ( r, int(n), t, logloss )
        log ( s, indent )

#-----------------------------------------------------------------------------
# FUNCTION: Print model parameters.
#-----------------------------------------------------------------------------

def print_model_parameters ( model, indent ):

    for key in model.get_params():                   
        parameter_str  = '{0:.<24}'.format ( key )
        parameter_str += ' = '
        parameter_str += str ( model.get_params() [ key ] )
        log ( parameter_str, indent = indent )

=======

    # print header.
    
    r       = data.columns[0]
    n       = data.columns[1]
    t       = data.columns[2]
    logloss = data.columns[3]
    
    s = '{:<8}{:<8}{:<16}{:<16}'.format ( r, n, t, logloss )
    log ( s, indent )
    
    # Print data.
    
    for row in data.values:
        r       = row[0]
        n       = row[1] 
        t       = row[2]
        logloss = row[3]
        s = '{:<8}{:<8}{:<16}{:<16.6f}'.format ( r, int(n), t, logloss )
        log ( s, indent )

#-----------------------------------------------------------------------------
# FUNCTION: Print model parameters.
#-----------------------------------------------------------------------------

def print_model_parameters ( model, indent ):

    for key in model.get_params():                   
        parameter_str  = '{0:.<24}'.format ( key )
        parameter_str += ' = '
        parameter_str += str ( model.get_params() [ key ] )
        log ( parameter_str, indent = indent )

>>>>>>> dev
#-----------------------------------------------------------------------------
# Plot model data to console.
#-----------------------------------------------------------------------------

def report_training_results ( model, t, t_predictions, t_prediction_probabilities ):
    
    CONSOLE_ALIGN_KEY = '{0:.<24}'
    INDENT_HEADER     = 2
    INDENT_DATA       = 3
        
    accuracy = metrics.accuracy_score ( t.values, t_predictions              ) * 100
    auc      = metrics.roc_auc_score  ( t,        t_prediction_probabilities )
    logloss  = metrics.log_loss       ( t,        t_prediction_probabilities )
        
    log ( 'Training Results:', indent = INDENT_HEADER )
    log ( CONSOLE_ALIGN_KEY.format ( 'Accuracy' ) + ' = ' + '{:.3f}'.format ( accuracy ), indent = INDENT_DATA )
    log ( CONSOLE_ALIGN_KEY.format ( 'AUC' )      + ' = ' + '{:.6f}'.format ( auc ),      indent = INDENT_DATA )
    log ( CONSOLE_ALIGN_KEY.format ( 'logloss' )  + ' = ' + '{:.6f}'.format ( logloss ),  indent = INDENT_DATA )
    


#-----------------------------------------------------------------------------
# FUNCTION: log message to console.
#-----------------------------------------------------------------------------

def log ( message, indent = 0, newline = True, frequency = 0 ):
    
    # Loacal constants.
    
    INDENT_STRING = Constant.Text.INDENT
    
    # Compile time string.
    
    time_now               = time.time()
    elapsed_time_formatted = time_to_string ( time_now - time_event )
    
    # Compile indentation string.
    
    indent_string = INDENT_STRING * indent
    
    # Compile string line prefix.
    
    line_prefix = '[' + elapsed_time_formatted + '] '
    
    # Print message.    
    
    if newline:
        print ( line_prefix + indent_string + message )
    else:
        print ( line_prefix + indent_string + message, end = '' )
        
    # Play sound.
        
    if frequency == 0:
        winsound.Beep ( Constant.Sound.EVENT_FREQUENCY, Constant.Sound.EVENT_PERIOD )
    else:
        winsound.Beep ( frequency, 70 )

#/////////////////////////////////////////////////////////////////////////////
# Program entry point.
#/////////////////////////////////////////////////////////////////////////////

def main():
    
    # Initialize application.    
    
    log ( 'PROGRAM.START: ' + str ( datetime.datetime.now() ) )
    winsound.Beep ( Constant.Sound.START_FREQUENCY, Constant.Sound.START_PERIOD )
<<<<<<< HEAD
    
    # Initialize Model.
    
    model = initialize_model ()    
            
    # Tain model.
    
=======
    
    # Initialize Model.
    
    model = initialize_model ()    
            
    # Tain model.
    
>>>>>>> dev
    model = train_model ( model )
        
    # Apply model.
    
    apply_model ( model )
    
    # Shut down application.
        
    log ( 'PROGRAM.STOP: ' + str ( datetime.datetime.now() ) )
    print ('')
    winsound.Beep ( Constant.Sound.STOP_FREQUENCY, Constant.Sound.STOP_PERIOD )    
    
    #debug_show_sample_data ( training_data, cols_features, col_target, row_count = 3, precision = 16 )

if __name__ == "__main__":

     main()
   
#/////////////////////////////////////////////////////////////////////////////
