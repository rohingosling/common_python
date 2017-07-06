# Model optimization functions.
import time

import numpy   as np
import pandas  as pd
import xgboost as xgb

from sklearn.grid_search import GridSearchCV
from xgboost.sklearn     import XGBClassifier
from numerai_constants   import Constant
from utility             import log, print_model_parameters, time_to_string, time_event, print_data_frame

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
    
    log ( 'Optimal regularization alpha  = ' + str ( reg_alpha                  ), indent = indent+1 )
    log ( 'Optimal regularization lambda = ' + str ( reg_lambda                 ), indent = indent+1 )
    log ( 'Best Score                    = ' + str ( grid_search_1.best_score_  ), indent = indent+1 )
        
    # update model using optimized parameters.
    
    model.set_params ( reg_alpha  = reg_alpha )
    model.set_params ( reg_lambda = reg_lambda )
    
    # Return updated model.
    
    return model


#-----------------------------------------------------------------------------
# FUNCTION: Optimize min_child_weight and max_depth
#-----------------------------------------------------------------------------

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

def compute_gamma ( model, x, t, gamma ):
    
    # Local Constants.
    
    ESTIMATOR_COUNT = 0
    
    # Local variables.
    
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
