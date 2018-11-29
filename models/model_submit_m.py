import pandas as pd
import numpy as np 
from sklearn import svm  
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold 
from sklearn.cross_validation import KFold as kfo
import xgboost as xgb 
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline 
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from mlxtend.regressor import StackingCVRegressor
from lightgbm import LGBMRegressor 
import feature_list 

    
def select_drop_standand(traindata, testdata, num):
    # select features, from feature_list.py
    if num == 1:
        selected, select_list = feature_list.select_feature1(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature1(testdata, False)
    if num == 2:
        selected, select_list = feature_list.select_feature2(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature2(testdata, False)
    if num == 3:
        selected, select_list = feature_list.select_feature3(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature3(testdata, False)
    if num == 4:
        selected, select_list = feature_list.select_feature4(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature4(testdata, False)
    if num == 5:
        selected, select_list = feature_list.select_feature5(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature5(testdata, False)
    if num == 6:
        selected, select_list = feature_list.select_feature6(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature6(testdata, False)
    if num == 7:
        selected, select_list = feature_list.select_feature7(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature7(testdata, False)
    if num == 8:
        selected, select_list = feature_list.select_feature8(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature8(testdata, False)
    if num == 9:
        selected, select_list = feature_list.select_feature9(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature9(testdata, False)
    if num == 10:
        selected, select_list = feature_list.select_feature10(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature10(testdata, False)
    if num == 11:
        selected, select_list = feature_list.select_feature11(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature11(testdata, False)
    if num == 12:
        selected, select_list = feature_list.select_feature12(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature12(testdata, False)
    if num == 13:
        selected, select_list = feature_list.select_feature13(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature13(testdata, False)
    if num == 14:
        selected, select_list = feature_list.select_feature14(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature14(testdata, False)
    if num == 15:
        selected, select_list = feature_list.select_feature15(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature15(testdata, False)
    if num == 16:
        selected, select_list = feature_list.select_feature16(traindata, True) 
        selected_testB_features, select_list_testB = feature_list.select_feature16(testdata, False)
    selected.reset_index(drop=True, inplace=True) 
    selected_testB_features.reset_index(drop=True, inplace=True)

    # clear empty row
    selected_nonan = selected.dropna(axis=0, how='any')
    train_targets = pd.DataFrame(selected_nonan['charge_energy'], columns=['charge_energy'])
    train_nonan_features = selected_nonan.drop(['charge_energy'], axis=1)
    train_test_features = pd.concat([train_nonan_features, selected_testB_features], axis=0)
    train_test_features.reset_index(drop=True, inplace=True)
    
    # RobustScaler quantile_range=(25.0, 75.0)  # Standardization based on quantile  
    select_list.remove('charge_energy')
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    n_X_train_test = x_scaler.fit_transform(np.array(train_test_features))
#    n_y_train = y_scaler.fit_transform(np.log1p(np.array(train_targets))) # ln(x+1) Transformation
    n_y_train = y_scaler.fit_transform(np.array(train_targets)) 
    n_X_train_test_pd = pd.DataFrame(n_X_train_test, columns=select_list)
    n_X_train_test_mer = n_X_train_test_pd.copy()
    
    # Time dimension sparse matrix
#    chargemode_dummies = pd.get_dummies(train_test_features['charge_mode'], prefix='mode', prefix_sep='_')
#    hour_dummies = pd.get_dummies(train_test_features['hour'], prefix='hour', prefix_sep='_')
#    week_dummies = pd.get_dummies(train_test_features['week'], prefix='week', prefix_sep='_') 
#    month_dummies = pd.get_dummies(train_test_features['month'], prefix='month', prefix_sep='_')  
#    if 'phase' in select_list:
#        phase_dummies = pd.get_dummies(train_test_features['phase'], prefix='phase', prefix_sep='_')
#        n_X_train_test_mer = pd.concat([n_X_train_test_pd, chargemode_dummies, hour_dummies, week_dummies, month_dummies,phase_dummies], axis=1)
#        n_X_train_test_mer.drop(['charge_mode', 'hour', 'week', 'month', 'phase'], axis=1, inplace=True)
#    else:
#        n_X_train_test_mer = pd.concat([n_X_train_test_pd, chargemode_dummies, hour_dummies, week_dummies, month_dummies], axis=1)
#        n_X_train_test_mer.drop(['charge_mode', 'hour', 'week', 'month'], axis=1, inplace=True)
    
    n_testB = n_X_train_test_mer.tail(selected_testB_features.shape[0])
    n_X_train = n_X_train_test_mer.drop(n_testB.index.tolist())
    return n_X_train, n_y_train, n_testB, y_scaler

    
ram_num = 5
kfolds = KFold(n_splits=10, shuffle=True, random_state=ram_num)
def cv_rmse(model, train, y_train):  
    rmse = np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = kfolds))
    return(rmse)
 
    
def ridge_selector(k, X, y):
    model = make_pipeline(RidgeCV(alphas = [k], cv=kfolds)).fit(X, y) 
    rmse = cv_rmse(model, X, y).mean()
    return(rmse)
    
    
def lasso_selector(k, X, y):  
    model = make_pipeline(LassoCV(max_iter=1e7, alphas = [k], 
                                 cv = kfolds)).fit(X, y) 
    rmse = cv_rmse(model, X, y).mean()
    return(rmse) 
 
    
def stack_level1(clf, x_train, y_train, x_test, kf):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    level1_train = np.zeros((num_train,))
    level1_test = np.zeros((num_test,))
    level1_test_kfold = np.empty((10, num_test)) # kfold = 10

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        level1_train[test_index] = clf.predict(x_te)
        level1_test_kfold[i, :] = clf.predict(x_test)

    level1_test[:] = level1_test_kfold.mean(axis=0)
    return level1_train.reshape(-1, 1), level1_test.reshape(-1, 1)


if __name__ == '__main__':  
    # Read the data of 16 cars separately. features + target
    readFile_carfeatures = []
    readFile_testfeatures = [] 
    car_train_list = []
    car_test_list = []
    filenum = 17 
    
    for i in range(1,filenum):
        readFile_carfeatures.append('../dataset/feature/train_feature/car' + str(i) + '_features.csv') 
    for i in range(1,filenum):
        readFile_testfeatures.append('../dataset/feature/test_feature/car' + str(i) + 'testB_features.csv') 
        
    # train features + target
    for i in range(len(readFile_carfeatures)):
        car_train = pd.read_csv(readFile_carfeatures[i], dtype={'charge_start_time': str, 'charge_end_time': str}) 
        car_train_list.append(car_train) 
    # test features 
    for i in range(len(readFile_carfeatures)):
        car_test = pd.read_csv(readFile_testfeatures[i]) 
        car_test_list.append(car_test) 
    
    car_index = 9 # 0 = car1
    car_train = pd.read_csv(readFile_carfeatures[car_index], dtype={'charge_start_time': str, 'charge_end_time': str})
    car_test = pd.read_csv(readFile_testfeatures[car_index]) 
    
    # Differentiate fast and slow charging
#    car_train = car_train[car_train['charge_mode'].isin([0])]  # car4 fastcharging mode=2 
#    car_test = car_test[car_test['charge_mode'].isin([0])] 
    # Differentiate battery properties
#    car_train = car_train[car_train['phase'].isin([1])]
#    car_test = car_test[car_test['phase'].isin([1])] 
 
    norm_X_train, norm_y_train, norm_test, y_scal = select_drop_standand(car_train, car_test, num=car_index+1)
    train_target = np.ravel(np.array(norm_y_train))
    
    
     
    # 6 base models
    # Ridge model 
    r_alphas_best = [0.0008] 
    ridge = make_pipeline(RidgeCV(alphas = r_alphas_best, cv = kfolds)).fit(norm_X_train, train_target)    
    
    # Lasso model  
    l_alphas_best = [0.00001] 
    lasso = make_pipeline(LassoCV(max_iter=1e7, alphas = l_alphas_best, cv = kfolds)).fit(norm_X_train, train_target) 
    
    # elastic_model 
    e_alphas_best = [0.0001] 
    e_l1ratio_best = [5]  
    elastic = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e5, alphas=e_alphas_best, cv=kfolds, l1_ratio=e_l1ratio_best)) 
    elastic_model = elastic.fit(norm_X_train, train_target)  
    
    # xgboost_model   
    xgboost_model = xgb.XGBRegressor(max_depth=3, min_child_weight=0.9, gamma=0.0001,subsample=0.55,
                             scale_pos_weight=1, learning_rate=0.008, reg_alpha=0.001,colsample_bytree=0.9,
                             booster='gbtree', n_estimators=3000) 
    xgboost = make_pipeline(xgboost_model)
    xgb_score = cv_rmse(xgboost, norm_X_train, train_target)
    
    # gbm model    
    gbm_model = GradientBoostingRegressor(n_estimators=2000, max_depth=4, min_samples_split=2, 
                                          min_samples_leaf=2, max_features='auto', 
                                          subsample=0.6, learning_rate=0.008) 
    gbm = make_pipeline(gbm_model)
    
    # lightgbm model  
#    lightgmb_model = LGBMRegressor(objective='regression'
#                                   ,learning_rate=0.008, n_estimators=2000,
#                                   max_bin = 55, bagging_fraction = 0.9,
#                                   feature_fraction = 0.8,
#                                   min_data_in_leaf =6, 
#                                   min_sum_hessian_in_leaf = 0.7)
#    lightgbm = make_pipeline(lightgmb_model) 
    
    # SVM 
    svr_opt = svm.SVR(C=250, gamma=0.001) 
    svr = make_pipeline(svr_opt)  
    
    # stack model   
    stackX = np.array(norm_X_train)
    stacky = np.array(train_target)
    stacktest = np.array(norm_test) 
    kf = kfo(stackX.shape[0], n_folds=10, random_state=1)
    
    p_ridge, t_ridge = stack_level1(ridge, stackX, stacky, stacktest, kf) 
    p_lasso, t_lasso = stack_level1(lasso, stackX, stacky, stacktest, kf)
    p_enet, t_enet = stack_level1(elastic, stackX, stacky, stacktest, kf) 
    p_xgb, t_xgb = stack_level1(xgboost, stackX, stacky, stacktest, kf) 
    p_gbm, t_gbm = stack_level1(gbm, stackX, stacky, stacktest, kf) 
    p_svc, t_svc = stack_level1(svr, stackX, stacky, stacktest, kf) 
    
    train_level2 = np.concatenate((p_ridge, p_lasso, p_enet, p_xgb, p_gbm, p_svc), axis=1)
    test_level2 = np.concatenate((t_ridge, t_lasso, t_enet, t_xgb, t_gbm, t_svc), axis=1)
    
    # predict & submit
    stack_model = gbm_model.fit(train_level2, stacky)
    stack_preds_norm = stack_model.predict(test_level2)
    stack_preds = y_scal.inverse_transform(stack_preds_norm.reshape(-1, 1))[:, 0] 
    stack_preds_pd = pd.DataFrame(stack_preds, columns=['pred_energy'])
    stack_preds_pd.to_csv('../dataset/submit/stack_preds_car' + str(car_index+1) + '.csv') 
    
    