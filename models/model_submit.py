import pandas as pd
import numpy as np 
from sklearn import svm  
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold 
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
    # 选取特征, 特征在feature_list.py 中
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

    # 清理空行  
    selected_nonan = selected.dropna(axis=0, how='any')
    train_targets = pd.DataFrame(selected_nonan['charge_energy'], columns=['charge_energy'])
    train_nonan_features = selected_nonan.drop(['charge_energy'], axis=1)
    train_test_features = pd.concat([train_nonan_features, selected_testB_features], axis=0)
    train_test_features.reset_index(drop=True, inplace=True)
    
    # 注意标准化方法,RobustScaler quantile_range=(25.0, 75.0)  # 基于分位数标准化 features  
    select_list.remove('charge_energy')
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    n_X_train_test = x_scaler.fit_transform(np.array(train_test_features))
#    n_y_train = y_scaler.fit_transform(np.log1p(np.array(train_targets))) # ln(x+1)变换
    n_y_train = y_scaler.fit_transform(np.array(train_targets)) 
    n_X_train_test_pd = pd.DataFrame(n_X_train_test, columns=select_list)
    n_X_train_test_mer = n_X_train_test_pd.copy()
    
    # 时间维稀疏矩阵 
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
       
   
if __name__ == '__main__':  
    # 分别读取16个车的数据 features + target
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
    
    car_index = 0 # 0 = car1
    car_train = pd.read_csv(readFile_carfeatures[car_index], dtype={'charge_start_time': str, 'charge_end_time': str})
    car_test = pd.read_csv(readFile_testfeatures[car_index]) 
    
    # 区分快慢充电
#    car_train = car_train[car_train['charge_mode'].isin([0])]  # car4 快充数据mode=2 
#    car_test = car_test[car_test['charge_mode'].isin([0])] 
    # 区分电池阶段
#    car_train = car_train[car_train['phase'].isin([1])]
#    car_test = car_test[car_test['phase'].isin([1])] 
 
    norm_X_train, norm_y_train, norm_test, y_scal = select_drop_standand(car_train, car_test, num=car_index+1)
    train_target = np.ravel(np.array(norm_y_train))
     
    # 6个基学习器
    # Ridge model 
    r_alphas_best = [0.0008] 
    ridge = make_pipeline(RidgeCV(alphas = r_alphas_best, cv = kfolds)).fit(norm_X_train, train_target)    
    
    # Lasso model  
    l_alphas_best = [0.00001] 
    lasso = make_pipeline(LassoCV(max_iter=1e7, alphas = l_alphas_best, cv = kfolds)).fit(norm_X_train, train_target) 
    
    # elastic_model 
    e_alphas_best = [0.0001] 
    e_l1ratio_best = [5]  
    elastic = make_pipeline(RobustScaler(),
                            ElasticNetCV(max_iter=1e5, alphas=e_alphas_best,
                                         cv=kfolds, l1_ratio=e_l1ratio_best)) 
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
    stack_model1 = StackingCVRegressor(regressors=(ridge, lasso, elastic, xgboost, gbm, svr),
                                      meta_regressor=lasso, use_features_in_secondary=True, cv=kfolds) # level2: lasso
    stack_model2 = StackingCVRegressor(regressors=(ridge, lasso, elastic, xgboost, gbm, svr),
                                      meta_regressor=gbm, use_features_in_secondary=True, cv=kfolds) # level2: gbm
    
    stackX = np.array(norm_X_train)
    stacky = np.array(train_target)
    stacktest = np.array(norm_test)
    # stack model scoring 
    print("cross validated scores")
    for model, label in zip([ridge, lasso, elastic, xgboost, gbm, svr
#                             , stack_model1
                             ],
                            ['RidgeCV', 'LassoCV', 'ElasticNetCV', 'xgboost', 'gbm', 'svr'
#                             , 'StackingCVRegressor'
                             ]): 
        SG_scores = cross_val_score(model, stackX, stacky, cv=kfolds,
                                    scoring='neg_mean_squared_error')
        print("RMSE", np.sqrt(-SG_scores.mean()), "std", SG_scores.std(), label)
     
    stack_model_fit1 = stack_model1.fit(stackX, stacky) 
    stack_model_fit2 = stack_model2.fit(stackX, stacky) 
     
    stack_preds_norm1 = stack_model_fit1.predict(stacktest) 
    stack_preds_norm2 = stack_model_fit2.predict(stacktest) 
#    stack_preds = np.expm1(y_scal.inverse_transform(stack_preds_norm.reshape(-1, 1))[:, 0]) 
    stack_preds1 = y_scal.inverse_transform(stack_preds_norm1.reshape(-1, 1))[:, 0] 
    stack_preds2 = y_scal.inverse_transform(stack_preds_norm2.reshape(-1, 1))[:, 0] 
    stack_preds = (stack_preds1 + stack_preds2) / 2   
    stack_preds_pd = pd.DataFrame(stack_preds, columns=['pred_energy'])
    stack_preds_pd.to_csv('../dataset/submit/stack_preds_car' + str(car_index+1) + '.csv')
    
    stack_preds_v_norm = stack_model_fit1.predict(stackX)
    stack_preds_v = y_scal.inverse_transform(stack_preds_v_norm.reshape(-1, 1))[:, 0]
    stack_target_v = y_scal.inverse_transform(norm_y_train.reshape(-1, 1))[:, 0]
    stack_preds_v_pd = pd.DataFrame(stack_preds_v, columns=['pred_energy'])
    stack_y_v_pd = pd.DataFrame(stack_target_v, columns=['energy'])
    stack_preds_v_pd_ = pd.concat([stack_preds_v_pd, stack_y_v_pd], axis=1) 
    stack_preds_v_pd_.to_csv('../dataset/submit/preds_valid_car' + str(car_index+1) + '.csv')
    
    
    
    
    