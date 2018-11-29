import pandas as pd
import numpy as np 
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import learning_curve 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold 
from sklearn import linear_model
import xgboost as xgb 
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline 
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV 
from lightgbm import LGBMRegressor
import feature_list


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), verbose=0):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,
                                                                            n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

 
def select_drop_standand(traindata, testdata, num):
    # 选取特征
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
    
    # 区分快慢充电
#    car_train = car_train[car_train['charge_mode'].isin([0])] 
    # 区分电池阶段
#    car_train = car_train[car_train['phase'].isin([0])] 
    
    norm_X_train, norm_y_train, norm_test, y_scal = select_drop_standand(car_train, car_test, num=car_index+1)
    train_target = np.ravel(np.array(norm_y_train))
    
    # plot_learning_curve 
    r_alphas_best = {'alpha': 0.0008} 
    title = "Learning Curves Ridge" 
    plot_learning_curve(linear_model.Ridge(**r_alphas_best), title, norm_X_train, train_target, cv=5)
    
    l_alphas_best = {'alpha': 0.00001} 
    title = "Learning Curves Lasso" 
    plot_learning_curve(linear_model.Lasso(**l_alphas_best), title, norm_X_train, train_target, cv=5)
    
    ElasticNet_pram = {'alpha': 0.0001, 'l1_ratio': 5} 
    title = "Learning Curves ElasticNet" 
    plot_learning_curve(linear_model.ElasticNet(**ElasticNet_pram), title, norm_X_train, train_target, cv=5)
    
    gbm_param = {'n_estimators': 2000, 'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 2,
                          'max_features': 'auto', 'subsample': 0.6, 'learning_rate': 0.008}
    title = "Learning Curves gbm" 
    plot_learning_curve(GradientBoostingRegressor(**gbm_param), title, norm_X_train, train_target, cv=5)
    
    xgb_param = {'max_depth': 3, 'min_child_weight': 0.9, 'gamma': 0.0001, 'subsample': 0.55,
                          'scale_pos_weight': 1, 'learning_rate': 0.008, 'reg_alpha': 0.001,
                          'colsample_bytree': 0.9, 'booster': 'gbtree', 'n_estimators': 3000} 
    title = "Learning Curves xgb" 
    plot_learning_curve(xgb.XGBRegressor(**xgb_param), title, norm_X_train, train_target, cv=5) 
     
    svr_pram = {'C': 250, 'gamma': 0.001} 
    title = "Learning Curves SVR" 
    plot_learning_curve(svm.SVR(**svr_pram), title, norm_X_train, train_target, cv=5)
     
#### Ridge 选取最佳参数  
    r_alphas = [0,0.00001,0.0001,0.0008,0.001,0.005,0.1,0.4,1,10,15,20,30,40,50]
    r_alphas = [0,0.00001,0.0001,0.0008,0.001,0.005,0.1,0.4,1]
    ridge_scores = []
    for alpha in r_alphas:
        score = ridge_selector(alpha, norm_X_train, train_target)
        ridge_scores.append(score)
    ridge_score_table = pd.DataFrame(ridge_scores, r_alphas, columns=['Ridge_RMSE'])
    print(ridge_score_table)
    # 用最佳参数进行计算
    r_alphas_best = [0.0008]
    ridge = make_pipeline(RidgeCV(alphas = r_alphas_best, cv = kfolds))   
    ridge_model_score = cv_rmse(ridge, norm_X_train, train_target)  
    plt.plot(r_alphas, ridge_scores, label='Ridge')
    plt.legend('center')
    plt.xlabel('alpha')
    plt.ylabel('score')
    print("ridge cv score: {0:.6f}".format(ridge_model_score.mean()))   
    
#### Lasso 选取最佳参数
    l_alphas = [0.00001,0.0001,0.001,0.003,0.008,0.01,0.05,0.1,0.2]
    lasso_scores = []
    for alpha in l_alphas:
        score = lasso_selector(alpha, norm_X_train, train_target)
        lasso_scores.append(score) 
    lasso_score_table = pd.DataFrame(lasso_scores, l_alphas, columns=['Lasso_RMSE'])
    print(lasso_score_table) 
    # 用最佳参数进行计算
    l_alphas_best = [0.001]
    lasso = make_pipeline(LassoCV(max_iter=1e7, alphas = l_alphas_best, cv = kfolds)) 
    lasso_model_score = cv_rmse(lasso, norm_X_train, train_target) 
    print("Lasso cv score: {0:.6f}".format(lasso_model_score.mean()))
    
    lasso_model2 = make_pipeline(RobustScaler(),
                             LassoCV(max_iter=1e7,
                                    alphas = l_alphas,
                                    random_state = 42)).fit(norm_X_train, train_target)
    scores = lasso_model2.steps[1][1].mse_path_ 
    plt.figure()  
    plt.xlabel('alpha')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.show()
    
#### 分析lasso k-fold 过拟合 特征
    alphas_mse = [0.00001,0.0001,0.006,0.001,0.003,0.008,0.01,0.05,0.1,0.2,0.3,0.4,0.5]
    lasso_model_mse = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas = alphas_mse, cv = kfolds
                                )).fit(norm_X_train, train_target)
    lasso_model_score = cv_rmse(lasso_model_mse, norm_X_train, train_target) 
    print("Lasso cv score: {0:.6f}".format(lasso_model_score.mean())) 
    lcv_scores = lasso_model_mse.steps[1][1].mse_path_
    plt.plot(alphas_mse, lcv_scores, label='Lasso')
    coeffs = pd.DataFrame(list(zip(norm_X_train.columns, lasso_model_mse.steps[1][1].coef_)), columns=['Features', 'Coefficients'])
    used_coeffs = coeffs[coeffs['Coefficients'] != 0].sort_values(by='Coefficients', ascending=False)
    print(used_coeffs.shape)
    print(used_coeffs)
    used_coeffs_values = norm_X_train[used_coeffs['Features']]
    used_coeffs_values.shape
    overfit_test2 = []
    for i in used_coeffs_values.columns:
        counts2 = used_coeffs_values[i].value_counts()
        zeros2 = counts2.iloc[0]
        if zeros2 / len(used_coeffs_values) * 100 > 40:
            overfit_test2.append(i)  
    print('Overfit Features')
    print(overfit_test2)
    
#### elastic_model  
    e_alphas_best = [0.0008,0.0001,0.0002,0.001,0.01,0.1,1,10,100]
    e_l1ratio_best = [0.001,0.01,0.1,1,5,10,100]  
    elastic = make_pipeline(RobustScaler(), 
                               ElasticNetCV(max_iter=1e5, alphas=e_alphas_best, 
                                            cv=kfolds, l1_ratio=e_l1ratio_best)) 
    elastic_model = elastic.fit(norm_X_train, train_target)
    elastic_model_score = cv_rmse(elastic_model, norm_X_train, train_target)
    print("elastic cv score: {0:.6f}".format(elastic_model_score.mean()))  
    print(elastic_model.steps[1][1].alpha_)
    print(elastic_model.steps[1][1].l1_ratio_)
    
#### xgb 选取最佳参数 
    xgb_reg = xgb.XGBRegressor()
    xgb_reg_param_grid = {'max_depth': [3,4,6], 'min_child_weight': [0.9,1], 'gamma': [0.0001],'colsample_bytree': [0.9,0.8],
                          'subsample': [0.7,0.55], 'scale_pos_weight': [1], 'learning_rate': [0.01], 'reg_alpha': [0.001],
                          'booster': ['gbtree'], 'n_estimators': [3000]}
    xgb_reg_param_grid = {'max_depth': [4], 'min_child_weight': [1], 'gamma': [0.0001],'colsample_bytree': [0.9],
                          'subsample': [0.55], 'scale_pos_weight': [1], 'learning_rate': [0.01], 'reg_alpha': [0.001],
                          'booster': ['gbtree'], 'n_estimators': [3000]}
    xgb_reg_grid = model_selection.GridSearchCV(xgb_reg, xgb_reg_param_grid, cv=10, verbose=1, n_jobs=-1,
                                                scoring='neg_mean_squared_error')
    xgb_reg_grid.fit(norm_X_train, train_target)  
    print('Best XGB Params:' + str(xgb_reg_grid.best_params_)) 
    print('Best XGB score:' + str(np.sqrt(-xgb_reg_grid.best_score_))) 
    feature_imp_sorted_xgb = pd.DataFrame({'feature': list(norm_X_train),
                                           'importance': xgb_reg_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_xgb = feature_imp_sorted_xgb.head(10)['feature'] 
    xgb_feature_importance = 100.0 * (feature_imp_sorted_xgb['importance'] / feature_imp_sorted_xgb['importance'].max())
    xgb_important_idx = np.where(xgb_feature_importance)[0]
    posxgb = np.arange(xgb_important_idx.shape[0]) + .5  
    plt.barh(posxgb, np.array(xgb_feature_importance[xgb_feature_importance != 0]))
    plt.yticks(posxgb, feature_imp_sorted_xgb['feature'])
    plt.xlabel('Relative Importance')
    plt.title('XGB Features Importance')
    plt.show()  
    
#### gbm 选取最佳参数 
    gbm_reg = GradientBoostingRegressor(random_state=1)
    gbm_reg_param_grid = {'n_estimators': [2000,3000], 'max_depth': [3,4], 'min_samples_split': [2,10,15], 'min_samples_leaf': [2,5],
                          'max_features': ['auto'], 'subsample': [0.6,0.7], 'learning_rate': [0.01]}
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'min_samples_split': [2], 'min_samples_leaf': [2],
                          'max_features': ['auto'], 'subsample': [0.6], 'learning_rate': [0.01]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, verbose=1, n_jobs=-1,
                                                scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(norm_X_train, train_target) 
    print('Best gbm Params:' + str(gbm_reg_grid.best_params_)) 
    print('Best gbm score:' + str(np.sqrt(-gbm_reg_grid.best_score_))) 
    feature_imp_sorted_gbm = pd.DataFrame({'feature': list(norm_X_train),
                                           'importance': gbm_reg_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gbm = feature_imp_sorted_gbm.head(10)['feature']  
    gbm_feature_importance = 100.0 * (feature_imp_sorted_gbm['importance'] / feature_imp_sorted_gbm['importance'].max())
    gbm_important_idx = np.where(gbm_feature_importance)[0] 
    posgbm = np.arange(gbm_important_idx.shape[0]) + .5   
    plt.figure()
    plt.barh(posgbm, np.array(gbm_feature_importance[gbm_feature_importance != 0]))
    plt.yticks(posgbm, feature_imp_sorted_gbm['feature'])
    plt.xlabel('Relative Importance')
    plt.title('GradientBoosting Feature Importance')
    
#### lgm 选取最佳参数 
#    lgm_reg = LGBMRegressor()
#    lgm_reg_param_grid = {'learning_rate':[0.01], 'n_estimators':[2000],
#                          'max_depth':[3], 'num_leaves':[4],
#                          'max_bin':[55],  
#                          'feature_fraction': [0.8],'bagging_fraction':[0.9],
#                          'min_data_in_leaf':[6],'min_sum_hessian_in_leaf':[0.7]}
#    lgm_reg_grid = model_selection.GridSearchCV(lgm_reg, lgm_reg_param_grid, cv=10, verbose=1, n_jobs=-1,
#                                                scoring='neg_mean_squared_error')
#    lgm_reg_grid.fit(norm_X_train, train_target) 
#    print('Best lgm Params:' + str(lgm_reg_grid.best_params_))
#    print('Best lgm score:' + str(np.sqrt(-lgm_reg_grid.best_score_)))
#    feature_imp_sorted_lgm = pd.DataFrame({'feature': list(norm_X_train),
#                                           'importance': lgm_reg_grid.best_estimator_.feature_importances_}).sort_values(
#        'importance', ascending=False)
#    features_top_n_lgm = feature_imp_sorted_lgm.head(10)['feature']  
#    lgm_feature_importance = 100.0 * (feature_imp_sorted_lgm['importance'] / feature_imp_sorted_lgm['importance'].max())
#    lgm_important_idx = np.where(lgm_feature_importance)[0] 
#    poslgm = np.arange(lgm_important_idx.shape[0]) + .5   
#    plt.barh(poslgm, np.array(lgm_feature_importance[lgm_feature_importance != 0]))
#    plt.yticks(poslgm, feature_imp_sorted_lgm['feature'])
#    plt.xlabel('Relative Importance')
#    plt.title('Lgbm Feature Importance')
    
#### svr model
    svr_reg = svm.SVR()
    svr_reg_param_grid = {'C':[0.1,1,10,20,40,60,70,100,200,250,300,350,400,500,1000,2000,3000,4000], 
                          'gamma':[0.00001,0.0001,0.0003,0.0005,0.001,0.005,0.01,0.1,1,10,100,1000]}
    svr_reg_grid = model_selection.GridSearchCV(svr_reg, svr_reg_param_grid, cv=10, verbose=1, n_jobs=-1,
                                                scoring='neg_mean_squared_error')
    svr_reg_grid.fit(norm_X_train, train_target) 
    print('Best svr Params:' + str(svr_reg_grid.best_params_))
    print('Best svr score:' + str(np.sqrt(-svr_reg_grid.best_score_)))
    
    
    
