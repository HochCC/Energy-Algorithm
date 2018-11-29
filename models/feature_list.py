def select_feature1(data, trainornot): 
#    select_list = ['charge_hour', 'dsoc', 
#                   'charge_start_soc', 'charge_end_soc', 'charge_start_U', 'charge_end_U', 'dU', 
#                   'mileage', #'ddsoc',
#                   'dtemp', 'dsoc/hour', 'charge_start_I', 'charge_end_I',
#                   'sum_charge', 'charge_max_temp', 'charge_min_temp',
#                   'year', 'month', 'day', 'hour', 'week', 'charge_mode'
#                   ]
    select_list = [  # car1 m
                   'charge_hour', 
#                   'dsoc', 
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
#                   'charge_min_temp',
#                   'charge_end_U',
#                   'charge_start_U', 
#                   'dU',  
#                    'charge_start_I', 
#                    'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    select_list = [  # car1 k
                   'charge_hour', 
                   'dsoc', 
                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
                   'charge_min_temp',
#                   'charge_end_U',
                   'charge_start_U', 
                   'dU',  
                    'charge_start_I', 
#                    'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature2(data, trainornot): 
    select_list = [  # car2 m
                   'charge_hour', 
                   'dsoc', 
                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
                   'charge_min_temp',
#                   'charge_end_U',
                   'charge_start_U', 
                   'dU',  
                    'charge_start_I', 
#                    'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    select_list = [  # car2 k
                   'charge_hour', 
                   'dsoc', 
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
#                   'charge_min_temp',
##                   'charge_end_U',
#                   'charge_start_U', 
#                   'dU',  
#                   'charge_start_I', 
#                    'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature3(data, trainornot):  
    select_list = [  # car3 m
                   'charge_hour', 
#                   'dsoc', 
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
#                   'charge_min_temp',
##                   'charge_end_U',
#                   'charge_start_U', 
#                   'dU',  
#                   'charge_start_I', 
#                    'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ]
    select_list = [  # car3 k
                   'charge_hour', 
                   'dsoc', 
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
                   'charge_min_temp',
#                   'charge_end_U',
#                   'charge_start_U', 
#                   'dU',  
                    'charge_start_I', 
                    'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ]  
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature4(data, trainornot): 
    # p1 m1
    select_list = ['charge_hour', 
                   'dsoc', #'charge_mode',
#                   'charge_start_soc', 'charge_end_soc', 
                   'dsoc/hour',
                   'dU/dsoc',
                   'dtemp', 
#                   'charge_max_temp', 
                   'charge_min_temp',
                   #'charge_end_U',#'charge_start_U', ## 'dU', 
                    #'charge_start_I', 'charge_end_I',
                    #'dmileage',  #'ddsoc',
                   #'sum_charge', 'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ]
    # p1 k
    select_list = ['charge_hour', 
                   'dsoc', #'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
                   'charge_min_temp',
#                   'charge_end_U',
                   #'charge_start_U', ## 'dU', 
                    #
                    'charge_start_I', 
                    'charge_end_I',
                    #'dmileage',  #'ddsoc',
                   #'sum_charge', 'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ]
    select_list = [  
                   'charge_hour', 
                   'dsoc', 
                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
                   'charge_min_temp',
#                   'charge_end_U',
#                   'charge_start_U', 
#                   'dU',  
                    'charge_start_I', 
                    'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature5(data, trainornot): 
#   car5 不区分快慢充， 用小能量调参
    select_list = [ # small energy 合并快慢充 drop energy<1, dsoc<1, charge_energy>20  小数据drop后调参，用全部数据去训练
            # 调好后一起训练,不删除大数据
                   'charge_hour', 
                   'dsoc',  
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
                   'dtemp', 
#                   'charge_max_temp', 
#                   'charge_min_temp',
#                   'charge_end_U',
#                   'charge_start_U', 
                   'dU',  
                   'charge_start_I', 
                   'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ]
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature6(data, trainornot):   
    select_list = [ # 合并快慢充  
#                   'charge_hour', 
                   'dsoc', #
#                   'charge_mode',
                   'charge_start_soc', 
                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
                   'charge_min_temp',
#                   'charge_end_U',
                   'charge_start_U', 
#                   'dU',  
                    'charge_start_I', 
#                    'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature7(data, trainornot): 
    select_list = [  #p2
                   'charge_hour', 
                   'dsoc', #
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
#                   'charge_min_temp',
                   'charge_end_U',
                   'charge_start_U', 
#                   'dU',  
                   'charge_start_I', 
                   'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature8(data, trainornot): 
    select_list = [  #p2
#                   'charge_hour', 
                   'dsoc', #
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
                   'charge_min_temp',
                   'charge_end_U',
                   'charge_start_U', 
                   'dU',  
                   'charge_start_I', 
                   'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature9(data, trainornot):  
    select_list = [  # 9车小电流， drop car['charge_energy'] > 10, 不区分快慢充
                   'charge_hour', 
                   'dsoc', #
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
#                   'charge_min_temp',
#                   'charge_end_U',
#                   'charge_start_U', 
#                   'dU',  
                   'charge_start_I', 
                   'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    select_list = [   #car 9 k
                   'charge_hour', 
                   'dsoc', #
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
                   'charge_min_temp',
                   'charge_end_U',
                   'charge_start_U', 
                   'dU',  
                   'charge_start_I', 
                   'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    select_list = [   #car 9 m
                   'charge_hour', 
#                   'dsoc', #
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
#                   'charge_min_temp',
#                   'charge_end_U',
#                   'charge_start_U', 
#                   'dU',  
#                   'charge_start_I', 
#                   'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ]
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature10(data, trainornot): 
    select_list = [  
                   'charge_hour', 
                   'dsoc', #
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
                   'charge_min_temp',
                   'charge_end_U',
                   'charge_start_U', 
                   'dU',  
                   'charge_start_I', 
                   'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature11(data, trainornot): 
    select_list = [  
                   'charge_hour', 
                   'dsoc', #
                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
                   'charge_min_temp',
                   'charge_end_U',
                   'charge_start_U', 
                   'dU',  
                   'charge_start_I', 
                   'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
#    select_list = [   #p2
#                   'charge_hour', 
#                   'dsoc', #
##                   'charge_mode',
##                   'charge_start_soc', 
##                   'charge_end_soc', 
##                   'dsoc/hour',
##                   'dU/dsoc',
##                   'dtemp', 
##                   'charge_max_temp', 
#                   'charge_min_temp',
#                   'charge_end_U',
#                   'charge_start_U', 
#                   'dU',  
#                   'charge_start_I', 
#                   'charge_end_I',
##                    'dmileage',  #'ddsoc',
##                   'sum_charge', 
##                   'mileage', 
#                  # 'year', 'month', 'week', 'day', 'hour'
#                   ]
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature12(data, trainornot):  
    select_list = [  #p1
                   'charge_hour', 
                   'dsoc', 
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
                   'dtemp', 
#                   'charge_max_temp', 
#                   'charge_min_temp',
#                   'charge_end_U',
                   'charge_start_U', 
#                   'dU',  
                   'charge_start_I', 
#                   'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature13(data, trainornot): 
    select_list = ['charge_hour', 'dsoc', 'charge_mode',
#                   'charge_start_soc', 'charge_end_soc', 
#                   'dsoc/hour','dU/dsoc',
                   'dtemp', 'charge_max_temp', 'charge_min_temp',
                   #'charge_end_U',#'charge_start_U', ## 'dU', 
                    #'charge_start_I', 'charge_end_I',
                    #'dmileage',  #'ddsoc',
                   #'sum_charge', 'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ]
    select_list = ['charge_hour', 'dsoc', 'charge_mode',
                   'charge_start_soc', 'charge_end_soc', 
                   'dsoc/hour',
                    'dU/dsoc', 'mileage', 
#                   'dtemp', 'charge_max_temp', 'charge_min_temp',
#                   'charge_end_U','charge_start_U', 'dU', 
#                    'charge_start_I', 'charge_end_I',
#                    'dmileage',  'ddsoc',
#                   'sum_charge', 'mileage', 
#                  'year', 'month', 'week', 'day', 'hour'
                   ]
    select_list = [  #p1
#                   'charge_hour', 
                   'dsoc', 
#                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
                   'dsoc/hour',
#                   'dU/dsoc',
                   'dtemp', 
#                   'charge_max_temp', 
#                   'charge_min_temp',
                   'charge_end_U',
                   'charge_start_U', 
                   'dU',  
                   'charge_start_I', 
                   'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list  


def select_feature14(data, trainornot):  
    select_list = [  
                   'charge_hour', 
                   'dsoc', #
                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
                   'charge_min_temp',
                   'charge_end_U',
#                   'charge_start_U', 
                   'dU',  
#                   'charge_start_I', 
                   'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    select_list = [  
                   'charge_hour', 
                   'dsoc', #
                   'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
#                   'charge_max_temp', 
#                   'charge_min_temp',
                   'charge_end_U',
#                   'charge_start_U', 
                   'dU',  
#                   'charge_start_I', 
                   'charge_end_I',
#                    'dmileage',  #'ddsoc',
#                   'sum_charge', 
#                   'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ] 
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 
 

def select_feature15(data, trainornot): 
    select_list = [
                   'charge_hour', 
                   'dsoc', #'charge_mode',
                   'charge_start_soc', 
                   'charge_end_soc', 
#                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
                   'charge_max_temp', 
#                   'charge_min_temp',
#                   'charge_end_U',
                   'charge_start_U', 
                   'dU',  
                    'charge_start_I', 
#                    'charge_end_I',
#                    'dmileage',  #'ddsoc',
                   #'sum_charge', 'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ]
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 


def select_feature16(data, trainornot): 
    select_list = [
                   'charge_hour', 
                   'dsoc', #'charge_mode',
#                   'charge_start_soc', 
#                   'charge_end_soc', 
                   'dsoc/hour',
#                   'dU/dsoc',
#                   'dtemp', 
                   'charge_max_temp', 
#                   'charge_min_temp',
#                   'charge_end_U',
                   'charge_start_U', 
                   'dU',  
                   'charge_start_I', 
#                    'charge_end_I',
#                    'dmileage',  #'ddsoc',
                   #'sum_charge', 'mileage', 
                  # 'year', 'month', 'week', 'day', 'hour'
                   ]
    if trainornot == True:
        select_list.append('charge_energy')
    selected_features = data[select_list]
    processed_features = selected_features.copy()
    return processed_features, select_list 