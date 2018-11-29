import pandas as pd
import numpy as np
import datetime  
from sklearn.model_selection import learning_curve 
import matplotlib.pyplot as plt


def get_interval(series, col1, col2):
    """获得秒数差，DataFrame level"""
    def chargetime(start, end):
        """计算充电时长"""
        h1 = datetime.datetime.strptime(start, '%Y%m%d%H%M%S')
        h2 = datetime.datetime.strptime(end, '%Y%m%d%H%M%S')
        delta = h2 - h1
        return delta.total_seconds()
    interval = chargetime(series[col1].strip(), series[col2].strip())  # strip去两端空白
    return interval


def formattime(series, col):
    def formatt(cl):
        return datetime.datetime.strptime(cl, '%Y%m%d%H%M%S')
    return formatt(series[col].strip())


def yearfeature(series, col):
    def formatt(cl):
        time = datetime.datetime.strptime(cl, '%Y%m%d%H%M%S')
        return time.year
    return formatt(series[col].strip())


def monfeature(series, col):
    def formatt(cl):
        time = datetime.datetime.strptime(cl, '%Y%m%d%H%M%S')
        return time.month
    return formatt(series[col].strip())


def weekfeature(series, col):
    def formatt(cl):
        time = datetime.datetime.strptime(cl, '%Y%m%d%H%M%S') 
        return time.weekday()
    return formatt(series[col].strip())


def dayfeature(series, col):
    def formatt(cl):
        time = datetime.datetime.strptime(cl, '%Y%m%d%H%M%S')
        return time.day
    return formatt(series[col].strip())


def hourfeature(series, col):
    def formatt(cl):
        time = datetime.datetime.strptime(cl, '%Y%m%d%H%M%S')
        return time.hour
    return formatt(series[col].strip())


def getmin(series, col):
    """"获得充电间隔分钟数"""
    def formatt(cl):
        return cl.total_seconds() / 60
    return formatt(series[col])


#def totolday(series, col):
#    def formatt(cl):
#        time = datetime.datetime.strptime(cl, '%Y%m%d%H%M%S') 
#        t=datetime.datetime(2017,11,1,hour=0,minute=0,second=0) 
#        return (time - t).days
#    return formatt(series[col].strip()) 
 

def preprocess_features(data, train_ornot):
    """预处理原始数据""" 
    process_feature = data.copy()
    process_feature['year'] = process_feature.apply(yearfeature, axis=1, args=('charge_start_time',))
    process_feature['month'] = process_feature.apply(monfeature, axis=1, args=('charge_start_time',))
    process_feature['week'] = process_feature.apply(weekfeature, axis=1, args=('charge_start_time',))
    process_feature['day'] = process_feature.apply(dayfeature, axis=1, args=('charge_start_time',))
    process_feature['hour'] = process_feature.apply(hourfeature, axis=1, args=('charge_start_time',))
    process_feature['night'] = 0
#    process_feature['totolday'] = process_feature.apply(totolday, axis=1, args=('charge_start_time',))
#    process_feature.loc[(process_feature['hour'] > 0) & (process_feature['hour'] < 8), ['night']] = 1
    # 格式化充电时间， axis=1 表示对整行，这里是单格应用apply， axis=0 表示对整列应用apply
    process_feature['start'] = process_feature.apply(formattime, axis=1, args=('charge_start_time',))
    process_feature['end'] = process_feature.apply(formattime, axis=1, args=('charge_end_time',))
    # 计算充电起始，结束时间秒数差， axis=1 表示横向处理DataFrame
    process_feature['second_interval'] = process_feature.apply(get_interval, axis=1,
                                                               args=('charge_start_time', 'charge_end_time',))
    process_feature['charge_hour'] = process_feature['second_interval'] / 3600 
    # 计算dU, 电压差值
    process_feature['dU'] = process_feature['charge_end_U'] - process_feature['charge_start_U']
    # drop axis=1 表示整列删除，axis=0默认行删除, inplace=True修改原数组
    process_feature.drop(['charge_start_time'], axis=1, inplace=True)
    process_feature.drop(['charge_end_time'], axis=1, inplace=True)
    # dsoc,
    process_feature['dsoc'] = process_feature['charge_end_soc'] - process_feature['charge_start_soc'] 
    process_feature['ddsoc'] = process_feature['charge_end_soc'].shift(1) - process_feature['charge_start_soc']
    process_feature['dsoc/hour'] = process_feature['dsoc'] / process_feature['charge_hour'] 
    process_feature['dtemp'] = process_feature['charge_max_temp'] - process_feature['charge_min_temp']
    # process_feature['dmileage'] = process_feature['mileage'].apply(lambda x: x.diff(1))
    process_feature['dmileage'] = (process_feature.groupby('vehicle_id')['mileage'].diff(1))
    process_feature['dmileage/soc'] = (process_feature['dmileage'] / (process_feature['charge_end_soc'].shift(1) -
                   process_feature['charge_start_soc']))
    # 计算充电间隔，carbycar
    process_feature['interval_min'] = process_feature['start'] - process_feature['end'].shift(1)
    process_feature['interval_min'] = process_feature.apply(getmin, axis=1, args=('interval_min',))
    process_feature['dU/dsoc'] = process_feature['dU'] / process_feature['dsoc']
    if train_ornot == True:
        process_feature['energy/dsoc'] = process_feature['charge_energy'] / process_feature['dsoc']
        process_feature['P(kW)'] = process_feature['charge_energy'] / process_feature['charge_hour']
    return process_feature 

 
def car1_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    
    if train_ornot == True:  
        # 新增列phase 标记不同的充电电池
        car['phase'] = 1 
        car['phase'][:221] = 0 
        # 分阶段
        car_phase0 = car[car['phase'].isin([0])].copy()
#        car_phase1 = car[car['phase'].isin([1])].copy()
        # 补全 慢充 charge_end_soc
        car_phase0.loc[(car_phase0.charge_end_U > 395) & (car_phase0.charge_end_I > -8) & (car_phase0.charge_start_I > -8), ['charge_end_soc']] = 100 
         
        # 补全soc后,对涉及soc的重新计算
        car_phase0['dsoc'] = car_phase0['charge_end_soc'] - car_phase0['charge_start_soc']
        car_phase0['dU/dsoc'] = car_phase0['dU'] / car_phase0['dsoc']
        car_phase0['dsoc/hour'] = car_phase0['dsoc'] / car_phase0['charge_hour']  
        car_phase0['energy/dsoc'] = car_phase0['charge_energy'] / car_phase0['dsoc']
        car_phase0['P/kW'] = car_phase0['charge_energy'] / car_phase0['charge_hour']
        
        # phase0 区分快慢充电
        car_phase0['charge_mode'] = 1 
        car_phase0.loc[((car_phase0.charge_start_I > -8) & (car_phase0.charge_end_I > -8)), ['charge_mode']] = 0 
        # 删除异常数据, 异常数据不要随便踢，有可能信息量丰富
#        car_phase0.drop(car_phase0.loc[car_phase0['dsoc'] < 1].index.tolist(), inplace=True) 
        car_phase0.drop(car_phase0.loc[car_phase0['dsoc'] < 2].index.tolist(), inplace=True) 
#        car_phase0.drop(car_phase0.loc[(car_phase0['charge_mode'] == 0) & (car_phase0['charge_hour'] < 3.7)].index.tolist(), inplace=True) 
        
    if train_ornot == False:
        car['phase'] = 0 
        car['charge_mode'] = 1  
        car_phase0 = car[car['phase'].isin([0])].copy()
        car_phase0.loc[((car_phase0.charge_start_I > -8) & (car_phase0.charge_end_I > -8)), ['charge_mode']] = 0  
#        car_phase1 = car[car['phase'].isin([1])].copy()  
    return car_phase0 


def car2_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    
    if train_ornot == True:  
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan   
        # 区分快慢充电
        car['charge_mode'] = 1 
        car.loc[((car.charge_start_I > -8) & (car.charge_end_I > -8)), ['charge_mode']] = 0 
        # 删除异常数据
#        car.drop(car.loc[car['charge_energy'] < 2].index.tolist(), inplace=True) # high P
#        car.drop(car.loc[(car['charge_mode'] == 0) & (car['charge_end_soc'] != 100)].index.tolist(), inplace=True) # low P
        
    if train_ornot == False: 
        car['charge_mode'] = 1   
        car.loc[((car.charge_start_I > -8) & (car.charge_end_I > -8)), ['charge_mode']] = 0   
    return car 
 
    
def car3_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    
    if train_ornot == True:  
        # 新增列phase 标记不同的充电电池
        car['phase'] = 1 
        car['phase'][:211] = 0 
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan  
        # 分阶段
        car_phase0 = car[car['phase'].isin([0])].copy()
#        car_phase1 = car[car['phase'].isin([1])].copy()
        # 补全 慢充 charge_end_soc
        car_phase0.loc[(car_phase0.charge_end_U > 389) & (car_phase0.charge_end_I > -29), ['charge_end_soc']] = 100 
         
        # 补全soc后,对涉及soc的重新计算
        car_phase0['dsoc'] = car_phase0['charge_end_soc'] - car_phase0['charge_start_soc']
        car_phase0['dU/dsoc'] = car_phase0['dU'] / car_phase0['dsoc']
        car_phase0['dsoc/hour'] = car_phase0['dsoc'] / car_phase0['charge_hour']  
        car_phase0['energy/dsoc'] = car_phase0['charge_energy'] / car_phase0['dsoc']
        car_phase0['P/kW'] = car_phase0['charge_energy'] / car_phase0['charge_hour']
        
        # phase0 区分快慢充电
        car_phase0['charge_mode'] = 1 
        car_phase0.loc[(car_phase0['dsoc/hour'] < 10), ['charge_mode']] = 0 
        # 删除异常数据, 异常数据不要随便踢，有可能信息量丰富
        car_phase0.drop(car_phase0.loc[car_phase0['dsoc'] < 1].index.tolist(), inplace=True)
        car_phase0.drop(car_phase0.loc[(car_phase0['charge_mode'] == 1) & (car_phase0['charge_hour'] > 4)].index.tolist(), inplace=True) 
        
    if train_ornot == False:
        car['phase'] = 0 
        car['charge_mode'] = 1  
        car_phase0 = car[car['phase'].isin([0])].copy()
        car_phase0.loc[(car_phase0['dsoc/hour'] < 10), ['charge_mode']] = 0 
#        car_phase1 = car[car['phase'].isin([1])].copy()  
    return car_phase0 


def car4_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    
    if train_ornot == True:  
        # 新增列phase 标记不同的充电电池
        car['phase'] = 1 
        car['phase'][:210] = 0 
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan  
        # 分阶段
        car_phase0 = car[car['phase'].isin([0])].copy()
#        car_phase1 = car[car['phase'].isin([1])].copy()
        # 补全 慢充 charge_end_soc
        car_phase0.loc[(car_phase0.charge_start_I > -8) & (car_phase0.charge_end_I > -8) & (car_phase0.charge_end_U > 382), ['charge_end_soc']] = 100 
         
        # 补全soc后,对涉及soc的重新计算
        car_phase0['dsoc'] = car_phase0['charge_end_soc'] - car_phase0['charge_start_soc']
        car_phase0['dU/dsoc'] = car_phase0['dU'] / car_phase0['dsoc']
        car_phase0['dsoc/hour'] = car_phase0['dsoc'] / car_phase0['charge_hour']  
        car_phase0['energy/dsoc'] = car_phase0['charge_energy'] / car_phase0['dsoc']
        car_phase0['P/kW'] = car_phase0['charge_energy'] / car_phase0['charge_hour']
        
        # phase0 区分快慢充电
        car_phase0['charge_mode'] = 2
        car_phase0.loc[((car_phase0.charge_start_I > -8) & (car_phase0.charge_end_I > -8) & (car_phase0['dsoc/hour'] < 8.78)), ['charge_mode']] = 0 
        car_phase0.loc[((car_phase0.charge_start_I > -8) & (car_phase0.charge_end_I > -8) & (car_phase0['dsoc/hour'] >= 8.78)), ['charge_mode']] = 1
        
        # 删除异常数据, 异常数据不要随便踢，有可能信息量丰富
        car_phase0.drop(car_phase0.loc[car_phase0['dsoc'] <= 1].index.tolist(), inplace=True)
        car_phase0.drop(car_phase0.loc[car_phase0['energy/dsoc'] < 0.25].index.tolist(), inplace=True)
        car_phase0.drop(car_phase0.loc[car_phase0['energy/dsoc'] > 0.35].index.tolist(), inplace=True) 
        car_phase0.drop(car_phase0.loc[(car_phase0['charge_mode'] == 0) & (car_phase0['charge_hour'] > 10)].index.tolist(), inplace=True) 
        
    if train_ornot == False:
        car['phase'] = 0 
        car['charge_mode'] = 2
        car_phase0 = car[car['phase'].isin([0])].copy()
        car_phase0.loc[((car_phase0.charge_start_I > -8) & (car_phase0.charge_end_I > -8) & (car_phase0['dsoc/hour'] < 8.78)), ['charge_mode']] = 0 
        car_phase0.loc[((car_phase0.charge_start_I > -8) & (car_phase0.charge_end_I > -8) & (car_phase0['dsoc/hour'] >= 8.78)), ['charge_mode']] = 1
    return car_phase0 


def car5_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    
    if train_ornot == True:  
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan   
        # 区分快慢充电
        car['charge_mode'] = 1 
        car.loc[car['dsoc/hour'] < 15, ['charge_mode']] = 0  
#        car.drop(car.loc[car['charge_energy'] > 20].index.tolist(), inplace=True) 
        # 删除异常数据
        car.drop(car.loc[car['dsoc'] <= 1].index.tolist(), inplace=True) 
#        car.drop(car.loc[car['charge_energy'] <= 1].index.tolist(), inplace=True) 
#        car.drop(car.loc[car['charge_energy'] > 15].index.tolist(), inplace=True) 
#        car.drop(car.loc[(car['charge_mode'] == 0) & (car['charge_end_soc'] != 100)].index.tolist(), inplace=True) 
        
    if train_ornot == False: 
        car['charge_mode'] = 1   
        car.loc[car['dsoc/hour'] < 15, ['charge_mode']] = 0  
    return car 


def car6_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    
    if train_ornot == True:  
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan   
        # 区分快慢充电
        car['charge_mode'] = 1 
        car.loc[car['dsoc/hour'] < 15, ['charge_mode']] = 0 
        # 删除异常数据
        car.drop(car.loc[car['dsoc'] <= 1].index.tolist(), inplace=True) 
        car.drop(car.loc[car['charge_energy'] > 300].index.tolist(), inplace=True)  
    if train_ornot == False: 
        car['charge_mode'] = 1   
        car.loc[car['dsoc/hour'] < 15, ['charge_mode']] = 0  
    return car 


def car7_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    
    if train_ornot == True:  
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan   
        # 区分快慢充电
        car['charge_mode'] = 2
        car.loc[car['dsoc/hour'] <= 9, ['charge_mode']] = 0 
        car.loc[(car['dsoc/hour'] > 9) & (car['dsoc/hour'] <= 23), ['charge_mode']] = 1
        # 删除异常数据
        car.drop(car.loc[car['dsoc'] <= 1].index.tolist(), inplace=True)  
    if train_ornot == False: 
        car['charge_mode'] = 2   
        car.loc[car['dsoc/hour'] <= 9, ['charge_mode']] = 0 
        car.loc[(car['dsoc/hour'] > 9) & (car['dsoc/hour'] <= 23), ['charge_mode']] = 1
    return car 


def car8_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    
    if train_ornot == True:  
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan   
        # 区分快慢充电
        car['charge_mode'] = 1
        car.loc[car['dsoc/hour'] < 9, ['charge_mode']] = 0 
        # 删除异常数据
        car.drop(car.loc[car['dsoc'] <= 1].index.tolist(), inplace=True)  
        
    if train_ornot == False: 
        car['charge_mode'] = 1
        car.loc[car['dsoc/hour'] < 9, ['charge_mode']] = 0 
    return car


def car9_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    
    if train_ornot == True:  
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan   
        # 区分快慢充电
        car['charge_mode'] = 1
        car.loc[car['dsoc/hour'] < 6, ['charge_mode']] = 0 
        # 删除异常数据
        car.drop(car.loc[car['dsoc'] <= 1].index.tolist(), inplace=True) 
#        car.drop(car.loc[car['charge_energy'] > 10].index.tolist(), inplace=True) 
#        car.drop(car.loc[car['dsoc'] < 1].index.tolist(), inplace=True) 
#        car.drop(car.loc[car['dsoc'] > 10].index.tolist(), inplace=True) 
#        car.drop(car.loc[car['charge_energy'] > 300].index.tolist(), inplace=True) 
        
    if train_ornot == False: 
        car['charge_mode'] = 1
        car.loc[car['dsoc/hour'] < 6, ['charge_mode']] = 0 
    return car


def car10_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    
    if train_ornot == True:  
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan   
        # 不区分快慢充电
        car['charge_mode'] = 3
        # 删除异常数据
        car.drop(car.loc[car['dsoc'] <= 1].index.tolist(), inplace=True)  
        
    if train_ornot == False: 
        car['charge_mode'] = 3
    return car


def car11_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    if train_ornot == True:  
        # 新增列phase 标记不同的充电电池
        car['phase'] = 2
        car['phase'][:215] = 0 
        car['phase'][219:] = 1
        # 不区分快慢充电
        car['charge_mode'] = 3 
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan  
        # 删除异常数据, 异常数据不要随便踢，有可能信息量丰富
        car.drop(car.loc[car['dsoc'] <= 1].index.tolist(), inplace=True)
        # 分阶段
        car_phase0 = car[car['phase'].isin([0])].copy()
        car_phase0['sum_charge'] = car_phase0['charge_hour'].cumsum() 
        car_phase1 = car[car['phase'].isin([1])].copy()
        car_phase1['sum_charge'] = car_phase1['charge_hour'].cumsum()   
        recar = pd.concat([car_phase0, car_phase1], axis=0) 
    if train_ornot == False:
        car['phase'] = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]
        car['charge_mode'] = 3 
        recar = car.copy()
    return recar 


def car12_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    if train_ornot == True:
        # 新增列phase 标记不同的充电电池
        car['phase'] = 2
        car['phase'][:292] = 0 
        car['phase'][298:] = 1
        # 区分快慢充电
        car['charge_mode'] = 1 
        car.loc[((car['dsoc/hour'] > 3) & (car['dsoc/hour'] < 5)), ['charge_mode']] = 0
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan  
        car['energy/dsoc'] = car['charge_energy'] / car['dsoc']
        # 删除异常数据, 异常数据不要随便踢，有可能信息量丰富
        car.drop(car.loc[car['dsoc'] < 1].index.tolist(), inplace=True) 
        # 分阶段
        car_phase0 = car[car['phase'].isin([0])].copy()
        car_phase0['sum_charge'] = car_phase0['charge_hour'].cumsum() 
        car_phase1 = car[car['phase'].isin([1])].copy()
        car_phase1['sum_charge'] = car_phase1['charge_hour'].cumsum()
        # 分别对phase 删除 energy/dsoc 的离群点 
        car_phase0.drop(car_phase0.loc[car_phase0['energy/dsoc'] < 0.6].index.tolist(), inplace=True)
        car_phase0.drop(car_phase0.loc[car_phase0['energy/dsoc'] > 1].index.tolist(), inplace=True)
        car_phase1.drop(car_phase1.loc[car_phase1['energy/dsoc'] < 0.2].index.tolist(), inplace=True)
        car_phase1.drop(car_phase1.loc[car_phase1['energy/dsoc'] > 0.4].index.tolist(), inplace=True) 
        recar = pd.concat([car_phase0, car_phase1], axis=0) 
    if train_ornot == False:
        car['phase'] = [0,0,0,0,0,0,0,0,0,1,1,1,1]
        # 区分快慢充电
        car['charge_mode'] = 1
        car.loc[((car['dsoc/hour'] > 3) & (car['dsoc/hour'] < 5)), ['charge_mode']] = 0
        recar = car.copy()
    return recar


def car13_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    if train_ornot == True:
        # 新增列phase 标记不同的充电电池
        car['phase'] = 2
        car['phase'][:171] = 0 
        car['phase'][174:] = 1 
        car['charge_mode'] = 3
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan  
        car['energy/dsoc'] = car['charge_energy'] / car['dsoc']
        # 删除异常数据, 异常数据不要随便踢，有可能信息量丰富
        car.drop(car.loc[car['dsoc'] <= 1].index.tolist(), inplace=True)
        # 分阶段
        car_phase0 = car[car['phase'].isin([0])].copy()
        car_phase0['sum_charge'] = car_phase0['charge_hour'].cumsum() 
        car_phase1 = car[car['phase'].isin([1])].copy()
        car_phase1['sum_charge'] = car_phase1['charge_hour'].cumsum() 
        # 分别对phase 删除 energy/dsoc 的离群点 
        car_phase0.drop(car_phase0.loc[car_phase0['energy/dsoc'] < 0.6].index.tolist(), inplace=True)
        car_phase0.drop(car_phase0.loc[car_phase0['energy/dsoc'] > 1].index.tolist(), inplace=True)
        car_phase1.drop(car_phase1.loc[car_phase1['energy/dsoc'] < 0.3].index.tolist(), inplace=True)
        car_phase1.drop(car_phase1.loc[car_phase1['energy/dsoc'] > 0.35].index.tolist(), inplace=True) 
#        car_phase0.drop(car_phase0.loc[(car_phase0['charge_mode'] == 1) & (car_phase0['charge_hour'] > 10)].index.tolist(), inplace=True) 
        recar = pd.concat([car_phase0, car_phase1], axis=0)
    if train_ornot == False:
        car['phase'] = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]
        car['charge_mode'] = 3
        recar = car.copy()
    return recar


def car14_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    if train_ornot == True:
        # 新增列phase 标记不同的充电电池
        car['phase'] = 2
        car['phase'][:223] = 0 
        car['phase'][229:] = 1
        # 区分快慢充电
        car['charge_mode'] = 1
        car.loc[((car.charge_start_I > -10) & (car.charge_end_I > -10)), ['charge_mode']] = 0
#        car.loc[((car.charge_mode == 1) & (car.phase == 1) & (car.dmileage != 0)), ['charge_mode']] = 2
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan  
        car['energy/dsoc'] = car['charge_energy'] / car['dsoc']
        # 删除异常数据, 异常数据不要随便踢，有可能信息量丰富
        car.drop(car.loc[car['dsoc'] <= 3].index.tolist(), inplace=True)  
        # 分阶段
        car_phase0 = car[car['phase'].isin([0])].copy()
        car_phase0['sum_charge'] = car_phase0['charge_hour'].cumsum() 
        car_phase1 = car[car['phase'].isin([1])].copy()
        car_phase1['sum_charge'] = car_phase1['charge_hour'].cumsum() 
        # 分别对phase 删除 energy/dsoc 的离群点 
        car_phase0.drop(car_phase0.loc[car_phase0['energy/dsoc'] < 0.4].index.tolist(), inplace=True)
        car_phase0.drop(car_phase0.loc[car_phase0['energy/dsoc'] > 1].index.tolist(), inplace=True)
        car_phase1.drop(car_phase1.loc[car_phase1['energy/dsoc'] < 0.2].index.tolist(), inplace=True)
        car_phase1.drop(car_phase1.loc[car_phase1['energy/dsoc'] > 0.4].index.tolist(), inplace=True) 
        recar = pd.concat([car_phase0, car_phase1], axis=0) 
    if train_ornot == False:
        car['phase'] = [0,0,0,0,0,0,0,0,0,0,1,1,1]
        # 区分快慢充电
        car['charge_mode'] = 1
        car.loc[((car.charge_start_I > -10) & (car.charge_end_I > -10)), ['charge_mode']] = 0
#        car.loc[((car.charge_mode == 1) & (car.phase == 1) & (car.dmileage != 0)), ['charge_mode']] = 2
        recar = car.copy()
    return recar


def car15_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    
    if train_ornot == True:  
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan   
        # 不区分快慢充电
        car['charge_mode'] = 3 
        # 删除异常数据
        car.drop(car.loc[car['dsoc'] <= 1].index.tolist(), inplace=True) 
        car.drop(car.loc[car['charge_hour'] > 5].index.tolist(), inplace=True) 
        
    if train_ornot == False: 
        car['charge_mode'] = 3
    return car


def car16_feature(ca, train_ornot):    
    car = ca.copy()
    car['sum_charge'] = car['charge_hour'].cumsum() 
    
    if train_ornot == True:  
        # 第一个interval_min为上车数据
        car.loc[0, 'interval_min'] = np.nan   
        # 不区分快慢充电
        car['charge_mode'] = 3 
        # 删除异常数据
        car.drop(car.loc[car['dsoc'] <= 1].index.tolist(), inplace=True) 
        car.drop(car.loc[car['charge_hour'] > 5].index.tolist(), inplace=True) 
        car.drop(car.loc[car['dsoc/hour'] > 100].index.tolist(), inplace=True) 
    if train_ornot == False: 
        car['charge_mode'] = 3
    return car


if __name__ == '__main__':
    # 读取原始数据
    readFile = '../dataset/energy_train1029.csv' 
    readTestB = '../dataset/energy_test1029.csv'
    initialData = pd.read_csv(readFile, dtype={'charge_start_time': str, 'charge_end_time': str})
    # 删除多余最后一列
    initialData.drop(['dy'], inplace=True, axis=1) 
    testDataB = pd.read_csv(readTestB, dtype={'charge_start_time': str, 'charge_end_time': str})
    initialData['tag'] = 'DATA' 
    testDataB['tag'] = 'B'
    # 拼接TEST和原始数据
    all_data = pd.concat([initialData, testDataB], axis = 0)
    all_data.sort_values(by=['vehicle_id', 'charge_start_time'], inplace=True) 
    all_data.reset_index(drop=True, inplace=True)
    # 调用preprocess_features 数据预处理，新增特征
    process_all_data = preprocess_features(all_data, False)  
    process_testB = process_all_data[process_all_data['tag'].isin(['B'])].copy() 
    process_all_data.drop(process_all_data.loc[process_all_data['tag'] == 'B'].index.tolist(), inplace=True) 
    process_all_data1 = process_all_data.drop(['tag'], axis=1)
    process_testB.drop(['tag'], axis=1, inplace=True) 
    process_testB.drop(['charge_energy'], axis=1, inplace=True)
    train_features = process_all_data1.reset_index(drop=True)
    train_features.to_csv('../dataset/feature/energy_train_1029_features.csv', index=False) 
    process_testB.to_csv('../dataset/feature/energy_test1029_features.csv', index=False)

    # car 1 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([1])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car1_features = car1_feature(process_feature_car, True)
    car1_features.reset_index(drop=True, inplace=True)
    car1_features.to_csv('../dataset/feature/train_feature/car1_features.csv', index=False)
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([1])].copy()
    car1testB_features = car1_feature(process_feature_cartestB, False) 
    car1testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car1testB_features.to_csv('../dataset/feature/test_feature/car1testB_features.csv', index=False) 
    # car 2 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([2])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car2_features = car2_feature(process_feature_car, True)
    car2_features.reset_index(drop=True, inplace=True)
    car2_features.to_csv('../dataset/feature/train_feature/car2_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([2])].copy()
    car2testB_features = car2_feature(process_feature_cartestB, False) 
    car2testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car2testB_features.to_csv('../dataset/feature/test_feature/car2testB_features.csv', index=False) 
    # car 3 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([3])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car3_features = car3_feature(process_feature_car, True)
    car3_features.reset_index(drop=True, inplace=True)
    car3_features.to_csv('../dataset/feature/train_feature/car3_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([3])].copy()
    car3testB_features = car3_feature(process_feature_cartestB, False) 
    car3testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car3testB_features.to_csv('../dataset/feature/test_feature/car3testB_features.csv', index=False)
    # car 4 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([4])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car4_features = car4_feature(process_feature_car, True)
    car4_features.reset_index(drop=True, inplace=True)
    car4_features.to_csv('../dataset/feature/train_feature/car4_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([4])].copy()
    car4testB_features = car4_feature(process_feature_cartestB, False) 
    car4testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car4testB_features.to_csv('../dataset/feature/test_feature/car4testB_features.csv', index=False)
    # car 5 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([5])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car5_features = car5_feature(process_feature_car, True)
    car5_features.reset_index(drop=True, inplace=True)
    car5_features.to_csv('../dataset/feature/train_feature/car5_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([5])].copy()
    car5testB_features = car5_feature(process_feature_cartestB, False) 
    car5testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car5testB_features.to_csv('../dataset/feature/test_feature/car5testB_features.csv', index=False)
    # car 6 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([6])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car6_features = car6_feature(process_feature_car, True)
    car6_features.reset_index(drop=True, inplace=True)
    car6_features.to_csv('../dataset/feature/train_feature/car6_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([6])].copy()
    car6testB_features = car6_feature(process_feature_cartestB, False) 
    car6testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car6testB_features.to_csv('../dataset/feature/test_feature/car6testB_features.csv', index=False)
    # car 7 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([7])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car7_features = car7_feature(process_feature_car, True)
    car7_features.reset_index(drop=True, inplace=True)
    car7_features.to_csv('../dataset/feature/train_feature/car7_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([7])].copy()
    car7testB_features = car7_feature(process_feature_cartestB, False) 
    car7testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car7testB_features.to_csv('../dataset/feature/test_feature/car7testB_features.csv', index=False)
    # car 8 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([8])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car8_features = car8_feature(process_feature_car, True)
    car8_features.reset_index(drop=True, inplace=True)
    car8_features.to_csv('../dataset/feature/train_feature/car8_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([8])].copy()
    car8testB_features = car8_feature(process_feature_cartestB, False) 
    car8testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car8testB_features.to_csv('../dataset/feature/test_feature/car8testB_features.csv', index=False)
    # car 9 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([9])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car9_features = car9_feature(process_feature_car, True)
    car9_features.reset_index(drop=True, inplace=True)
    car9_features.to_csv('../dataset/feature/train_feature/car9_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([9])].copy()
    car9testB_features = car9_feature(process_feature_cartestB, False) 
    car9testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car9testB_features.to_csv('../dataset/feature/test_feature/car9testB_features.csv', index=False)
    # car 10 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([10])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car10_features = car10_feature(process_feature_car, True)
    car10_features.reset_index(drop=True, inplace=True)
    car10_features.to_csv('../dataset/feature/train_feature/car10_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([10])].copy()
    car10testB_features = car10_feature(process_feature_cartestB, False) 
    car10testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car10testB_features.to_csv('../dataset/feature/test_feature/car10testB_features.csv', index=False)
    # car 11 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([11])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car11_features = car11_feature(process_feature_car, True)
    car11_features.reset_index(drop=True, inplace=True)
    car11_features.to_csv('../dataset/feature/train_feature/car11_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([11])].copy()
    car11testB_features = car11_feature(process_feature_cartestB, False) 
    car11testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car11testB_features.to_csv('../dataset/feature/test_feature/car11testB_features.csv', index=False)
    # car 12 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([12])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car12_features = car12_feature(process_feature_car, True)
    car12_features.reset_index(drop=True, inplace=True)
    car12_features.to_csv('../dataset/feature/train_feature/car12_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([12])].copy()
    car12testB_features = car12_feature(process_feature_cartestB, False) 
    car12testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car12testB_features.to_csv('../dataset/feature/test_feature/car12testB_features.csv', index=False)
    # car 13 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([13])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car13_features = car13_feature(process_feature_car, True)
    car13_features.reset_index(drop=True, inplace=True)
    car13_features.to_csv('../dataset/feature/train_feature/car13_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([13])].copy()
    car13testB_features = car13_feature(process_feature_cartestB, False) 
    car13testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car13testB_features.to_csv('../dataset/feature/test_feature/car13testB_features.csv', index=False)
    # car 14 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([14])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car14_features = car14_feature(process_feature_car, True)
    car14_features.reset_index(drop=True, inplace=True)
    car14_features.to_csv('../dataset/feature/train_feature/car14_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([14])].copy()
    car14testB_features = car14_feature(process_feature_cartestB, False) 
    car14testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car14testB_features.to_csv('../dataset/feature/test_feature/car14testB_features.csv', index=False)
    # car 15 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([15])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car15_features = car15_feature(process_feature_car, True)
    car15_features.reset_index(drop=True, inplace=True)
    car15_features.to_csv('../dataset/feature/train_feature/car15_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([15])].copy()
    car15testB_features = car15_feature(process_feature_cartestB, False) 
    car15testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car15testB_features.to_csv('../dataset/feature/test_feature/car15testB_features.csv', index=False)
    # car 16 =======================================================================================
    process_feature_car = train_features[train_features['vehicle_id'].isin([16])].copy()
    process_feature_car.reset_index(drop=True, inplace=True)
    car16_features = car16_feature(process_feature_car, True)
    car16_features.reset_index(drop=True, inplace=True)
    car16_features.to_csv('../dataset/feature/train_feature/car16_features.csv', index=False)  
    process_feature_cartestB = process_testB[process_testB['vehicle_id'].isin([16])].copy()
    car16testB_features = car16_feature(process_feature_cartestB, False) 
    car16testB_features.drop(['vehicle_id'], axis=1, inplace=True)
    car16testB_features.to_csv('../dataset/feature/test_feature/car16testB_features.csv', index=False)
    
    
    
    