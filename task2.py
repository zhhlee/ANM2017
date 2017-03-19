import pandas as pd
import numpy as np
import os
from scipy import stats

DATA_DIR = '/Users/zhh.lee/ANM2017/Project_data_task1-4/'

for root,dirs,files in os.walk(DATA_DIR):
    file_list = files[1:]

def read_data():
    buf = []
    for file in file_list:
        df = pd.read_csv(DATA_DIR + file)
        buf.append(df)
    df = pd.concat(buf, ignore_index=True)
    return df

def kendall_correlation(arr1, arr2):
    tau, p_value = stats.kendalltau(arr1, arr2)
    return tau

df = read_data()
'''
#calculate kendall correlation between SRT and Images
rk1 = df['#Images'].rank()
rk2 = df['SRT'].rank()
tau = kendall_correlation(rk1, rk2)
print(tau)             ##0.123028801451
'''

p_SRT = {}
p_Image = {}

p_SRT['SRT<800'] = len(df[df['SRT']<850].index) / len(df.index)
p_SRT['SRT>=800'] = 1 - p_SRT['SRT<800']

H_SRT = -(p_SRT['SRT<800']*np.log(p_SRT['SRT<800']) + p_SRT['SRT>=800']*np.log(p_SRT['SRT>=800']))

'''
#calculate RIG: SRT with Images
p_Image['Image<50'] = len(df[df['#Images']<50].index) / len(df.index)
p_Image['Image>=50'] = 1 - p_Image['Image<50']

df1 = df[df['#Images']<50]
p_00 = len(df1[df1['SRT']<800].index) / len(df1.index)    #P(SRT<800|Image<50)
p_10 = 1 - p_00      #P(SRT>=800|Image<50)
df2 = df[df['#Images']>=50]
p_01 = len(df2[df2['SRT']<800].index) / len(df2.index)    #P(SRT<800|Image>=50)
p_11 = 1 - p_01      #P(SRT>=800|Image>=50)

H_SRT_Img = p_Image['Image<50']*(-p_00*np.log(p_00) - p_10*np.log(p_10)) + p_Image['Image>=50'] * (-p_01*np.log(p_01) - p_11*np.log(p_11))
RIG_SRT_Img = (H_SRT - H_SRT_Img)/H_SRT

print(RIG_SRT_Img)          #0.00037376075773
'''

def RIG_SRT(attr):
    df_new = df[['SRT', attr]]
    total_len = len(df_new.index)
    H_SRT_attr = 0
    grouped = df_new.groupby(attr)
    for name, group in grouped:
        p_0 = len(group[group['SRT']<850].index) / len(group.index)
        p_1 = 1 - p_0
        entropy = len(group.index) / total_len * (-p_0*np.log(p_0) - p_1*np.log(p_1))
        H_SRT_attr += entropy
    RIG_SRT_attr = (H_SRT - H_SRT_attr) / H_SRT
    print(RIG_SRT_attr)

RIG_SRT('UA')        #0.0135425053653
RIG_SRT('ISP')        #0.00734623694718
RIG_SRT('Province')   #0.00401019675426
RIG_SRT('PageType')   #0.00888949711582

'''
# SRT threshold 900
0.0140136792118
0.00637017220712
0.00356705465566
0.00906073223474
'''