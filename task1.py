import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

dates = ['20140922', '20140923', '20140924', '20140925', '20140926', '20140927', '20140928',
         '20140929', '20140930', '20141001', '20141002', '20141003', '20141004', '20141005']

def read_data():
    buf = []
    for day in dates:
        df = pd.read_csv('~/ANM2017/Project_data_task1-4/%s.csv' % day)
        buf.append(df)
    df = pd.concat(buf, ignore_index=True)
    return df

def plot_cdf(df, col):
    img = df[col]
    ecdf = sm.distributions.ECDF(img)
    fig = plt.figure()
    x = np.linspace(min(img), max(img))
    y = ecdf(x)
    plt.plot(x,y)
    plt.suptitle('CDF of %s' % col)
    plt.xlabel(col)
    plt.show()

def plot_PVs(df):
    series = pd.Series(data=np.ones(len(df), dtype=np.int), index=pd.to_datetime(df['Timestamp'], unit='s'))
    freq = '1T'
    PVs_per_minute = series.resample(freq, closed='left', label='left').sum()
    plt.figure()
    PVs_per_minute.plot(x='Timestamp', y='PVs per minute')
    plt.suptitle('minute-level PVs line chart')
    plt.xlabel('Date Time')
    plt.ylabel('PVs per minute')
    plt.subplots_adjust(bottom=0.2)
    plt.show()

def plot_histogram(df):
    counts = df.groupby('Province').size()
    plt.figure()
    counts.plot(kind='bar', x='Province', y='PVs')
    plt.suptitle('PVs-Province histogram chart')
    plt.xlabel('Province')
    plt.ylabel('PVs')
    plt.subplots_adjust(bottom=0.25)
    plt.show()

df = read_data()
#plot_cdf(df, '#Images')
plot_PVs(df)
#plot_histogram(df)