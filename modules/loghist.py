# 
# This module contains some utilities for drawing histograms with logarithmic scale
# and more.
#


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Draw histogram of df[field] on a linear x scale.
# This function is assuming that the data represents time delta, expressed as floats that represent either seconds or minutes.
# The histogram will be drawn in 8 versions, each showing larger order of magnitude of the data.
# Parameters:
# unit = 'second' or 'minute', depending what your data represents
 
def hist8t(df, 
           field,  
           unit):
    
    fig, ax = plt.subplots(4,2, figsize = (20,9))
    ax = ax.flatten()

    data = df[field]
    
    if unit == 'second':
        pass
    elif unit == 'minute':
        data = data * 60
    else :
        print(f'Unit: {unit}')
        raise ValueError('Unit must be second or minute')

    # Now the data has been normalized: unit is second
            
    second = 1
    minute = 60 * second
    quarter = 15 * minute 
    hour = 60 * minute
    day = hour * 24
    week = day * 7
    month = day * 31
    year = week * 53

    axis = ax[0]
    h = axis.hist(stacked = True, 
                  x = data, 
                  bins = 60, 
                  range = (0, minute))
    axis.set_title('Range: 1 min, bin: 1 second')

    axis = ax[1]
    axis.hist(stacked = True, 
              x = data, 
              bins = 5 * 6, 
              range = (0, 5 * minute))
    axis.set_title('Range: 300 seconds(5 mins), bin: 10 seconds')

    axis = ax[2]
    axis.hist(stacked = True, 
              x = data / minute, 
              bins = 60, 
              range = (0, hour // minute))
    #axis.set_ylim((0,10000))
    axis.set_title('Range: 1 hour, bin: 1 minute')

    axis = ax[3]
    axis.hist(stacked = True, 
              x = data / quarter, 
              bins = 4 * 24, 
              range = (0, day // quarter))
    #axis.set_ylim((0,200))
    axis.set_title('Range: 1 day, bin: 1 quarter (15 mins)')

    axis = ax[4]
    axis.hist(stacked = True, 
              x = data / hour, 
              bins = 24 * 7, 
              range = (0,week // hour))
    #axis.set_ylim((0,600))
    axis.set_title('Range: 1 week, bin: 1 hour')

    axis = ax[5]
    axis.hist(stacked = True, 
              x = data / day, 
              bins = 31, 
              range = (0, month // day))
    #axis.set_ylim((0,1000))
    axis.set_title('Range: 32 days, bin: 1 day')

    axis = ax[6]
    axis.hist(stacked = True, 
              x = data / week, 
              bins = 53, 
              range = (0, year // week))
    #axis.set_ylim((0,1000))
    axis.set_title('Range: one year, bin: 1 week')

    axis = ax[7]
    axis.hist(stacked = True, 
              x = data / day, 
              bins = 100)
    #axis.set_ylim((0,20))
    axis.set_title('No range, bin: automatic')

    plt.tight_layout()
    plt.show()
    
# stacked version of hist8t.
# parameters:
# group_field = based on this field the data will be grouped, and then stacked
# (groups to be represented as different colors in a stacked histogram)

def hist8t_stacked(df, 
           field,  
           unit,
           group_field):
    
    fig, ax = plt.subplots(4,2, figsize = (20,9))
    ax = ax.flatten()

    if unit == 'second':
        pass
    elif unit == 'minute':
        df = df.copy()
        df[field] = df[field] * 60
    else :
        print(f'Unit: {unit}')
        raise ValueError('Unit must be second or minute')

    # Now the data has been normalized: unit is second

    # if no stacking
    labels = None
    mygroups = df[field]

    # if stacking
    if(group_field != None):
        mygroups, labels = [], []
        for g, records in df.groupby(group_field):
            mygroups.append(records[field])
            labels.append(g)

            
    second = 1
    minute = 60 * second
    quarter = 15 * minute 
    hour = 60 * minute
    day = hour * 24
    week = day * 7
    month = day * 31
    year = week * 53

    axis = ax[0]
    h = axis.hist(stacked = True, 
                  x = mygroups, 
                  bins = 60, 
                  range = (0, minute))
    axis.set_title('Range: 1 min, bin: 1 second')

    axis = ax[1]
    axis.hist(stacked = True, 
              x = mygroups, 
              bins = 5 * 6, 
              range = (0, 5 * minute))
    axis.set_title('Range: 300 seconds(5 mins), bin: 10 seconds')

    axis = ax[2]
    axis.hist(stacked = True, 
              x = [g / minute for g in mygroups] , 
              bins = 60, 
              range = (0, hour // minute))
    #axis.set_ylim((0,10000))
    axis.set_title('Range: 1 hour, bin: 1 minute')

    axis = ax[3]
    axis.hist(stacked = True, 
              x = [g / quarter for g in mygroups], 
              bins = 4 * 24, 
              range = (0, day // quarter))
    #axis.set_ylim((0,200))
    axis.set_title('Range: 1 day, bin: 1 quarter (15 mins)')

    axis = ax[4]
    axis.hist(stacked = True, 
              x = [g / hour for g in mygroups], 
              bins = 24 * 7, 
              range = (0,week // hour))
    #axis.set_ylim((0,600))
    axis.set_title('Range: 1 week, bin: 1 hour')

    axis = ax[5]
    axis.hist(stacked = True, 
              x = [g / day for g in mygroups], 
              bins = 31, 
              range = (0, month // day))
    #axis.set_ylim((0,1000))
    axis.set_title('Range: 32 days, bin: 1 day')

    axis = ax[6]
    axis.hist(stacked = True, 
              x = [g / week for g in mygroups], 
              bins = 53, 
              range = (0, year // week))
    #axis.set_ylim((0,1000))
    axis.set_title('Range: one year, bin: 1 week')

    axis = ax[7]
    axis.hist(stacked = True, 
              x = [g / day for g in mygroups], 
              bins = 100)
    #axis.set_ylim((0,20))
    axis.set_title('No range, bin: automatic')

    plt.tight_layout()
    plt.show()
    

    
# Draw histogram of df[field] on a logarithmic x scale.
# Parameters:
# bin density = how many bins should fit in x axis representing one order of magnitude 
# (e.g. between 10 and 100)
# group_field = if provided, the histogram will be stacked, and the stacked groups (colors) will derive
# from grouping of records based on this field.

def lhist(axis, 
          df, 
          field, 
          group_field = None, 
          bin_density = 16,
          ax_titles = None,
         annotate = True):

    # cannot present negative values on logarithm scale

    df = df[df[field] >=0 ]

    # Do we have zeros in data? Then we will cheat a little, and upgrade zero to some small value
    smallest_observed_value = df[df[field] >0][field].min()
    almost_zero = smallest_observed_value / 2
    df.loc[df[field] == 0, field] = almost_zero
    
    data = df[field]

    labels = None
    mygroups = df[field]
    
    if(group_field != None):
        mygroups, labels = [], []
        for g, records in df.groupby(group_field):
            mygroups.append(records[field])
            labels.append(g)

    # prepare bins in log10 scale, 
    # make edges equally matching the powers of 10.
    start = np.floor(np.log10(data.min()))
    stop = np.ceil(np.log10(data.max()))
    # return numbers spaced evenly on log scale, starting from base ** start to base ** stop
    bins = np.logspace(start = start, stop = stop, num = int(stop - start) * bin_density)

    axis.hist(stacked = True, x = mygroups, bins = bins, label = labels)
    plt.xscale('log')
    
    axis.axvline(x=almost_zero, color='r', linestyle='dashed', linewidth=2, label = 'zero')
    axis.axvline(x=smallest_observed_value, color='b', linestyle='dashed', linewidth=2, label = '{:.3}'.format(smallest_observed_value))
    axis.axvline(x=df[field].median(), color='black', linestyle='dashed', linewidth=2, label = 'median')
    from matplotlib.offsetbox import AnchoredText
    if (annotate):
        message = 'Red dash represents zero.\nThere are {} zero values\nBlue dash = smallest positive value'.format(len(data[data == almost_zero]))
        anchored_text = AnchoredText(message, loc=4)
        axis.add_artist(anchored_text)
    axis.legend()
    
 