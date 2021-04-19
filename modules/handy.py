import numpy as np
import pandas as pd

format_slash = '%d/%m/%Y %H:%M'
format_dash = '%Y-%m-%d %H:%M:%S'
# 1.10.19 0:25
format_dots = '%d.%m.%Y %H:%M:%S'
# 30-11-20 18:59
format_dash_short = '%d-%m-%y %H:%M'

# unix epoch
epoch = pd.Timestamp("1970-01-01")

# just a reminder how to convert from string to datetime column
# use this for columns: created and resolved

def to_datetime(df, input_column, output_column, fmt=format_dash, errors='coerce'):
    df[output_column] = pd.to_datetime(df[input_column], format=fmt, errors=errors)
    return df

# unix timestamp from DateTimeIndex
def to_unix(s):
    return (s - epoch) // pd.Timedelta("1s")

    # alternative implementation would be:
    # return s.astype('int64') // int(1e9)

# series of pd.Timestamps -> DateTimeIndex of pd.Timestamps
def series_to_dateTimeIndex(s):
    return pd.to_datetime(s.values)

# requirement: dataframe must have 'created' and 'resolved' fields
def augment_columns(df):
    df['delta'] = df['resolved'] - df['created']
    df['delta_m'] = df.apply(lambda r: r['delta'].total_seconds() / 60, axis=1)
    # time of day of ticket open (float)
    df['tod'] = df.apply(lambda row: row['created'].hour + row['created'].minute / 60, axis=1)
    df['weekday'] = df.apply(lambda row: row['created'].weekday(), axis=1)
    df['hour'] = df.apply(lambda row: row['created'].hour, axis=1)
    # time of week (float)
    df['tow'] = df.weekday + df['tod'] / 24
    df['weekhour'] = np.floor(df.tow * 24)

    start = df.created.min()
    # monday before our 1st incident
    start_week_1 = monday_before(start)
    # number of day (and week) from beginning of time, that is Monday before the first noted event
    df['day_nr'] = df.apply(lambda r: (r.created - start_week_1).days, axis=1)
    df['week_nr'] = df.day_nr // 7

    return df


# return Monday 00:00:00 before given moment
def monday_before(now):
    monday = now - pd.Timedelta(now.weekday(), 'days')
    # Monday 00:00:00
    return pd.Timestamp(monday.date())


# return Monday 00:00:00 after given moment
def monday_after(now):
    # trick: compute Monday before 1 week from now... it's the same.
    return monday_before(now + pd.Timedelta(7, 'days'))


# use this to have full week span, spanning tne entire period
# returns: Monday before, Monday after, number of weeks between
def outer_week_boundaries(series):
    start, end = monday_before(series.min()), monday_after(series.max())
    return start, end, (end - start).days / 7


def inner_week_boundaries(series):
    start, end = monday_after(series.min()), monday_before(series.max())
    return start, end, (end - start).days / 7


# return: array of weekhour histograms, median histogram, and average histogram

# exact number of days, including fraction of day (float)
def fractional_days(data_start, data_end):
    delta = data_end - data_start
    return delta.days + delta.seconds / (60 * 60 * 24)


# number of full 24-hour periods
def inner_days(data_start, data_end):
    return (data_end - data_start).days


# number of days between midnight-before-first-record and midnight-after-last-record
def outer_days(data_start, data_end):
    return (data_end.date() - data_start.date()).days + 1


def weekly_bin_edges(outer_start, howmany):
    # add 1 for we count bin edges rather than bins
    week = pd.Timedelta(7, 'days')
    return [outer_start + i * week for i in np.arange(howmany + 1)]


def daily_bin_edges(start, howmany):
    # add 1 for we count bin edges rather than bins
    day = pd.Timedelta(1, 'days')
    return [start.date() + i * day for i in np.arange(howmany + 1)]





class WeeklyStats:

    def __init__(self, data) -> object:
        self.outer_start, self.outer_end, self.outer_weeks = outer_week_boundaries(data)
        self.inner_start, self.inner_end, self.inner_weeks = inner_week_boundaries(data)
        self.data_start, self.data_end = data.min(), data.max()
        self.weekly_bins = weekly_bin_edges(self.outer_start, self.outer_weeks)

        self.days = fractional_days(self.data_start, self.data_end)
        self.outer_days = outer_days(self.data_start, self.data_end)
        self.daily_bins = daily_bin_edges(self.data_start, self.outer_days)
        self.weeks = self.days / 7

        # to be implemented

        # numpy histogram works with numbers only, that's why...
        #to_timestamp = np.vectorize(lambda x: x.timestamp())
        # to_timestamp = np.vectorize(lambda x: x.value)
        #to_timestamp = np.vectorize(lambda x: x.total_seconds())
        #ts_data = to_timestamp(data)
        #self.week_values, _ = np.histogram(ts_data, bins=self.weekly_bins)
        #self.fullweek_values = self.week_values[1:-1]

        #self.weekly_minimum = self.fullweek_values.min()
        #self.weekly_maximum = self.week_values.max()

        #self.day_values, _ = np.histogram(ts_data, bins=self.daily_bins)
        #self.fullday_values = self.day_values[1:-1]
        #self.daily_minimum = self.fullday_values.min()
        #self.daily_maximum = self.daily_values.max()

