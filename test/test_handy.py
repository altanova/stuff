import logging
import os
import sys

import pandas as pd
import pytest
import handy as hd

log: logging.Logger


@pytest.fixture
def setup_logging():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    global log
    log = logging.getLogger('handy test')
    log.setLevel(logging.INFO)
    return log


def test_nothing(setup_logging):
    global log
    # this is to show how to use logging with pycharm + pytest
    # it will be printed if pytest is run with options `-p no:logging -s`
    # Add them to "Additional Arguments" in your Run Configuration or
    # Run Configuration Template for pytest
    print('\nignore the communicates from this method. This is test.')
    print('this is how to print to console, without logging')
    log.warning('Just a test warning message. Ignore it.')
    log.warning('This is how to print a warning with logging')
    log.info('This is how to print an info with logging')
    assert True, "dummy assertion"


def test_to_datetime():
    days = ['2021-04-05 00:00',  # Mon
            '2021-04-10 11:46',  # Sat
            '2021-04-11 23:59'  # Sun
            ]
    df = pd.DataFrame({'input': days})
    df = hd.to_datetime(df, input_column='input', output_column='output')
    assert df.output[2] == pd.to_datetime(days[2])


def test_monday_before_and_after():
    days = ['2021-04-05 00:00',  # Mon
            '2021-04-10 11:46',  # Sat
            '2021-04-11-23:59'  # Sun
            ]
    days_dt = pd.to_datetime(days)
    mon_before = pd.to_datetime('2021-04-05 00:00')
    mon_after = pd.to_datetime('2021-04-12 00:00')
    for d in days_dt:
        assert (hd.monday_before(d) == mon_before)
        assert (hd.monday_after(d) == mon_after)
    log.info('test_monday_before_after: PASS')


@pytest.fixture
def load_testdata_5m():
    # load the 5-months data set to be used for tests
    data_dir = '../data'
    src_file = 'sample01.csv'
    f = os.path.join(data_dir, src_file)
    df = pd.read_csv(f, encoding='latin_1', sep=';', error_bad_lines=False)
    df['created'] = pd.to_datetime(df['created'], format=hd.format_dash, errors='coerce')
    df['resolved'] = pd.to_datetime(df['resolved'], format=hd.format_dash, errors='coerce')
    df = hd.augment_columns(df)
    return df


# expect the data as loaded by load_testdata_5m
def test_week_boundaries(setup_logging, load_testdata_5m):
    df = load_testdata_5m
    # correct results
    inner_boundaries = (pd.Timestamp('2020-09-07 00:00:00'), pd.Timestamp('2021-01-25 00:00:00'), 20.0)
    outer_boundaries = (pd.Timestamp('2020-08-31 00:00:00'), pd.Timestamp('2021-02-01 00:00:00'), 22.0)
    assert (hd.inner_week_boundaries(df.created) == inner_boundaries)
    assert (hd.outer_week_boundaries(df.created) == outer_boundaries)


def test_weekly_stats(load_testdata_5m):
    df = load_testdata_5m
    ws = hd.WeeklyStats(df.created)
    assert hd is not None
    assert ws.data_start == pd.Timestamp('2020-09-01 00:37:07')
    assert ws.data_end == pd.Timestamp('2021-01-31 23:59:16')
