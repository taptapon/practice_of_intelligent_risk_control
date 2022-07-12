# -*- coding: utf-8 -*-

import time
import pytz
import numpy as np
import datetime as dt
from dateutil.parser import parse


def stamp_to_date(time_stamp, timezone=None):
    """
    时间戳转日期函数
    :param time_stamp:int，时间戳
    :param timezone:string，时区
    :return: datetime
    """
    try:
        if timezone is None:
            stamp_str = str(time_stamp)
            if len(stamp_str) >= 10:
                stamp_str = stamp_str[:10]
            else:
                stamp_str = stamp_str
            time_stamp = int(stamp_str)
            date = dt.datetime.fromtimestamp(time_stamp)
            return date
        else:
            stamp_str = str(time_stamp)
            if len(stamp_str) >= 10:
                stamp_str = stamp_str[:10]
            else:
                stamp_str = stamp_str
            time_stamp = int(stamp_str)
            tz = pytz.timezone(timezone)
            date = dt.datetime.fromtimestamp(time_stamp, tz).strftime('%Y-%m-%d %H:%M:%S')
            date = parse(date)
            return date
    except:
        return parse('2100-01-01')


def date_to_stamp(date_time):
    """
    将日期转换为时间戳
    :param date_time: string，datetime
    :return: int
    """
    try:
        if isinstance(date_time, str):
            date_time = parse(date_time)
        return int(time.mktime(date_time.timetuple()))
    except:
        return int(631123200)


def date_to_week(date):
    '''
    日期转换为星期
    :param date:datetime，string
    :return: int
    '''
    try:
        if isinstance(date, str):
            date = parse(date)
        if_weekend = date.weekday()
        return if_weekend
    except:
        return np.nan
