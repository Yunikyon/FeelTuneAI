from datetime import datetime
import context.extract as extract
import numpy as np


def get_weather_type(data, id):
    if data is None:
        return "N/A"
    for item in data['data']:
        if item['idWeatherType'] == id:
            return item['descWeatherTypeEN']
    return "N/A"


def get_precipitation_type(data, id):
    if data is None:
        return "N/A"
    if np.isnan(id):
        return "No precipitation"
    for item in data['data']:
        if int(item['classPrecInt']) == id:
            return item['descClassPrecIntEN']
    return "N/A"


def get_wind_speed_type(data, id):
    if data is None:
        return "N/A"
    for item in data['data']:
        if int(item['classWindSpeed']) == id:
            return item['descClassWindSpeedDailyEN']
    return "N/A"


def transform_hours_into_day_classification(date):
    if date is None:
        return "N/A"
    hours = date.split("T")[1]
    if "00:00:00" <= hours < "05:00:00":
        return 'Night'
    if "05:00:00" <= hours < "08:00:00":
        return 'Early Morning'
    if "08:00:00" <= hours < "12:00:00":
        return 'Morning'
    if "12:00:00" <= hours < "18:00:00":
        return 'Afternoon'
    if "18:00:00" <= hours < "21:00:00":
        return 'Evening'
    if "21:00:00" <= hours < "24:00:00":
        return 'Night'
    return 'N/A'


'''
    Night: 12am to 5am
    Early Morning: 5am to 8am
    Morning: 8am to 12pm
    Afternoon: 12pm to 6pm
    Evening: 6pm to 9pm
    Nighttime: 9pm to 12am
'''


def get_is_work_day(date):
    if date is None:
        return None
    if "T" not in date:
        return None
    day = date.split("T")[0]
    day = datetime.strptime(day, '%Y-%m-%d').date()
    if day.weekday() < 5:
        return 'Yes'
    # Call api to check whether current day is an holiday or not
    api_key = "3db2fcbdd9654b2d8d0483278ec5a7c3"
    d, m, y = day.day, day.month, day.year
    result = extract.getJsonResponseFromUrl(f"https://holidays.abstractapi.com/v1?api_key={api_key}&country=PT&day={d}&month={m}&year={y}")
    if result is None:
        return 'No'
    result = result.json()
    # If there is a holiday, then the response is not an empty array
    if len(result) != 0:
        return 'No'
    return 'Yes'

