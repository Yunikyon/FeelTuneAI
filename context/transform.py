import numpy as np


def get_weather_type(data, id):
    for item in data['data']:
        if item['idWeatherType'] == id:
            return item['descWeatherTypeEN']
    return "N/A"


def get_precipitation_type(data, id):
    if np.isnan(id):
        return "No precipitation"
    for item in data['data']:
        if int(item['classPrecInt']) == id:
            return item['descClassPrecIntEN']
    return "N/A"


def get_wind_speed_type(data, id):
    for item in data['data']:
        if int(item['classWindSpeed']) == id:
            return item['descClassWindSpeedDailyEN']
    return "N/A"


def transform_hours_into_day_classification(date):
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

