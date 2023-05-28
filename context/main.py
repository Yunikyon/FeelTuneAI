import context.extract as extract
import context.transform as transform
from datetime import datetime
import pandas as pd
import geocoder
from geopy.distance import geodesic


def execute():
    # -------------------------- EXTRACTION PROCESS --------------------------
    start_time = datetime.now()
    print(f"Beginning Context Extract and Transformation Process at {start_time.strftime('%H:%M:%S')}")

    json_data = extract.get_ipma_data()
    if json_data is None:
        print("ERROR: No data was extracted")
        return None

    # Conversion from JSON to Dataframe
    df = pd.DataFrame(json_data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Filtrar o DataFrame para ter só a localização atual do sistema
    g = geocoder.ip('me')
    latitude, longitude = g.latlng
    print(f"Latitude: {latitude}, Longitude: {longitude}")
    df['distance'] = df.apply(lambda row: geodesic((latitude, longitude), (row['latitude'], row['longitude'])).km, axis=1) # km -> to return in km, axis = 1 -> apply to each row
    closest_city = df.loc[df['distance'].idxmin()]['local']
    print(f"Closest city: {closest_city}")
    df_mylocation = df[df['local'] == 'Leiria']

    values = []
    for index, row in df_mylocation.iterrows():
        response = extract.getJsonResponseFromUrl(f"https://api.sunrise-sunset.org/json?lat={row['latitude']}&lng={row['longitude']}&date={row['forecastDate']}")
        if response is None:
            continue
        response = response.json()
        new_record = {'globalIdLocal': row['globalIdLocal'], 'currentTime': datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}
        if response['status'] == 'OK':
            sunrise = response['results']['sunrise']
            sunset = response['results']['sunset']
            day_length = response['results']['day_length']
            new_record['sunrise'] = sunrise
            new_record['sunset'] = sunset
            new_record['day_length'] = day_length
        else:
            new_record['sunrise'] = "N/A"
            new_record['sunset'] = "N/A"
            new_record['day_length'] = "N/A"

        responseWeatherApi = extract.getJsonResponseFromUrl(
            f"https://api.api-ninjas.com/v1/weather?lat={row['latitude']}&lon={row['longitude']}",
            "X-Api-Key:gsNY5AepnuvZYcIOG0q4rg==W9sSXxN89hDQSAi0")
        if responseWeatherApi is None:
            continue
        responseWeatherApi = responseWeatherApi.json()

        for key in responseWeatherApi:
            if key == 'sunset' or key == 'sunrise' or key == 'day_length':
                continue
            new_record[key] = responseWeatherApi[key]

        values.append(new_record)

    df2 = pd.DataFrame(values)
    df3 = pd.merge(df_mylocation, df2, on='globalIdLocal')

    # -------------------------- TRANSFORM PROCESS --------------------------
    def get_code_to_value_transformation():
        info_weather_type = extract.getJsonResponseFromUrl("https://api.ipma.pt/open-data/weather-type-classe.json")
        if info_weather_type is not None:
            info_weather_type = info_weather_type.json()

        info_precipitation_type = extract.getJsonResponseFromUrl(
            "https://api.ipma.pt/open-data/precipitation-classe.json")
        if info_precipitation_type is not None:
            info_precipitation_type = info_precipitation_type.json()

        info_wind_class_type = extract.getJsonResponseFromUrl(
            "https://api.ipma.pt/open-data/wind-speed-daily-classe.json")
        if info_wind_class_type is not None:
            info_wind_class_type = info_wind_class_type.json()

        return info_weather_type, info_precipitation_type, info_wind_class_type

    info_weather_type, info_precipitation_type, info_wind_class_type = get_code_to_value_transformation()

    for index, row in df3.iterrows():
        # --- classPrecInt attribute transformation ---
        if  'classPrecInt' in df3.columns:
            precipitation_id = df3.at[index, 'classPrecInt']
            df3.at[index, 'classPrecInt'] = transform.get_precipitation_type(info_precipitation_type, precipitation_id)

        # --- classWindSpeed attribute transformation ---
        if 'classWindSpeed' in df3.columns:
            wind_speed_id = df3.at[index, 'classWindSpeed']
            df3.at[index, 'classWindSpeed'] = transform.get_wind_speed_type(info_wind_class_type, wind_speed_id)

        # --- idWeatherType attribute transformation ---
        if 'idWeatherType' in df3.columns:
            weather_id = df3.at[index, 'idWeatherType']
            df3.at[index, 'idWeatherType'] = transform.get_weather_type(info_weather_type, weather_id)

        # result = transform.transform_hours_into_day_classification(df3.at[index, 'currentTime'])
        # df3['timeOfDay'] = result.rstrip('\r')

        # --- sunrise and sunset attributes transformation ---
        sunrise = df3.at[index, 'sunrise']
        sunset = df3.at[index, 'sunset']
        df3.at[index, 'sunrise'] = datetime.strptime(sunrise, '%I:%M:%S %p').time()
        df3.at[index, 'sunset'] = datetime.strptime(sunset, '%I:%M:%S %p').time()

        df3['timeOfDay'] = transform.transform_hours_into_day_classification(df3.at[index, 'currentTime'])

    end_time = datetime.now()
    print(f"Finished Extraction and Transform Process at {end_time.strftime('%H:%M:%S')}")
    print(f"Time elapsed: {(end_time - start_time).total_seconds()} seconds")

    df3.drop(labels=['longitude', 'latitude', 'globalIdLocal', 'forecastDate', 'updateTime', 'distance', 'currentTime'], axis=1, inplace=True)
    return df3

