import requests
import json

# Porto -> Latitude: 41.15, Longitude: -8.61024
# Lisboa -> Latitude: 38.71667, Longitude: -9.13333
# Leiria -> Latitude: 39.7473, Longitude: -8.8069
# Angra do Heroísmo -> Latitude: 38.6586, Longitude: -27.2159
# api_url = "https://api.api-ninjas.com/v1/weather?lat=39.7473&lon=-8.8069"
# api_url = "https://api.api-ninjas.com/v1/weather?lat=39.7473&lon=-8.8069"
# response = requests.get(api_url, headers={'X-Api-Key': 'gsNY5AepnuvZYcIOG0q4rg==W9sSXxN89hDQSAi0'})
# if response.status_code == requests.codes.ok:
#     print(response.text)
# else:
#     print("Error:", response.status_code, response.text)


def get_ipma_data():
    response = getJsonResponseFromUrl("http://api.ipma.pt/open-data/forecast/meteorology/cities/daily/hp-daily-forecast-day0.json")
    if response is None:
        return None
    response = response.json()
    date = response['forecastDate']
    forecast_data = response['data']
    global_ids_local = getGlobalIdLocal()
    for item in forecast_data:
        global_id = item['globalIdLocal']
        local = global_ids_local[global_id]
        item['local'] = local
        item['forecastDate'] = date
        item['updateTime'] = response['dataUpdate']
    return forecast_data


def getJsonResponseFromUrl(url, header_args=""):
    headers = {}
    if header_args != "":
        if "," in header_args:
            total_headers = header_args.split(',')
            for header in total_headers:
                key, value = header.split(':')
                headers[key] = value
        else:
            # When I only pass 1 header
            key, value = header_args.split(':')
            headers[key] = value
    try:
        request = requests.get(url, headers=headers, timeout=5)
        if not request.ok:
            print("ERROR: ", request.text)
            return None
        return request
    except ConnectionError as e:
        print("ERROR: ", e)
        return None


def getResponseFromUrl(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response


def getGlobalIdLocal():
    # Gets the corresponding globalIdLocal for each location
    response = requests.get("https://api.ipma.pt/open-data/distrits-islands.json")
    if response.status_code != 200:
        return None

    data = json.loads(response.text)
    values = {}
    for item in data['data']:
        global_id = item['globalIdLocal']
        local = item['local']
        values[global_id] = local
    return values

