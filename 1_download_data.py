import requests
import json
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Replace with your actual client_id and client_secret
client_id = '8617'
client_secret = ''

# Authorization URL
auth_url = (
    f'https://www.strava.com/oauth/authorize?client_id={client_id}&'
    'redirect_uri=http://seriot.ch&response_type=code&scope=activity:read&approval_prompt=auto'
)

print("Open the following URL in your browser to authorize the application:")
print(auth_url)

# You need to manually retrieve the authorization code from the URL after authorization
authorization_code = input("Enter the authorization code: ")

# Requesting the access token
payload = {
    'client_id': client_id,
    'client_secret': client_secret,
    'code': authorization_code,
    'grant_type': 'authorization_code'
}

response = requests.post('https://www.strava.com/api/v3/oauth/token', data=payload, verify=False)
tokens = response.json()
access_token = tokens['access_token']
refresh_token = tokens['refresh_token']

print(f"Access Token = {access_token}")
print(f"Refresh Token = {refresh_token}")

# Function to refresh the access token
def refresh_access_token(client_id, client_secret, refresh_token):
    payload = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token
    }
    response = requests.post('https://www.strava.com/api/v3/oauth/token', data=payload, verify=False)
    tokens = response.json()
    return tokens['access_token'], tokens['refresh_token']

# Fetching activities
page = 1
status = 200

while status == 200 and page < 10:
    print(f"-- Fetching page {page}")

    header = {'Authorization': 'Bearer ' + access_token}
    param = {'per_page': 200, 'page': page}
    response = requests.get("https://www.strava.com/api/v3/athlete/activities", headers=header, params=param, verify=False)

    status = response.status_code

    if status == 401:  # Unauthorized, refresh the token
        print("Access token expired, refreshing token...")
        access_token, refresh_token = refresh_access_token(client_id, client_secret, refresh_token)
        header = {'Authorization': 'Bearer ' + access_token}
        response = requests.get("https://www.strava.com/api/v3/athlete/activities", headers=header, params=param, verify=False)
        status = response.status_code

    if status == 200:
        data = response.json()
        with open(f'activities_page_{page}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        page += 1
    else:
        print(f"Error: received status code {status}")

print("Activities fetched successfully.")
