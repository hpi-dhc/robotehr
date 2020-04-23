import json

import requests


def http_post(url, data, headers={}):
    default_headers = {"Content-Type": "application/json"}
    response = requests.post(
        url, data=json.dumps(data), headers={**default_headers, **headers},
    )
    if response.status_code != 200:
        raise ValueError(
            "Request returned an error %s, the response is:\n%s"
            % (response.status_code, response.text)
        )
    return response
