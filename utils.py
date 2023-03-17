import requests
import numpy as np


def post_request_image(url, img):
    headers = {'Content-Type': 'image/tif'}
    response = requests.post(f'{url}/get_embed', data=img, headers=headers)
    if response.status_code != 200:
        print('Error posting image:', response.status_code)
        
def get_request_image(url):
    response = requests.get(f'{url}/get_image')
    return response.content
