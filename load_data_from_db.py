import requests
import pandas as pd

pd.set_option('display.max_columns', None)

def reshape_data(product):
    new_prod = product[0]
    new_prod['rating'] = product[1]
    return new_prod

def load_data():
    response = requests.get('http://localhost:8000/products')
    data = response.json()
    data = list(map(reshape_data, data))
    df = pd.DataFrame(data)
    return df