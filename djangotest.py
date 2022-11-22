import requests

def get_data():
    url = 'http://127.0.0.1:8000/getdata'
    x = requests.get(url)
    print(x.text)

def post_data():
    url = 'http://127.0.0.1:8000/getdata'
    data = {'choice': 50}
    x = requests.get(url, json=data)
    print(x.text)

if __name__ == "__main__":
    get_data()
