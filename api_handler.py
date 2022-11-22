import requests

url = 'http://127.0.0.1:5000/choice'
myobj = {'choice': 5}


def request(): 
    x = requests.post(url, json = myobj)
    print(x.text)
    return x.text

if __name__ == '__main__':
    request()
    