import requests
from Crypto.Util import number
from utils import fast_power
from Crypto.Random import random
import json
import hashlib
SERVER_ADDR = "http://127.0.0.1:8080"
prime = number.getPrime(256)


import hashlib
# r = requests.get('https://api.github.com/events')
# print( r.json())
# r = requests.post('https://httpbin.org/post', data = {'key':'value'})
# print(r.json())
# payload = {'key1': 'value1', 'key2': 'value2'}
# r = requests.get('https://httpbin.org/get', params=payload)
# #print(r.url)
# #print(r.text)
#
class User:
    def __init__(self):
        self._g = 0
        self._K = 0
        self._p = 0
        self._login =""
        self._password = ""
    def autorize(self):

        r = requests.post(SERVER_ADDR, data= json.dumps({"type" : "getpg"}))
        print(r)
        data = r.json()
        print("1st")
        self._p = data["p"]
        self._g = data["g"]
        B = data["B"]
        print(B)
        a = prime
        assert( (self._p != 0) and (self._g != 0) )
        print("write your login pls")
        print(">> ",end='')
        self._login = input().strip()
        print("enter password")
        print(">> ", end='')
        password = input().strip()
        m = hashlib.sha256()
        m.update(bytes(password))
        password = m
        A = fast_power(self._g,a,mod=self._p)
        self._K = fast_power(B,a,mod=self._p)
        password = password^self._K
        assert(self._K != 0)
        data = (requests.post(SERVER_ADDR, data=json.dumps({"type": "register","login": self._login, "A": A, "password": password})))
        print(data)
        data= data.json()
        while (data["error"] == "accupied"):
            print("choose another login pls")
            print(">> ", end='')
            self._login = input().strip()
            data = (requests.get(SERVER_ADDR, data=json.dumps({"type": "register", "login": self._login, "A" : A, "password":password}))).json()
        print("succefful autorization")
        self._password = password
    def send_message_to_user(self, user, message):
        data = requests.post(SERVER_ADDR, data=json.dumps({"type": "send", "login": self._login, "password": self._password,"user":user, "message" : message})).json()
    def recv_message(self):
        data = requests.post(SERVER_ADDR,
                            data=json.dumps({"type": "recv", "login": self._login, "password": self._password})).json()
        print(data)


print("hello, your enter my first client app")
user = User()
user.autorize()
user.send_message_to_user("norrilsk","LOOOOOL")
user.recv_message()

#while (input().strip().lower() != "exit"):

