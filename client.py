import requests
from Crypto.Util import number
from utils import fast_power
from Crypto.Random import random
import json
import hashlib
from utils import int_to_bytes, bytes_to_int
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
        self._R2 = 0
        self._R3 = 0
        self._login =""
        self._password = ""
    def register(self):
        r = requests.post(SERVER_ADDR, data= json.dumps({"type" : "getR2"}))
        data = r.json()
        self._R2 = data["R2"]
        print("write your login pls")
        print(">> ",end='')
        self._login = input().strip()
        print("enter password")
        print(">> ", end='')
        password = input().strip()
        m = hashlib.sha256()
        m.update(bytes(password,'utf-8'))
        m.update(int_to_bytes(self._R2))
        password = m.digest()
        password = bytes_to_int(password)
        data = (requests.post(SERVER_ADDR, data=json.dumps({"type": "register","login": self._login, "password": password})))
        print(data)
        data= data.json()
        while (data["error"] == "accupied"):
            print("choose another login pls")
            print(">> ", end='')
            self._login = input().strip()
            data = (requests.post(SERVER_ADDR, data=json.dumps({"type": "register", "login": self._login, "password":password}))).json()
        print("succefful registration")
        self._password = password
    def autorize(self):

        print("write your login pls")
        print(">> ", end='')
        self._login = input().strip()
        r = requests.post(SERVER_ADDR, data=json.dumps({"type": "getR3", "login": self._login}))
        data = r.json()
        self._R3 = data["R3"]
        print("enter password")
        print(">> ", end='')
        password = input().strip()
        m = hashlib.sha256()
        m.update(bytes(password, 'utf-8'))
        m.update(int_to_bytes(self._R2))
        password = m.digest()
        password = bytes_to_int(password)
        self._password = password
        m = hashlib.sha256()
        m.update(int_to_bytes(password))
        m.update(int_to_bytes(self._R3))
        password = m.digest()
        password = bytes_to_int(password)

        data = (requests.post(SERVER_ADDR,
                             data=json.dumps({"type": "autorize", "login": self._login, "password": password}))).json()
        if (data["error"] == "password_error"):
            print("wrong password")
        elif(data["error"] == "OK"):
            print("correct password")
            self._R3 = data["R3"]
        else:
            print("some error")

    def send_message_to_user(self, user, message):
        m = hashlib.sha256()
        m.update(int_to_bytes(self._password))
        m.update(int_to_bytes(self._R3))
        password = m.digest()
        password = bytes_to_int(password)
        data = requests.post(SERVER_ADDR, data=json.dumps({"type": "send", "login": self._login, "password": password,"user":user, "message" : message})).json()
        if data["error"] != "OK":
            r = requests.post(SERVER_ADDR, data=json.dumps({"type": "getR3", "login": self._login}))
            data = r.json()
            self._R3 = data["R3"]
            m = hashlib.sha256()
            m.update(int_to_bytes(self._password))
            m.update(int_to_bytes(self._R3))
            password = m.digest()
            password = bytes_to_int(password)
            data = requests.post(SERVER_ADDR, data=json.dumps(
                {"type": "send", "login": self._login, "password": self._password, "user": user,
                 "message": message})).json()
        if data["error"] != "OK":
            print(data["error"])
            return
        self._R3 = data["R3"]

    def recv_message(self):
        m = hashlib.sha256()
        m.update(int_to_bytes(self._password))
        m.update(int_to_bytes(self._R3))
        password = m.digest()
        password = bytes_to_int(password)
        data = requests.post(SERVER_ADDR,
                             data=json.dumps({"type": "recv", "login": self._login, "password": password})).json()
        if data["error"] != "OK":
            r = requests.post(SERVER_ADDR, data=json.dumps({"type": "getR3", "login": self._login}))
            data = r.json()
            self._R3 = data["R3"]
            m = hashlib.sha256()
            m.update(int_to_bytes(self._password))
            m.update(int_to_bytes(self._R3))
            password = m.digest()
            password = bytes_to_int(password)
            data = requests.post(SERVER_ADDR,
                             data=json.dumps({"type": "recv", "login": self._login, "password": password})).json()
        if data["error"] != "OK":
            print(data["error"])
            return
        self._R3 = data["R3"]
        print( data["message"])

        print(data)


print("hello, your enter my first client app")
user = User()
user.register()
user.autorize()
user.send_message_to_user("norrilsk","LOOOOOL")
user.recv_message()

#while (input().strip().lower() != "exit"):

