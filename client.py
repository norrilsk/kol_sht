import requests
from Crypto.Util import number
from utils import fast_power
from Crypto.Random import random
from Crypto.Cipher import Salsa20
import json
import hashlib
from utils import int_to_bytes, bytes_to_int
from base64 import b64encode, b64decode
import re
import threading
SERVER_ADDR = "http://127.0.0.1:8080"
prime = number.getPrime(256)
import time

import hashlib
import argparse

class User:
    def __init__(self,bio = 0):
        self._g = 0
        self._K = {}
        self._p = 0
        self._a = {}
        self._A = {}
        self._B = {}
        self._R2 = 0
        self._R3 = 0
        self._bio = bio
        self._login =""
        self._password = ""
        self._parser = self.createParser()
        self._authorized = False
        my_thread = threading.Thread(target=self.reciever_loop)
        my_thread.start()
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
        self._authorized= False
        r = requests.post(SERVER_ADDR, data=json.dumps({"type": "getR2"}))
        data = r.json()
        self._R2 = data["R2"]
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
            r = requests.post(SERVER_ADDR, data=json.dumps({"type": "getpg"}))
            data = r.json()
            self._p = data["p"]
            self._g = data["g"]
            self._authorized = True
        else:
            print("some error")

    def diffi_with_user(self,user):
        self._a[user] =  random.randint(2, 2**32) + self._bio
        self._A[user] =  fast_power(self._g,self._a[user],self._p)
        self.send_message_to_user(user,"razrabotchyk_pidor90585355972355049027130019104418304977474815248020730362080159511218759452819 "+ str(self._A[user]),from_diffi = True)


    def send_message_to_user(self, user, message,from_diffi = False):
        if not from_diffi and not self._K.get(user):
            self.diffi_with_user(user)
            while not self._K.get(user):
                time.sleep(1)
        if not from_diffi:
            key = int_to_bytes(self._K.get(user))[-32:]
            cipher = Salsa20.new(key=key)
            message = b64encode(cipher.nonce  + cipher.encrypt(message.encode('utf-8'))).decode('utf-8')


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
        if (data["message"].strip()):
            m = re.match(r"\s*(?P<name>\S+):\s*razrabotchyk_pidor90585355972355049027130019104418304977474815248020730362080159511218759452819\s+(?P<text>\d*)",  data["message"].strip())
            m2 = re.match(r"\s*(?P<name>\S+):\s*(?P<text>.*)",  data["message"].strip())
            if m:
                user = m.group("name")
                self._B[user] = int(m.group("text"))
                if not self._A.get(user):
                    self.diffi_with_user(user)
                self._K[user] = fast_power(self._B[user], self._a[user],self._p)
            elif m2 and  self._K.get(m2.group("name")):
                user = m2.group("name")
                text = m2.group("text")
                msg = b64decode(text)
                msg_nonce = msg[:8]
                ciphertext = msg[8:]
                cipher = Salsa20.new(key=int_to_bytes(self._K[user])[-32:], nonce=msg_nonce)
                plaintext = cipher.decrypt(ciphertext)
                print(user+":_ ", plaintext.decode('utf-8'))
            else:
                print( data["message"].strip())#fixme good enough for demo lol( 1 message per second)




    def createParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-receive', '-recv',   action='store_const', const=True , help="recieve all messages from the server")
        parser.add_argument('-register', '-reg', action='store_const', const=True, help = "register new user" )
        parser.add_argument('-authorize', '-au', '-a', action='store_const', const=True, help = "authorize into system")
        parser.add_argument('-send', '-s', nargs=2, metavar=('user', 'message'),  help = "Send message to user")


        return parser

    def parse(self, string):
        m = re.match(r"(-send|-s)\s+(?P<name>\S+)\s+(?P<text>.*)",string)
        if m:
            my_thread = threading.Thread(target=self.send_message_to_user, args=(m.group("name"), m.group("text")))
            my_thread.start()
            #self.send_message_to_user(m.group("name"), m.group("text"))
            return

        namespace = self._parser.parse_args(string.split())
        if (namespace.receive):
            self.recv_message()
        if (namespace.register):
            self.register()
        if (namespace.authorize):
            self.autorize()
        if (namespace.send is not None):
            self.send_message_to_user(namespace.send[0],namespace.send[1])
        return
    def reciever_loop(self):
        while True:
            if (self._authorized):
                self.recv_message()
            time.sleep(5)




print("hello, your enter my first client app, use --help ")
user = User()
exit = False
while not (exit):
    string = input()
    try:
        user.parse(string)
    except:
        pass



