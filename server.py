# import requests
#
# r = requests.get('https://api.github.com/events')
# print( r.json())
# r = requests.post('https://httpbin.org/post', data = {'key':'value'})
# print(r.json())
# payload = {'key1': 'value1', 'key2': 'value2'}
# r = requests.get('https://httpbin.org/get', params=payload)
# #print(r.url)
# #print(r.text)

from http.server import BaseHTTPRequestHandler, HTTPServer
from Crypto.Util import number
from Crypto.Random import random, get_random_bytes
from utils import fast_power
from utils import bytes_to_int
from utils import int_to_bytes
import json
import hashlib

PORT_NUMBER = 8080

class Storage:
    def __init__(self):
        self._prime = number.getPrime(256)
        self._p = number.getPrime(256)
        self._g = random.randint(2, self._p)
        # self._B = fast_power(self._g, self._prime,mod = self._p)
        self._user_K = {}
        self._user_pas = {}
        self._user_mes = {}
        self._R2 = bytes_to_int(get_random_bytes(256 // 8))
        self._user_R3 = {}

storage = Storage()

# This class will handles any incoming request from
# the browser
class myHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):


        super().__init__(request, client_address, server)
    # Handler for the GET requests
    def do_GET(self):
        self.send_response(200)
        message = "hello jopka"
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        # Send the html message
        self.wfile.write(bytes(message, 'utf-8'))
        return

    def do_POST(self):
        self.send_response(200)

        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = (self.rfile.read(content_length)).decode("utf-8") # <--- Gets the data itself

        print(post_data)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        print(self.client_address)
        message = self.parse_args(post_data)
        # Send the html message
        if message:
            print(message)
            self.wfile.write(bytes(message, 'utf-8'))
        return
    def parse_args(self,data):
        print(data)
        data = json.loads(data)
        result = {}
        if (data["type"] == "getpg"):
            result["p"] = storage._p
            result["g"] = storage._g
            #result["B"] = storage._B
            return json.dumps(result)
        elif (data["type"] == "getR3"):
            if (storage._user_pas.get(data["login"])) is None:
                result["R3"] = storage._R2
                result["error"] = "OK"
                return json.dumps(result)
            else:
                result["R3"] = storage._user_R3[data["login"]]
                result["error"] = "OK"
                return json.dumps(result)
        elif (data["type"] == "getR2"):
            result["R2"] = storage._R2
            result["error"] = "OK"
            return json.dumps(result)
        elif (data["type"] == "register"):
            if (storage._user_pas.get(data["login"])) is None:
                storage._user_pas[data["login"]] = data["password"]
                storage._user_mes[data["login"]]  = {}
                storage._user_R3[data["login"]] = storage._R2
                print("u pas ", storage._user_pas)
                result["R3"] = storage._R2
                result["error"] = "OK"
                return json.dumps(result)
            else:
                result["error"] = "accupied"
                return json.dumps(result)
        elif (data["type"] == "autorize"):
            if (storage._user_pas.get(data["login"])) is None:
                result["error"] = "login_error"
                return json.dumps(result)
            login = data["login"]
            m = hashlib.sha256()
            m.update(int_to_bytes(storage._user_pas[login]))
            m.update(int_to_bytes(storage._user_R3[login]))
            t1 = bytes_to_int(m.digest())
            t2 = (data["password"])
            if (t1 == t2):
                storage._user_R3[login] += 1
                result["R3"] = storage._user_R3[login]
                result["error"] = "OK"
                return json.dumps(result)
            else:
                result["error"] = "password_error"
                return json.dumps(result)

        elif (data["type"] == "send"):
            if (storage._user_pas.get(data["login"])) is None:
                result["error"] = "login_error"
                return json.dumps(result)
            login = data["login"]
            m = hashlib.sha256()
            m.update(int_to_bytes(storage._user_pas[login]))
            m.update(int_to_bytes(storage._user_R3[login]))
            t1 = bytes_to_int(m.digest())
            t2 = (data["password"])
            if (t1 == t2):
                storage._user_R3[login] += 1
                result["R3"] = storage._user_R3[login]
                if ((storage._user_pas.get(data["user"])) is None):
                    result["error"] = "NOUSER"
                    return json.dumps(result)
                else:
                    if ((storage._user_mes[data["user"]].get(data["login"])) is not None):
                        storage._user_mes[data["user"]][data["login"]] += "\n" + str(data["message"])
                    else:
                        storage._user_mes[data["user"]][data["login"]] = str(data["message"])
                    result["error"] = "OK"
                    return json.dumps(result)
            else:
                result["error"] = "password_error"
                return json.dumps(result)
        elif (data["type"] == "recv"):
            if (storage._user_pas.get(data["login"])) is None:
                result["error"] = "login_error"
                return json.dumps(result)
            login = data["login"]
            m = hashlib.sha256()
            m.update(int_to_bytes(storage._user_pas[login]))
            m.update(int_to_bytes(storage._user_R3[login]))
            t1 = bytes_to_int(m.digest())
            t2 = (data["password"])
            if (t1 == t2):
                storage._user_R3[login] += 1
                result["R3"] = storage._user_R3[login]

                if ((storage._user_mes.get(data["login"])) is not None):
                    string = ""
                    for names in  storage._user_mes[data["login"]].keys():
                        if ( storage._user_mes[data["login"]][names].strip()):
                            string = string + names+": \n" +  storage._user_mes[data["login"]][names]+"\n"
                        storage._user_mes[data["login"]][names] = ""
                result["error"] = "OK"
                result["message"] = string
                return json.dumps(result)
            else:
                result["error"] = "password_error"
                return json.dumps(result)
        return json.dumps(result)
try:
    # Create a web server and define the handler to manage the
    # incoming request
    server = HTTPServer(('', PORT_NUMBER), myHandler)
    print('Started httpserver on port ', PORT_NUMBER)

    # Wait forever for incoming htto requests
    server.serve_forever()

except KeyboardInterrupt:
    print('^C received, shutting down the web server')
    server.socket.close()
