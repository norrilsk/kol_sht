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
from Crypto.Random import random
from utils import fast_power
import json

PORT_NUMBER = 8080


# This class will handles any incoming request from
# the browser
class myHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):

        self._prime = number.getPrime(256)
        self._p = number.getPrime(256)
        self._g = random.randint(2,self._p)
        self._B = fast_power(self._g, self._prime,mod = self._p)
        self._user_K = {}
        self._user_pas = {}
        self._user_mes = {}
        super().__init__(request, client_address, server)
    # Handler for the GET requests
    def do_GET(self):
        self.send_response(200)
        message = "hello world"
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
            result["p"] = self._p
            result["g"] = self._g
            result["B"] = self._B
            return json.dumps(result)
        elif (data["type"] == "register"):
            if (self._user_K.get(data["login"])) is None:
                self._user_K[data["login"]] = fast_power(data["A"],self._prime,mod=self._p)
                self._user_pas[data["login"]] = data["password"]^self._user
                self._user_mes[data["login"]]  = {}
                result["error"] = "OK"
                return json.dumps(result)
            else:
                result["error"] = "accupied"
                return json.dumps(result)

        elif (data["type"] == "send"):
            if (self._user_pass.get(data["login"]) == data["password"]):
                if ((self._user_K.get(data["user"])) is None):
                    result["error"] = "NOUSER"
                    return json.dumps(result)
                else:
                    if ((self._user_mes[data["login"]].get(data["user"])) is not None):
                        self._user_mes[data["login"]][data["user"]] += "\n" + str(data["message"])
                    else:
                        self._user_mes[data["login"]][data["user"]] = str(data["message"])
                    result["error"] = "OK"
                    return json.dumps(result)
            else:
                result["error"] = "password_error"
                return json.dumps(result)
        elif (data["type"] == "recv"):
            if ((self._user_K.get(data["login"])) is not None) and (
                    self._user_pass.get(data["login"]) == data["password"]):
                if ((self._user_mes[data["login"]].get(data["user"])) is not None):
                    result["message"] =   self._user_mes[data["login"]][data["user"]]
                else:
                    result["message"] = ""
                result["error"] = "OK"
                return json.dumps(result)
            else:
                result["error"] = "password_error"
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
