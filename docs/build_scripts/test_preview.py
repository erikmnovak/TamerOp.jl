"""Exercise real HTTP request handling in memory, without opening a port."""
from http.client import HTTPResponse
import io
from pathlib import Path
import tempfile
import unittest

from preview import PreviewHandler


class Request:
    def __init__(self, data):
        self.input = io.BytesIO(data)
        self.output = io.BytesIO()

    def makefile(self, *args, **kwargs):
        return self.input

    def sendall(self, data):
        self.output.write(data)


class QuietHandler(PreviewHandler):
    def log_message(self, *args):
        pass


class PreviewTests(unittest.TestCase):
    def test_home_routes_and_conditional_requests_return_current_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / 'index.html'
            home.write_text('<h1>Old introduction</h1>')

            def request(path, method='GET', headers=''):
                incoming = Request(f'{method} {path} HTTP/1.1\r\nHost: localhost\r\n'
                                   f'{headers}\r\n'.encode())
                QuietHandler(incoming, ('127.0.0.1', 1234), None, directory=directory)
                response = HTTPResponse(Request(incoming.output.getvalue()), method=method)
                response.begin()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader('Cache-Control'), 'no-store')
                return response.read()

            request('/index.html')
            home.write_text('<h1>New introduction</h1>')
            conditional = 'If-Modified-Since: Wed, 31 Dec 2098 23:59:59 GMT\r\n'
            for route in ('/', '/index.html', '/index.html?review=latest'):
                with self.subTest(route=route):
                    self.assertEqual(request(route, headers=conditional), home.read_bytes())
            self.assertEqual(request('/index.html', method='HEAD', headers=conditional), b'')
