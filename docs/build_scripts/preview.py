#!/usr/bin/env python3
"""Serve the built documentation locally without retaining older browser copies."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


BUILD = Path(__file__).resolve().parents[1] / 'build'


class PreviewHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def send_head(self):
        # A previous preview may have cached / and /index.html independently.
        # Always send the current bytes, even for a conditional browser request.
        for header in ('If-Modified-Since', 'If-None-Match'):
            if header in self.headers:
                del self.headers[header]
        return super().send_head()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not (BUILD / 'index.html').is_file():
        parser.error('Build the documentation first; see docs/README.md.')
    handler = partial(PreviewHandler, directory=str(BUILD))
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'Preview: http://localhost:{args.port}/ (browser caching disabled)', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
