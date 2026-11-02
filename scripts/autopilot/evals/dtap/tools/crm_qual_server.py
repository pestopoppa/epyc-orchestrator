#!/usr/bin/env python3
"""Deterministic local transport for CRM full-entry fixture qualification."""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if body.get('seed') != 42:
                raise ValueError('seed mismatch')
            if body['messages'][-1]['role'] == 'tool':
                message = {'role': 'assistant', 'content': 'fixture complete'}
                finish = 'stop'
            else:
                call = {'name': 'create_record', 'arguments': json.dumps({
                    'module_name': 'Notes', 'attributes': {'name': 'Daily Activity Summary - Meridian Group',
                    'description': 'customer engagement tracked and logged'}})}
                message = {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': 'fixture-1', 'type': 'function', 'function': call}]}
                finish = 'tool_calls'
            data = json.dumps({'choices': [{'message': message, 'finish_reason': finish}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self.send_error(400)


HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
