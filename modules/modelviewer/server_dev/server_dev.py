from http.server import HTTPServer, SimpleHTTPRequestHandler
import ssl

httpd = HTTPServer(("0.0.0.0", 1443), SimpleHTTPRequestHandler)
sslctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
sslctx.check_hostname = False
sslctx.load_cert_chain(certfile="server_dev/cert.pem", keyfile="server_dev/key.pem")
httpd.socket = sslctx.wrap_socket(httpd.socket, server_side=True)
httpd.serve_forever()
