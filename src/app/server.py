
import cherrypy
import pyspark as ps
from paste.translogger import TransLogger

from my_app import createMyApp

def run_server(app):
    app_logged = TransLogger(app)
    cherrypy.tree.graft(app_logged, '/')
    cherrypy.config.update({
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': 80,
        'server.socket_host': '0.0.0.0',
        'tools.sessions.on': True
        })
    cherrypy.engine.start()
    cherrypy.engine.block()

if __name__ == "__main__":
    m = 100
    k = 5
    sc = ps.SparkContext('local[8]')
    sc.addFile('../myLib/funcLib.py')
    sc.addFile('../myLib/engine.py')
    app = createMyApp(sc, m, k)
    run_server(app)
