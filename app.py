from flask import Flask, render_template, request, session
import sys
from flask_sqlalchemy import SQLAlchemy
from config import Config
from flask_migrate import Migrate
from flask import jsonify
from models import db
import webview  # Import pywebviewf
import argparse
import atexit
import mimetypes

argparser = argparse.ArgumentParser()
argparser.add_argument("--view", action='store_true', help='Create pywebview window')
argparser.add_argument("--debug", action='store_true', help='Debug mode')
mimetypes.add_type('application/javascript', '.js')

args = argparser.parse_args()

migrate = Migrate()
def create_app(config_class=Config):
    app = Flask(__name__)
    # Load configuration
    app.config.from_object(config_class)
    # Initialize database
    db.init_app(app)
    migrate.init_app(app, db)
    # Import and register blueprints
    from routes.routes import routes
    from routes.capturer import capturer_api
    from routes.projects import projects, sync_json_with_db
    from routes.shaft import shaft_route
    from routes.imagematcher import imagematcher_route
    from routes.modelviewer import modelviewer_route
    from routes.nlights import nlights_route

    app.register_blueprint(routes)
    app.register_blueprint(projects)
    app.register_blueprint(capturer_api)
    app.register_blueprint(shaft_route)
    app.register_blueprint(imagematcher_route)
    app.register_blueprint(modelviewer_route)
    app.register_blueprint(nlights_route)
    # Create tables
    with app.app_context():
        db.create_all()
        sync_json_with_db()
    print("App created")
   
    #print all config 
    print("Config: ", app.config)
    return app

def run_flask():
    app = create_app()
    if args.view:
        if args.debug:
            app.run(port=app.config['APP_PORT'],debug=True, use_reloader=True )
        else:
            app.run(port=app.config['APP_PORT'],debug=False, use_reloader=False )
    else:
        app.run(port=app.config['APP_PORT'],debug=False)  # Use a specific port
def cleanup_on_exit():
    db.session.remove()
    global sub_server_process
    if sub_server_process and sub_server_process.poll() is None:
        sub_server_process.terminate()
        sub_server_process = None
    # db.drop_all()
if __name__ == '__main__':
    if not args.view:
        app = create_app()
        from api import Api
        api=Api()
        window = webview.create_window(
            'Integrator',
            app,
            width=535,
            height=800,
            min_size=(535,800),
            resizable=True,
            text_select=True,
            frameless=False,  # Keep standard frame
            # These minimize the frame appearance:
            transparent=False,
            on_top=False,
            confirm_close=False,  # Remove close confirmation,
            # Set API object
            js_api=api

        )
        api.set_window(window)
        if args.debug:
            webview.start(
                gui='edgechromium',  # or 'qt', 'cef', 'edgechromium'
                debug=True,  # Enable dev tools,
            )
        else:
            webview.start(
                gui='edgechromium',  # or 'qt', 'cef', 'edgechromium'
                debug=False,  # Enable dev tools,
            )
    else:
        run_flask()
    sys.exit(0)