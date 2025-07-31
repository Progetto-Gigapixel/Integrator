
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect
from datetime import datetime
db = SQLAlchemy()
class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(20), unique=False,  default="TestTitle")
    artist = db.Column(db.String(20), unique=False, default="John Doe")
    location = db.Column(db.String(20), unique=False, default="Museo")
    address = db.Column(db.String(20), unique=False, default="via risorgimento 2, Bologna")
    width = db.Column(db.Integer, default=1000)
    height = db.Column(db.Integer, default= 1000)
    thumbnail = db.Column(db.String(100), default="test.jpg")
    created_at = db.Column(db.DateTime, default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    stand_type = db.Column(db.String(20), default="vertical")
    output_type = db.Column(db.String(20), default="3D")
    sensitivity = db.Column(db.Integer, default="low")
    vibration_check_time = db.Column(db.Integer, default=3)
    nshoty = db.Column(db.Integer, default=5)
    nshotx = db.Column(db.Integer, default=5)
    stepx = db.Column(db.Float, default=240)
    stepy = db.Column(db.Float, default=180)
    shot_mode = db.Column(db.String(20), default="multishot")
    # displacementx = db.Column(db.Float, default=0.1)
    # displacementy = db.Column(db.Float, default=0.1)
    #SHAFT
    shaft_colorchecker_path = db.Column(db.String(100), default="")
    shaft_flatfield_file_path = db.Column(db.String(100), default="")
    shaft_sharpen= db.Column(db.Boolean, default=False)
    shaft_light_balance= db.Column(db.Boolean, default=False)
    shaft_develop_folder_path = db.Column(db.String(100), default="")
    shaft_output_path = db.Column(db.String(100), default="")
    shaft_process_subfolders = db.Column(db.Boolean, default=False)
    shaft_overwrite = db.Column(db.Boolean, default=False)
    shaft_process_format = db.Column(db.String(20), default="tif")
    shaft_output_colorspace = db.Column(db.String(20), default="display-p3")
    shaft_savein_path= db.Column(db.String(100), default="")
    
    # shaft_output_path = db.Column(db.String(100), default="")
    #NLIGHTS
    nlights_all_lights_image_dir=db.Column(db.String(100), default="")
    nlights_direction_images_dir=db.Column(db.String(100), default="")
    nlights_output_directory= db.Column(db.String(100), default="")
    # optionals
    # nlights_height_map_format=db.Column(db.String(20), default="PLY")
    # nlights_light_order=db.Column(db.String(100), default="")
    # nlights_decimation_surface=db.Column(db.Integer, default=16)
    # nlights_poly_correction=db.Column(db.Boolean, default=True)
    #IMAGE MATCHER
    imagematcher_maps_path=db.Column(db.String(100), default="")
    imagematcher_tex_out_path=db.Column(db.String(100), default="")

    #MODELVIEWER 
    modelviewer_artwork_dir_path=db.Column(db.String(100), default="")
    modelviewer_template_dir_path=db.Column(db.String(100), default="") 
    modelviewer_output_path=db.Column(db.String(100), default="")
    modelviewer_template_zip=db.Column(db.String(20), default="")
    def __repr__(self):
        return '<Project %r>' % self.title
    def project_to_dict(project):
        """Convert Project instance to a serializable dictionary"""
        if not project:
            return None
     
        return {
            'id': project.id,
            'title': project.title,
            'artist': project.artist,
            'location': project.location,
            'address': project.address,
            'width': project.width,
            'height': project.height,
            'thumbnail': project.thumbnail,
            'created_at': project.created_at.isoformat() if project.created_at else None,
            'stand_type': project.stand_type,
            'output_type': project.output_type,
            'sensitivity': project.sensitivity,
            'vibration_check_time': project.vibration_check_time,
            'nshoty': project.nshoty,
            'nshotx': project.nshotx,
            'stepx': float(project.stepx),
            'stepy': float(project.stepy),
            'shot_mode': project.shot_mode,
            # 'displacementx': float(project.displacementx) if project.displacementx is not None else None,
            # 'displacementy': float(project.displacementy) if project.displacementy is not None else None,
            'shaft_colorchecker_path': project.shaft_colorchecker_path,
            'shaft_flatfield_file_path': project.shaft_flatfield_file_path,
            'shaft_sharpen': bool(project.shaft_sharpen),
            'shaft_light_balance': bool(project.shaft_light_balance),
            'shaft_develop_folder_path': project.shaft_develop_folder_path,
            'shaft_output_path': project.shaft_output_path,
            'shaft_process_subfolders': bool(project.shaft_process_subfolders),
            'shaft_overwrite': bool(project.shaft_overwrite),
            'shaft_process_format': project.shaft_process_format,
            'shaft_output_colorspace': project.shaft_output_colorspace,
            'shaft_savein_path': project.shaft_savein_path,
            #model viewer7
            'nlights_all_lights_image_dir': project.nlights_all_lights_image_dir,
            'nlights_direction_images_dir': project.nlights_direction_images_dir,
            'nlights_output_directory': project.nlights_output_directory,
            # 'nlights_height_map_format': project.nlights_height_map_format,
            # 'nlights_light_order': project.nlights_light_order,
            # 'nlights_decimation_surface': project.nlights_decimation_surface,
            # 'nlights_poly_correction': project.nlights_poly_correction,
            # IMAGE MATCHER
            'imagematcher_maps_path' : project.imagematcher_maps_path,
            'imagematcher_tex_out_path' : project.imagematcher_tex_out_path,
            #model viewer
            'modelviewer_template_zip': project.modelviewer_template_zip,
            'modelviewer_artwork_dir_path': project.modelviewer_artwork_dir_path,
            'modelviewer_template_dir_path': project.modelviewer_template_dir_path,
            'modelviewer_output_path': project.modelviewer_output_path
    }
    def project_list_to_dict(projects):
            return [Project.project_to_dict(project) for project in projects]
    def get_defaults():
        """Returns a dict of default values by inspecting the Project model."""
        inspector = inspect(Project)
        defaults = {}
        for column in inspector.columns:
            if column.default is None:
                continue  # Skip columns without defaults
            default = column.default

            # Handle SQLAlchemy's context-sensitive defaults (like datetime.now)
            if hasattr(default, 'arg'):
                # Case 1: Callable default (e.g., datetime.now)
                if callable(default.arg):
                    try:
                        defaults[column.name] = default.arg()  # Try calling it directly
                    except TypeError:
                        # If it fails (due to missing SQLAlchemy context), return None or a fallback
                        defaults[column.name] = datetime.now() # Fallback to current time
                # Case 2: Static default (e.g., "TestTitle", 0, False)
                else:
                    defaults[column.name] = default.arg
            # Handle direct Python defaults (less common in SQLAlchemy)
            elif callable(default):
                defaults[column.name] = default()
            else:
                defaults[column.name] = default

        return defaults