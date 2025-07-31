import math
from flask import Blueprint, jsonify,render_template, session,current_app,request, redirect
from models import Project
import subprocess
import time
import os
import atexit

from utility.shaft.setup import setRawTherapeeOptions

worker_process = None 
routes = Blueprint('main', __name__)
sub_server_process = None   
def letter_to_number(letter):
    return ord(letter.lower()) - ord('a') + 1

def number_to_letter(number):
    if 1 <= number <= 26:
        return str(chr(ord('A') + number - 1))  # Use 'A' for uppercase letters
def cal_nlights_grid_folder(nextGrid=False):
    nlightsGridFolderOptions = [
    ]
    #count how many shaft outputs image we have
    nlightsCount=0
    for root, dirs, files in os.walk(session['shaft_output_path']):
        for file in files:
            if file.endswith(".tiff"):
                nlightsCount+=1
    # ONLY FOR TESTING
    nlightsCount=18

    nGrids=nlightsCount//9 
    nGrids=9
    p_width,p_height= session['width'],session['height']
    stepx, stepy= session['stepx'],session['stepy']
    shot_width, shot_height= 500, 375
    if nGrids>0:
        # calculate how many shots (column) we have in a row based on the painting width
        x_dimension= math.floor(p_width/stepx)*stepx+shot_width/2
        nGridX=x_dimension//shot_width
        nGridX=3
        # based nGridX calculate add A1.. A nGridX and for the remaining calculate B1.. B nGridX and so on till z
        for i in range(nGrids):
            row = int(i//nGridX)
            col = int(i%nGridX) 
            nlightsGridFolderOptions.append({
                "label": f"{number_to_letter(row+1)}{col+1}",
                "value": f"{number_to_letter(row+1)}{col+1}","url": "/session-var"
            })
    else:
        nlightsGridFolderOptions.append({
            "label": "A1", 
            "value": "A1","url": "/session-var"
        })
    if nextGrid:
        current = session['nlights_grid_folder']
        # find the next one after the current one
        for i in range(len(nlightsGridFolderOptions)):
            if i >= len(nlightsGridFolderOptions)-1:
                # return the last one
                session['nlights_grid_folder']= nlightsGridFolderOptions[i]['value']
                break
            if nlightsGridFolderOptions[i]['value'] == current:
                session['nlights_grid_folder'] = nlightsGridFolderOptions[i+1]['value']
                break
        
    gridLabel='A1' if session['nlights_grid_folder']=='' else  next((opt['label'] for opt in nlightsGridFolderOptions if opt['value'] == session['nlights_grid_folder']), "A1")
    return nlightsGridFolderOptions,gridLabel
outputOpts = [
    {"label": "2D", "value": "2D","url": "/session-var"},
    {"label": "3D", "value": "3D","url": "/session-var"},
    {"label": "Filigree", "value": "filigree","url": "/session-var"},
] 
vibrationOpts = [
    {"label": "Low (3.5)", "value": "low","url": "/session-var"},
    {"label": "High (3.8)", "value": "high","url": "/session-var"},
]
def new_session():
    session.clear()
    default_dict=Project.get_defaults()
    for key, value in default_dict.items():
        session[str(key)] = value
    session['session_id'] = time.time()
    session['is_shaft_developed']=False
    session['nlights_grid_folder']='A1'
def load_project_to_session(project_id):
    new_session()
    #overwrite session
    project = Project.query.get(project_id)
    if not project:
        return False
    project_data = Project.project_to_dict(project)
    for key, value in project_data.items():
        session[key] = value
    return True

def background_task():

    res=subprocess.run([current_app.config['PYTHON_PATH'],"webserver.py"], 
                            text=True,cwd="modules/capturer")
def cleanup_on_exit():
    global sub_server_process
    if sub_server_process and sub_server_process.poll() is None:
        sub_server_process.terminate()
        sub_server_process = None
atexit.register(cleanup_on_exit)
def calc_flow_mode():
    flow_mode=''
    if session['stand_type'] == 'vertical':
        flow_mode+='v'
    else:
        flow_mode+='h'
    if session['output_type'] == '2D':
        flow_mode+='2d'
    else:
        if session['output_type'] == 'filigree':
            flow_mode+='f'
        else:            
            flow_mode+='3d'
    if session['shot_mode'] == 'oneshot':
        flow_mode+='o'
    else:
        flow_mode+='m'
    return flow_mode

@routes.before_request
def before_request():
    global sub_server_process
    if sub_server_process and sub_server_process.poll() is None:
        sub_server_process.terminate()
        sub_server_process = None
    #create_dummy_projects()
    #check if session exists
    if 'session_id' not in session:
        #if not create a new session
        new_session()
    max_h=current_app.config['MAX_CONTENT_HEIGHT_MM']
    max_w=current_app.config['MAX_CONTENT_WIDTH_MM']
    session['shot_mode']='oneshot'
    if session['width'] > max_w or session['height'] > max_h:
        session['shot_mode']='multishot'
    session['flow_mode']=calc_flow_mode()
    # print(session['shot_mode'])
@routes.route('/')
def home():
    # Define sorting options
    sortOpts = [
        {"label": "A - Z","value": "alpha", "url": "/?sort=alpha","redirect": True},
        {"label": "Most value","value": "recent","url": "/?sort=recent","redirect": True},
        {"label": "Older","value": "older","url": "/?sort=older","redirect": True},
    ]
    # Get sort parameter from URL (default to 'recent' if not specified)
    sort_by = request.args.get('sort', default='Sort by', type=str)
    sort_label = next((opt['label'] for opt in sortOpts if opt['value'] == sort_by), "Sort by")
    # Query projects with appropriate sorting
    if sort_by == "alpha" or sort_by == "Sort by":
        projectsList = Project.query.order_by(Project.title.asc()).all()
    elif sort_by == "older":
        projectsList = Project.query.order_by(Project.created_at.asc()).all()
    else:  # default to 'recent'
        projectsList = Project.query.order_by(Project.created_at.desc()).all()
    projectsListDict=Project.project_list_to_dict(projectsList)
    return render_template(
        'index.html',
        title='Home',
        projectsList=projectsListDict,
        sortOpts=sortOpts,
        current_sort=sort_label  # Pass current sort option to template
    )

@routes.route('/projectpage/new')
def newproject():
    new_session()
    vibrationLabel='Sensitivity' if session['sensitivity']=='' else  next((opt['label'] for opt in vibrationOpts if opt['value'] == session['sensitivity']), "Sort by")
    return render_template('projectpage.html', title='projectpage', outputOpts=outputOpts,
                           vibrationOpts=vibrationOpts,vibrationLabel=vibrationLabel)
@routes.route('/projectpage')
def projectpage():
    # new_session()
    vibrationLabel='Sensitivity' if session['sensitivity']=='' else  next((opt['label'] for opt in vibrationOpts if opt['value'] == session['sensitivity']), "Sort by")
    return render_template('projectpage.html', title='projectpage', outputOpts=outputOpts,
                           vibrationOpts=vibrationOpts,vibrationLabel=vibrationLabel)
@routes.route('/projectpage/<project_id>')
def load_project(project_id):
    vibrationLabel='Sensitivity' if session['sensitivity']=='' else  next((opt['label'] for opt in vibrationOpts if opt['value'] == session['sensitivity']), "Sort by")

    res=load_project_to_session(project_id)
    return render_template('projectpage.html', title='projectpage', outputOpts=outputOpts,
                           vibrationOpts=vibrationOpts,vibrationLabel=vibrationLabel)
@routes.route('/first')
def firstModule():
    print(session['flow_mode'])
    if session['flow_mode'] == 'hfm':
        return redirect('/imagematcher')
    if session['flow_mode'] == 'hfo':
        return redirect('/projectpage')
    if session['stand_type'] == 'horizontal':
        return redirect('/shaft')   
    # redirect to /capturer
    return redirect('/capturer')
@routes.route('/nlights')
def nlights():
    nlightsGridFolderOptions, gridLabel=cal_nlights_grid_folder()
    return render_template('nlights.html', title='nlights',nlightsGridFolderOptions=nlightsGridFolderOptions, gridLabel=gridLabel)
@routes.route('/nlights/next')
def nlights_next():
    #clean the inputs
    session['nlights_all_lights_image_dir']=''
    session['nlights_direction_images_dir']=''
    nlightsGridFolderOptions, gridLabel=cal_nlights_grid_folder(nextGrid=True)
    return render_template('nlights.html', title='nlights',nlightsGridFolderOptions=nlightsGridFolderOptions, gridLabel=gridLabel)
@routes.route('/test')
def test():
    print(Project.get_defaults())   
    return render_template('test.html', title='test')
@routes.route('/imagematcher')
def imagematcher():
    return render_template('imagematcher.html', title='imagematcher')

@routes.route('/capturer')
def capturer(): 
    from routes.capturer import generate_capturer_config
    generate_capturer_config(current_app.config['CAPTURER_DIR'])
    global sub_server_process
    if not sub_server_process or sub_server_process.poll() is not None:
        sub_server_process = subprocess.Popen([current_app.config['PYTHON_PATH'],"webserver.py"], 
                    text=True,cwd=current_app.config['CAPTURER_DIR'])
    return render_template('capturer.html', title='capturer')

@routes.route('/shaft')
def shaft():
    outputColorOpts = [
        {"label": "sRGB", "value": "srgb","url": "/session-var"},
        {"label": "DisplayP3", "value": "display-p3","url": "/session-var"},
        {"label": "AdobeRGB", "value": "adobe-rgb","url": "/session-var"},
        {"label": "ProPhoto", "value": "pro-photo","url": "/session-var"},
    ] 
    outputFileOpts = [
        {"label": "JPEG", "value": "jpg","url": "/session-var"},
        {"label": "TIFF", "value": "tif","url": "/session-var"},
        {"label": "PNG", "value": "png","url": "/session-var"},
        {"label": "NEF", "value": "nef","url": "/session-var"}
    ] 
    config_path=os.path.join(current_app.config['SHAFT_DIR'],"config.ini")
    setRawTherapeeOptions(path=config_path, value=current_app.config['RAWTHERAPEE_PATH'])
    outputColorLabel='DisplayP3' if session['shaft_output_colorspace']=='' else  next((opt['label'] for opt in outputColorOpts if opt['value'] == session['shaft_output_colorspace']))
    processFormatLabel='TIFF' if session['shaft_process_format']=='' else  next((opt['label'] for opt in outputFileOpts if opt['value'] == session['shaft_process_format']))
    return render_template('shaft.html', title='shaft', outputColorOpts=outputColorOpts, outputFileOpts=outputFileOpts, outputColorLabel=outputColorLabel, processFormatLabel=processFormatLabel)


@routes.route('/modelviewer')
def modelviewer():
    modelviewerShotOpts = [
        {"label": "1 shot", "value": "one-shot.zip","url": "/session-var"},
        {"label": "Multiple shots", "value": "multiple-shots.zip","url": "/session-var"},
    ] 
    modelviewerShotLabel='Type' if session['modelviewer_template_zip']=='' else  next((opt['label'] for opt in modelviewerShotOpts if opt['value'] == session['modelviewer_template_zip']))
    return render_template('modelviewer.html', title='modelviewer',modelviewerShotOpts=modelviewerShotOpts,modelviewerShotLabel=modelviewerShotLabel)



@routes.route('/session-var', methods=['POST'])
def session_var():
    data = request.get_json()
    for key, value in data.items():
        session[key] = value
    return jsonify({
        'message': 'Session variables updated successfully',
        'session_data': dict(session)
    }), 200

