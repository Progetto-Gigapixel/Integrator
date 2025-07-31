from flask import Blueprint, jsonify,request, session
from flask import request
from app import db
from models import Project 
import os
from datetime import datetime
import json
projects = Blueprint('project', __name__, url_prefix='/api')
PROJECTS_DIR = os.path.join("./static", 'projects')
os.makedirs(PROJECTS_DIR, exist_ok=True)
def extract_model_fields_names(model):
    """Extract all fields from a SQLAlchemy model with their properties"""
    fields_names = []
    
    for column in model.__table__.columns:
        field_info = {
            'type': str(column.type),
            'nullable': column.nullable,
            'default': column.default.arg if column.default else None,
            'primary_key': column.primary_key,
            'unique': column.unique
        }
        fields_names.append(column.name)
    return fields_names

def project_from_json(json_data):
        # Create new from json
    project_data = {}
    for key, value in json_data.items():
        if key in extract_model_fields_names(Project):
            project_data[key] = value
    if 'created_at' in project_data:
        project_data['created_at'] = datetime.strptime(project_data['created_at'], "%Y-%m-%d %H:%M:%S")
    new_project = Project(**project_data)
    return new_project
def process_json_file(filename, json_dir, db_projects):
    """
    Process a single JSON file and synchronize with database records
    Returns tuple: (changes_made, error_message)
    """
    try:
        filepath = os.path.join(json_dir, filename)
        with open(filepath, 'r') as f:
            json_data = json.load(f)
            
        project_id = json_data.get('id')
        changes_made = False
        
        # Case 1: JSON exists but no matching DB record (create in DB)
        if project_id not in db_projects:
            new_project = project_from_json(json_data)
            db.session.add(new_project)
            print(f"Created missing project from JSON: {project_id}")
            return (True, None)
        
        # Case 2: JSON and DB exist (check if needs update)
        else:
            print(f"Processing project from JSON: {project_id}")
            db_project = db_projects[project_id]
            needs_update = False
            
            # Check each field for changes
            Project_fields = extract_model_fields_names(Project)
            fields_to_check = [field for field in Project_fields if field != "id"]
            for field in fields_to_check:
                # print(f"Checking field {field} in project from JSON: {project_id}")
                # print(f"DB value: {getattr(db_project, field)}")
                # print(f"JSON value: {json_data.get(field)}")
                if getattr(db_project, field) != json_data.get(field):
                    setattr(db_project, field, json_data.get(field))
                    print(f"Updated field {field} in project from JSON: {project_id}")
                    needs_update = True
            
            if needs_update:
                db_project.created_at = datetime.strptime(db_project.created_at, "%Y-%m-%d %H:%M:%S")
                print(f"Updated project from JSON: {project_id}")
                return (True, None)
            
            return (False, None)
            
    except json.JSONDecodeError as e:
        return (False, f"Invalid JSON in {filename}: {str(e)}")
    except KeyError as e:
        return (False, f"Missing required field in {filename}: {str(e)}")
    except Exception as e:
        return (False, f"Error processing {filename}: {str(e)}")

def save_project_to_json(project):
    """Save project data to a JSON file"""
    try:
        project_data=project
        filename = f"project_{project['id']}.json"
        filepath = os.path.join(PROJECTS_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(project_data, f, indent=4)
    except Exception as e:
        return (False, f"Error saving project to JSON: {str(e)}")
    
    return filepath
def sync_json_with_db():
    """Synchronize JSON files in directory with database records"""
    json_dir = PROJECTS_DIR
    changes_made = False
    
    # 1. Get all existing projects from DB
    db_projects = {p.id: p for p in Project.query.all()}
    
    # 2. Get all JSON files in directory
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # 3. Process each JSON file
    for filename in json_files:
        print(f"Processing JSON file: {filename}")
        file_changes, error = process_json_file(filename, json_dir, db_projects)
        if error:
            print(error)
        changes_made = changes_made or file_changes
    
    # 4. Check for DB records with no JSON file
    db_ids = set(db_projects.keys())
    json_ids = {int(f.split('_')[1].split('.')[0]) for f in json_files if f.startswith('project_')}
    
    # # Case 3: Create missing JSON files
    # for missing_id in db_ids - json_ids:
    #     project = db_projects[missing_id]
    #     save_project_to_json(project)
    #     print(f"Created missing JSON for project: {missing_id}")
    #     changes_made = True
    
    if changes_made:
        db.session.commit()
    return changes_made



@projects.route('/projects', methods=['POST'])
def create_project():
    data = request.get_json()

    # If title is provided in the request, check for existing project
    if 'title' in data:
        existing_project = Project.query.filter_by(title=data['title']).first()
        if existing_project:
            return jsonify({
                'error': f'Project with title {data["title"]} already exists',
                'existing_project': {
                    'id': existing_project.id,
                    'title': existing_project.title,
                    'artist': existing_project.artist
                }
            }), 409  # HTTP 409 Conflict

    try:
        new_project = project_from_json(data)
        new_project.created_at = datetime.strptime(new_project.created_at, "%Y-%m-%d %H:%M:%S")
        db.session.add(new_project)
        db.session.commit()
        print(f"Created project: {new_project}")
        # Save to JSON file after commit (so we have the ID)
        
        json_path = save_project_to_json(Project.project_to_dict(new_project))
        
        return jsonify({
            'message': 'Project created successfully',
            'id': new_project.id,
            'json_path': json_path
        }), 201
        
    except Exception as e:
        print(e)
        db.session.rollback()
        return jsonify({
            'error': 'Failed to create project',
            'details': str(e)
        }), 500
@projects.route('/projects', methods=['GET'])
def get_projects():
    projects = Project.query.all()
    result = Project.project_list_to_dict(projects)
    return jsonify(result), 200

@projects.route('/projects/<int:id>', methods=['GET'])
def get_project(id):
    project = Project.query.get_or_404(id)
    if not project:
        return jsonify({'message': 'Project not found'}), 404
    return Project.project_to_dict(project), 200

@projects.route('/projects/<int:id>', methods=['PUT'])
def update_project(id):
    project = Project.query.get_or_404(id)
    if not project:
        return jsonify({'message': 'Project not found'}), 404
    data = request.get_json()

    project.title = data.get('title', project.title)
    project.artist = data.get('artist', project.artist)
    project.width = data.get('width', project.width)
    project.height = data.get('height', project.height)
    project.thumbnail = data.get('thumbnail', project.thumbnail)

    db.session.commit()
    # Save to JSON file after commit (so we have the ID)
    json_path = save_project_to_json(project)
    return jsonify({'message': 'Project updated'}), 200

@projects.route('/projects/<int:id>', methods=['DELETE'])
def delete_project(id):
    project = Project.query.get_or_404(id)
    if not project:
        return jsonify({'message': 'Project not found'}), 404
    db.session.delete(project)
    db.session.commit()
    # Delete JSON file if exists
    filename = f"project_{project.id}.json"
    filepath = os.path.join(PROJECTS_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    return jsonify({'message': 'Project deleted'}), 200
@projects.route('/projects/sync', methods=['POST'])
def sync_projects():
    try:
        if sync_json_with_db():
            return jsonify({'message': 'Synchronization completed with changes'}), 200
        return jsonify({'message': 'Already in sync - no changes made'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@projects.route('/save_session', methods=['POST'])
def save_session():
    try:
        # Get the path from the request
        data = request.get_json()
        if not data or 'path' not in data:
            return jsonify({'error': 'Path parameter is required'}), 400
        path = data['path']
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert session to dict and save to JSON
        session_dict = dict(session)
        # convert date objects to strings
        if 'created_at' in session_dict:
            if isinstance(session_dict['created_at'], datetime):
                session_dict['created_at'] = session_dict['created_at'].strftime('%Y-%m-%d %H:%M:%S')

        with open(path, 'w') as f:
            json.dump(session_dict, f, indent=4)
        new_project = project_from_json(session_dict)
        db.session.add(new_project)
        db.session.commit()
        return jsonify({
            'message': 'Session saved successfully',
            'path': path,
            'session_data': session_dict
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to save session',
            'details': str(e)
        }), 500
    
@projects.route('/load_session', methods=['POST'])
def load_session():
    try:
        # Get the path from the request
        data = request.get_json()
        session_data = data
        # Clear existing session and load new data
        session.clear()
        for key, value in session_data.items():
            session[key] = value
        
        return jsonify({
            'message': 'Session loaded successfully',
            'loaded_data': session_data
        }), 200
        
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON file'}), 400
    except Exception as e:
        return jsonify({
            'error': 'Failed to load session',
            'details': str(e)
        }), 500