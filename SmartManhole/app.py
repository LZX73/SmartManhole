from flask import Flask, render_template, jsonify, send_file, Response, request, flash, session, redirect, url_for, session
from flask_session import Session
import cv2
import numpy as np
import urllib.request
import urllib.error
import requests
from datetime import datetime, timedelta
import base64
import io
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app, storage, db, auth
import os
import tempfile
from config import Config
from pyrebase import pyrebase
from flask_mail import Mail, Message
from firebase_admin import storage as firebase_storage
from datetime import datetime, timedelta, timezone
from google.cloud.firestore_v1 import GeoPoint
from opencage.geocoder import OpenCageGeocode
from flask_wtf import FlaskForm
from wtforms import SubmitField
from wtforms.validators import DataRequired
import json

app = Flask(__name__)
# Define the MaintenanceForm
class MaintenanceForm(FlaskForm):
    submit = SubmitField('Check')



# Initialize Firebase using the service account key
cred = credentials.Certificate(
    r'C:\Users\user\Documents\Year3S2\idp\SmartManhole\smart-manhole-f5530-firebase-adminsdk-rl149-f2d3b9b900.json')
# Initialize Firebase with Realtime Database and Storage
firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://smart-manhole-f5530-default-rtdb.firebaseio.com/',
        'storageBucket': 'smart-manhole-f5530.appspot.com'
})
bucket = storage.bucket()

detected_objects = []  # Initialize the detected_objects list

db_ref = db.reference('/data/5PdD2BuswlU0hxkvdQ94EbgKoKz1')
# Reference the Realtime Database
dbparameter_ref = db.reference('/parameter/callibration')
dbparameter = db.reference('/parameter')

app.config.from_object(Config)
app.secret_key = 'your_secret_key'
OPENCAGE_API_KEY = '39f5c53a693d4cd0969acebaf3b49fab'

# Define parametes for Object Detection
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile = 'coco.names'
esp_ip = '192.168.116.28'  # Replace with your ESP32's IP address

#192.168.116.28
# Read class names from file
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load YOLO model
modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Firebase configuration
firebaseConfig = {
    'apiKey': "AIzaSyC4tzOcu-ADuJxhwyiw0cPYLLdjSqTvKz4",
    'authDomain': "smart-manhole-f5530.firebaseapp.com",
    'databaseURL': "https://smart-manhole-f5530-default-rtdb.firebaseio.com",
    'projectId': "smart-manhole-f5530",
    'storageBucket': "smart-manhole-f5530.appspot.com",
    'messagingSenderId': "561649259899",
    'appId': "1:561649259899:web:cfce85efc06cdb9d4759de",
    'measurementId': "G-H3H55QGNCD"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

# Initialize Firebase using credentials.json

firebase_admin.initialize_app(cred, {'databaseURL': 'https://smart-manhole-f5530-default-rtdb.firebaseio.com/'},
                              name='first_app')
firestore_db = firestore.client()

# Route for Authentication
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user_email'] = email
            session['password'] = password
            session['user_id_token'] = user['idToken']

            # Update the MAIL_DEFAULT_SENDER with the login user's email
            app.config['MAIL_DEFAULT_SENDER'] = email

            return redirect(url_for('sensor'))

        except Exception as e:
            print(f'Error: {e}')
            flash('Invalid email or password. Please try again', 'error')

    return render_template('login.html')

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('user_email', None)
    flash('Logged out successfully', 'success')
    return redirect(url_for('home'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            auth.create_user_with_email_and_password(email, password)

            # Automatically log in the user after successful registration
            user = auth.sign_in_with_email_and_password(email, password)
            session['user_email'] = email

            flash('User created successfully! Please log in.', 'success')
            return redirect(url_for('monitoring'))
        except Exception as e:
            flash(f'Error: {e}')

    return render_template('signup.html')

@app.route('/report_dashboard', methods=['GET', 'POST'])
def report_dashboard():
    user_email = session.get('user_email')

    if not user_email:
        flash('You are not logged in. Please log in to access the dashboard.', 'error')
        return redirect(url_for('login'))

    # Fetch data from Firestore Manhole collection
    manhole_data = []

    # Get the search query from the request
    search_query = request.args.get('search', '').strip()

    # Retrieve data based on search query
    if search_query:
        collection_ref = firestore_db.collection('Manhole report')
        manhole_docs = collection_ref.where('Remarks', '>=', search_query).where('Remarks', '<=', search_query + '\uf8ff').stream()
    else:
        # Retrieve all data if no search query is provided
        collection_ref = firestore_db.collection('Manhole report')
        manhole_docs = collection_ref.stream()

    for doc in manhole_docs:
        cover_status = doc.get('Cover_Status')
        remarks = doc.get('Remarks')
        timestamp = doc.get('timestamp')
        report_id = doc.id

        # Extract GeoPoint information
        location = doc.get('Location')  # Assuming 'location' is the field name
        # Extract latitude and longitude from GeoPoint
        latitude = location.latitude if location else None
        longitude = location.longitude if location else None

        # Get the city name using the OpenCage Geocoding API
        city_name = get_city_name(latitude, longitude)

        image_data = get_all_image_urls()

        # Add extracted data to the list
        manhole_data.append({
            'id': report_id,
            'Cover_Status': str(cover_status) if cover_status else '',
            'Remarks': remarks,
            'Timestamp': timestamp,
            'Latitude': latitude,
            'Longitude': longitude,
            'Location': city_name,  # Replace with the city name

        })

    return render_template('firebase.html', data=manhole_data, search_query=search_query, image_data=image_data)

def get_all_image_urls():
    try:
        # Get a reference to the bucket
        bucket = firebase_storage.bucket()

        # List all objects in the "maintenance" folder
        blobs = bucket.list_blobs(prefix='maintenance/')

        # Initialize a list to store image URLs
        image_urls = []

        # Get the current time in UTC
        current_time_utc = datetime.now(timezone.utc)

        # Generate signed URLs for each image
        for blob in blobs:
            # Check if the blob is an image file
            if blob.name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Generate the signed URL with an expiration time of 1 hour
                expiration_time = current_time_utc + timedelta(hours=1)
                image_url = blob.generate_signed_url(expiration=expiration_time, method='GET')
                image_urls.append(image_url)

        return image_urls

    except Exception as e:
        print(f"Error getting image URLs: {e}")
        return []

def get_city_name(latitude, longitude):
    ocg = OpenCageGeocode('39f5c53a693d4cd0969acebaf3b49fab')
    # Make a request to the OpenCage Geocoding API
    results = ocg.reverse_geocode(latitude, longitude)

    # Extract the city from the API response
    if results and results[0] and 'formatted' in results[0]:
        city = results[0]['formatted']
        return city
    else:
        return 'N/A'

@app.route('/maintenance/<manhole_id>', methods=['GET', 'POST'])
def maintenance_page(manhole_id):

    if request.method == 'POST':
        manhole_data = {key.replace("manhole_data_", ""): value for key, value in request.form.items() if
                        "manhole_data_" in key}

        if not session.get('selected_manhole'):
            session['selected_manhole'] = manhole_data

        form = MaintenanceForm()
        selected_date = None

    if form.validate_on_submit():

        selected_date = request.form['selected_date']
        manhole_d = session.get('selected_manhole')
        print("get: ")
        print(manhole_d)

        # Check if the "Send" or "Send Selected Documents" button is clicked
        if 'send_email' in request.form:
            print("email: ")
            # selected_documents = request.form.getlist('selected_documents')
            deadline = request.form['deadline']  # Get the deadline value


            # Proceed with sending email for selected documents and deadline
            result = send_selected_documents_email(manhole_d, selected_date, deadline)


            # Handle the result as needed (e.g., show a success or error message)
            print(result)

    return render_template('maintenance_page.html', manhole_id=manhole_id, manhole_data=manhole_data, form=form, selected_date=selected_date )

# Define the function to send email for selected documents and deadline
def send_selected_documents_email(manhole_data, selected_date,  deadline):
    try:
        # if selected_date and manhole_data :

        if selected_date:
            recipient_email = 'lxr2002@gmail.com'  # Replace with the recipient's email address
            print("yes")


            selected_data_json = json.dumps(manhole_data)
            print("json :")
            print(selected_data_json)
            # Send the email with selected data and deadline
            subject = f'Selected Maintenance Data for {selected_date}'
            body_html = render_template('email_template.html',  deadline=deadline, manhole_data=selected_data_json)

            msg = Message(subject,
                          recipients=[recipient_email],
                          body=body_html,
                          sender=('Your Organization', 'noreply@example.com'),
                          reply_to='noreply@example.com',
                          extra_headers={'Precedence': 'bulk'},
                          html=body_html)
            mail.send(msg)
            if 'selected_manhole' in session:
                # del session['selected_manhole']
                session.pop('selected_manhole', None)

            return 'Email sent successfully!'

        else:
            return 'No data available to send.'
    except Exception as e:

        return f'Error: {str(e)}'




@app.route('/sensor', methods=['GET', 'POST'])
def sensor():
    user_email = session.get('user_email')
    
    if not user_email:
        flash('You are not logged in. Please log in to access the dashboard.', 'error')
        return redirect(url_for('login'))

    try:
        # Retrieve sensor data from the database
        sensor_data = db_ref.get()
        parameter_data = dbparameter.get()
        
        if sensor_data:
            sensor_list = []
            for key, value in sensor_data.items():
                # Check if the data format matches the desired format
                if 'timestamp' in value and 'Methane' in value and 'Pitch_angle' in value and 'Roll_angle' in value and 'Water_level' in value and 'Battery_level' in value :
                    # Extract sensor data fields
                    battery_level = value.get('Battery_level', '')
                    methane = value.get('Methane', '')
                    tilting_angle_x = value.get('Pitch_angle', '')
                    tilting_angle_y = value.get('Roll_angle', '')
                    temperature = value.get('Temperature', '')
                    water_level = value.get('Water_level', '')
                    timestamp = value.get('timestamp', '')

                    # Parse timestamp to datetime object
                    timestamp = datetime.strptime(timestamp, "%d %b %Y %I:%M%p")

                    # Append sensor data to the list
                    sensor_list.append({
                        'battery_level': battery_level,
                        'methane': methane,
                        'Pitch_angle': tilting_angle_x,
                        'Roll_angle': tilting_angle_y,
                        'temperature': temperature,
                        'water_level': water_level,
                        'timestamp': timestamp,
                    })
                    
                    
        sorted_sensor = sorted(sensor_list, key=lambda x: x['timestamp'])
        
        if parameter_data:
            
            parameter_list = []
            for key, value in parameter_data.items():
                if 'water_threshold' in value and 'gas_threshold' in value:
                    
                    water_threshold = value.get('water_threshold', '')
                    gas_threshold = value.get('gas_threshold', '')
                    
                    parameter_list.append({
                        'water_threshold': water_threshold,
                        'gas_threshold': gas_threshold
                    })
                
                print(parameter_list)
                

            # Render the template with sensor data
            return render_template('sensor.html', data=sorted_sensor, parameter_data_list = parameter_list)
        else:
            # If no sensor data available, render template with empty data
            return render_template('sensor.html', data=[])
    except Exception as e:
        # Handle any exceptions (including authentication errors)
        return f'Error: {str(e)}'
    
@app.route('/add_sensor_data', methods=['POST'])
def add_sensor_data():
    if request.method == 'POST':
        # Assuming the form data contains the sensor data fields
        # You can adjust this according to your actual form structure
        sensor_data = {
            'Battery_level': request.form['Battery_level'],
            'Methane': request.form['Methane'],
            'Pitch_angle': request.form['Pitch_angle'],
            'Roll_angle': request.form['Roll_angle'],
            'Temperature': request.form['Temperature'],
            'Water_level': request.form['Water_level'],
            'timestamp': datetime.now().strftime("%d %b %Y %I:%M%p")
        }
        
        # Push the new sensor data to Firebase
        db_ref.push(sensor_data)
        flash('Sensor data added successfully.', 'success')
        
        # Redirect back to the sensor page
        return redirect(url_for('sensor'))
    

@app.route('/delete_manhole/<zanhole_id>', methods=['POST'])
def delete_manhole(manhole_id):
    try:
        firestore_db.collection('Manhole report').document(manhole_id).delete()

        # Respond with a success message
        return jsonify({'success': True, 'message': 'Data deleted successfully'})
    except Exception as e:
        # Respond with an error message if deletion fails
        return jsonify({'success': False, 'error': str(e)})

@app.route('/Settings')
def Settings():
    try:
        # Get the latest data
        latest_data = dbparameter_ref.get()

        print(f"Type of latest_data: {type(latest_data)}")
        print(f"Content of latest_data: {latest_data}")

        if isinstance(latest_data, dict):
            # Access 'Manhole_depth' directly since 'latest_data' already contains the 'parameter' key
            manhole_depth = latest_data.get('Manhole_depth', '')
            water_threshold = latest_data.get('water_threshold', '')
            gas_threshold = latest_data.get('gas_threshold', '')
            Blower_duration = latest_data.get('Blower_duration', '')
            Waterjet_duration = latest_data.get('Waterjet_duration', '')
            sampling_time = latest_data.get('sampling_time', '')
            inlet_angle = latest_data.get('inlet_angle', '')
            outlet_angle = latest_data.get('outlet_angle', '')
            
            # Process the data and pass it to the template
            latest_data_list = [{'Manhole_depth': manhole_depth,'water_threshold':water_threshold,"gas_threshold":gas_threshold,"Blower_duration":Blower_duration,"Waterjet_duration":Waterjet_duration,"sampling_time":sampling_time,"inlet_angle":inlet_angle,"outlet_angle":outlet_angle }]

            print(latest_data_list)
            return render_template('Settings.html', data=latest_data_list)
        else:
            # Handle the case when 'latest_data' is not a dictionary
            return jsonify({'message': 'Invalid data format or no data found.'})

    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'})


@app.route('/update_parameter', methods=['POST'])
def update_parameter():
    try:
        # Get data from the request
        data = request.get_json()

        # Reference the Realtime Database
        db_ref = firebase_admin.db.reference('/parameter/callibration')

        parameter = data.get('parameter')
        new_value = data.get('value')
        # Update data in the Realtime Database
        db_ref.update({
            parameter: new_value
            
        })

        return jsonify({'success': True, 'message': 'Parameter updated successfully.'}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': 'Parameter update failed.'}), 400


# ESP 32 Cam
def get_angles_from_esp(esp_ip):  # get servo angle
    # Construct the URL for retrieving pan and tilt angles
    angles_url = f'http://{esp_ip}/get-angles'

    # Send GET request to retrieve angles
    response = requests.get(angles_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse response to get pan and tilt angles
        pan_angle, tilt_angle = map(int, response.text.split(','))
        return pan_angle, tilt_angle
    else:
        print(f"Error: Failed to retrieve angles. Status code: {response.status_code}")
        return None, None


def send_control_command(pan_angle, tilt_angle, spotlight_state, waterjet_state, blower_state,scan_state):
    # Control servo angle, spotlight state, and waterjet state
    control_url = f'http://{esp_ip}/control_servos?pan={pan_angle}&tilt={tilt_angle}&spotlight={spotlight_state}&waterjet={waterjet_state}&blower={blower_state}&scan={scan_state}'
    requests.get(control_url)


# Add function to handle POST request for updating angles
@app.route('/update_angles', methods=['POST'])
def update_angles():
    tilt_angle = int(request.form['tilt_angle'])
    pan_angle = int(request.form['pan_angle'])
    spotlight_state = request.form['spotlight_state']  # assuming this will be a string 'on' or 'off'
    quality = request.form['quality']

    if 'waterjet_state' in request.form:  # condition no use, set default using timeout in html
        waterjet_state = request.form['waterjet_state']
        print("waterjet state:")
        print(waterjet_state)
    else:
        waterjet_state = 0  # Default to 'off' if not provided

    if 'blower_state' in request.form:  # condition no use, set default using timeout in html
        blower_state = request.form['blower_state']
        print("blower state:")
        print(blower_state)
    else:
        blower_state = 0  # Default to 'off' if not provided
        
    if 'scan_state' in request.form:  # condition no use, set default using timeout in html
        scan_state = request.form['scan_state']
        print("scan state:")
        print(scan_state)
    else:
        scan_state = 0  # Default to 'off' if not provided

    # waterjet_state = request.form['waterjet_state']  # Default to 'off' if not provided

    send_control_command(pan_angle, tilt_angle, spotlight_state, waterjet_state, blower_state,scan_state)
    return 'OK'


# Function to find objects in the image and draw bounding boxes
def findObject(outputs, frame):
    hT, wT, cT = frame.shape
    bbox = []
    classIds = []
    confs = []
    found_cell_phone = False

    # Iterate over outputs to extract bounding boxes and confidence scores
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classNames[classId] == 'cell phone':
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

                similar_object = any(obj['class_name'] == 'cell phone' for obj in detected_objects)
                if not similar_object:
                    # Collect timestamp and servo angles
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    pan_angle, tilt_angle = get_angles_from_esp(esp_ip)

                    # Crop object image
                    cropped_image = frame[y:y + h, x:x + w]
                    _, buffer = cv2.imencode('.jpg', cropped_image)
                    cropped_image_bytes = base64.b64encode(buffer).decode('utf-8')

                    # Append object information to the list
                    detected_objects.append({
                        'timestamp': timestamp,
                        'pan_angle': pan_angle,
                        'tilt_angle': tilt_angle,
                        'class_name': 'Garbage',
                        'confidence': confidence,
                        'image_url': f'data:image/jpeg;base64,{cropped_image_bytes}'
                    })

    # Apply non-maximum suppression to remove redundant bounding boxes
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # Check if any cell phones are detected and draw bounding boxes
    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, 'Garbage', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            found_cell_phone = True

    # Check for alert condition
    if found_cell_phone:
        pan_angle, tilt_angle = get_angles_from_esp(esp_ip)
        send_control_command(pan_angle, tilt_angle, False, 0)
        if pan_angle is not None and tilt_angle is not None:
            print(f'Cell phone detected!({pan_angle},{tilt_angle})')


# Function for video feed
def video_feed(quality):
    url_map = {
        'low': f'http://{esp_ip}/cam-lo.jpg',
        'mid': f'http://{esp_ip}/cam-mid.jpg',
        'hi': f'http://{esp_ip}/cam-hi.jpg'
    }

    url = url_map.get(quality)
    if not url:
        error_message = b'Error: Camera feed not available'
        return error_message

    while True:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)

        # Prepare input blob for object detection
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        # Get output layer names
        layer_names = net.getLayerNames()
        output_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        # Forward pass through the network
        outputs = net.forward(output_names)

        # Perform object detection and draw bounding boxes
        findObject(outputs, frame)

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


# Function to capture a frame from the video feed
def capture_frame():
    # Adjust the URL based on your video feed source
    url = f'http://{esp_ip}/cam-mid.jpg'  # Example URL for mid-quality video feed

    try:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)

        return frame
    except Exception as e:
        print(f"Error: Failed to capture frame from the video feed. Exception: {str(e)}")
        return None


# Route to capture a frame and upload it to Firebase Storage
@app.route('/capture_and_upload_frame', methods=['POST'])
def capture_and_upload_frame():
    frame = capture_frame()

    if frame is not None:
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Create a temporary file to store the captured frame
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file_path = temp_file.name
            cv2.imwrite(temp_file_path, frame)

        # Construct the filename with timestamp
        filename = f"{timestamp}.jpg"

        # Specify the directory path in Firebase Storage
        directory_path = 'maintenance/'

        # Upload the captured frame to Firebase Storage
        with open(temp_file_path, 'rb') as f:
            blob = bucket.blob(directory_path + filename)
            blob.upload_from_file(f, content_type='image/jpeg')

        # Delete the temporary frame file
        os.remove(temp_file_path)

        return 'Frame uploaded successfully'
    else:
        return 'Error: Failed to capture frame from the video feed'

@app.route('/controlpanel')
def espcam():
    return render_template('espcam.html', camera_feed='/video_feed/mid')

@app.route('/video_feed/<quality>')
def video_feed_route(quality):
    return Response(video_feed(quality), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_objects')
def get_detected_objects():
    # Convert float32 values to float
    for obj in detected_objects:
        obj['confidence'] = format(float(obj['confidence']), '.2f')

    # Check if detected_objects is empty
    if not detected_objects:
        # If no objects are detected, return an empty list
        return jsonify([])
    else:
        # If objects are detected, return the list of detected objects
        return jsonify(detected_objects)

if __name__ == '__main__':
    app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
    app.run(debug=True)