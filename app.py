import os
import cv2
import csv
from flask import flash
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask import request, session, redirect, url_for, flash
from flask_mysqldb import MySQL
from database_details import dbhost, dbuser, dbpassword
from datetime import datetime, timedelta
from attendance_from_cam import capture_images, face_cropped_from_list, recognize_faces
from attendify_model import FaceRecognitionModel

# Define face_classifier globally
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Import TensorFlow and Keras from tensorflow.keras
# Disable oneDNN optimizations if required
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'attendify2023'
socketio = SocketIO(app)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'W7301@jqir#'
app.config['MYSQL_DB'] = 'attendify'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Modified face_cropped function
def face_cropped(img):
    try:
        if img is None or img.size == 0:
            print("Error: Empty or invalid input image")
            return None
        
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces == ():
            return None

        # Crop faces from the image
        cropped_faces = []
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            cropped_faces.append(cropped_face)

        return cropped_faces

    except Exception as e:
        print("Error in face_cropped:", e)
        return None


@app.route('/loginpage', methods=['POST', 'GET'])
def loginpage():
    if 'username' in session:
        # Redirect the user to the appropriate page if a session exists
        if session['username'].startswith('faculty'):
            return redirect(url_for('facultyprofile'))
        else:
            return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur = mysql.connection.cursor()
        # Check credentials against the faculties table
        cur.execute("SELECT * FROM faculties WHERE username = %s AND password = %s", (username, password))
        faculty = cur.fetchone()

        # If faculty login succeeds
        if faculty:
            session['user_name'] = faculty['name']
            session['username'] = faculty['username']
            flash('Login successful!', 'success')
            return redirect(url_for('facultyprofile'))

        # If faculty login fails, check against the users table
        cur.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cur.fetchone()
        if user:
            session['user_name'] = user['name']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('index'))

        cur.close()

        flash('Login failed. Check your username and password.', 'danger')
        return render_template('login.html')

    return render_template('login.html')




@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if request.method == 'POST':
        new_password = request.form['new_password']
        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET password = %s WHERE username = %s", (new_password, session['username']))
        mysql.connection.commit()
        cur.close()

        flash('Password changed successfully!', 'success')
        return redirect(url_for('index'))

    return render_template('change_password.html')

@app.route("/logout")
def logout():
    session.pop('user_name', None)
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/')
def index():
    if 'user_name' in session:
        if 'admin' in session['username']:
            admin = True
        else:
            admin = False
        return render_template('index.html', name=session['user_name'], admin=admin)
    else:
        return render_template('index.html', name=None)
    
    
@app.route("/about")
def about():
    if 'user_name' in session:
        return render_template('about.html', name=session['user_name'])
    else:
        return redirect(url_for('loginpage'))

students_path = "Attendify/Students"  # Replace with the actual path to the "students" directory

@app.route('/addfaculty')
def addfaculty():
    return render_template('./addfaculty.html')

@app.route('/addstudent', methods=['POST', 'GET'])
def addstudent():
    if 'user_name' in session:
        if request.method == 'POST':
            # Extract form data
            student_id = request.form['studentID'].upper()
            name = request.form['Name'].upper()
            year = request.form['year']
            group = request.form['group'].upper()
            batch = request.form['batch'].upper()
            reg_face = False

            if os.path.exists(students_path):
                # Get the list of folders inside the "students" directory
                student_folders = [folder for folder in os.listdir(students_path) if os.path.isdir(os.path.join(students_path, folder))]

                if(student_id in student_folders):
                    reg_face = True

            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO students (student_id, name, year, student_group, batch, reg_face) VALUES (%s, %s, %s, %s, %s, %s)",
                        (student_id, name, year, group, batch, reg_face))
            mysql.connection.commit()
            cur.close()

            # Now, label the captured images for this student
            student_images_path = os.path.join(students_path, student_id)
            label_images(student_images_path)  # Call the function to label images

        return render_template('addStudent.html')
    else:
        return redirect(url_for('index'))

# Function to label images
# Function to label images
def label_images(image_dir):
    mapping = {}  # Mapping dictionary to store image file paths and corresponding labels
    
    # Predefined mapping between filenames and student IDs or names
    # Replace this with your actual mapping logic
    # For example, if filenames follow a pattern like "2.1.jpg", "2.2.jpg", ..., "2.500.jpg",
    # you can extract the student ID or name from the filename
    # and use it as the label
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):  # Adjust file extension as needed
            # Extract student ID or name from the filename
            # For example, if filename format is "<student_ID>.<index>.jpg"
            # you can split the filename by '.' and extract the first part
            # You may need to customize this based on your actual filename format
            label = filename.split('.')[0]
            
            # Construct the full file path
            file_path = os.path.join(image_dir, filename)
            
            # Assign the label to the mapping dictionary
            mapping[file_path] = label

    # Save the mapping to a CSV file
    with open('labeling_dataset.csv', 'w', newline='') as csvfile:
        fieldnames = ['Image', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for file_path, label in mapping.items():
            writer.writerow({'Image': file_path, 'Label': label})

    print("Labeling completed. Mapping saved to 'labeling_dataset.csv'.")


@app.route('/addtt', methods=['POST'])


@app.route('/addtt', methods=['POST'])
def add_timetable():
    if request.method == 'POST':
        faculty_username = request.form['facultyID']
        class_timing = request.form['time']
        classroom = request.form['courseid']
        batch_name = request.form['batch']
        year = request.form['year']
        grp = request.form['group']
        class_type = request.form['teachtype']
        subject_name = request.form['subject_name']
        date = request.form['date']  

        # Check if the timetable entry already exists
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM class_details WHERE faculty_username = %s AND class_timing = %s AND classroom = %s",
                       (faculty_username, class_timing, classroom))
        existing_entry = cursor.fetchone()

        if existing_entry:
            # If the entry exists, update it
            cursor.execute("UPDATE class_details SET batch_name = %s, year = %s, grp = %s, class_type = %s, subject_name = %s, date = %s WHERE faculty_username = %s AND class_timing = %s AND classroom = %s",
                           (batch_name, year, grp, class_type, subject_name, date, faculty_username, class_timing, classroom))
            flash('Timetable updated successfully', 'success')
        else:
            # If the entry doesn't exist, insert a new one
            cursor.execute("INSERT INTO class_details (faculty_username, class_timing, classroom, batch_name, year, grp, class_type, subject_name, date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                           (faculty_username, class_timing, classroom, batch_name, year, grp, class_type, subject_name, date))
            flash('New timetable entry added successfully', 'success')

        mysql.connection.commit()
        cursor.close()

        return redirect(url_for('addfaculty'))

from flask import render_template, redirect, url_for, session


from datetime import datetime, timedelta, time


from datetime import datetime, timedelta

@app.route('/facultyprofile')
def facultyprofile():
    if 'username' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM faculties WHERE username = %s", (session['username'],))
        faculty = cur.fetchone()
        cur.close()

        if faculty:
            # Retrieve class details for the faculty
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM class_details WHERE faculty_username = %s", (session['username'],))
            class_details = cur.fetchall()
            cur.close()

            # Ensure class_timing is handled correctly and determine class duration
            timings = []
            class_duration = timedelta(hours=1)  # Assuming each class is 1 hour long

            for cd in class_details:
                class_timing = cd['class_timing']
                if isinstance(class_timing, str):
                    try:
                        class_time = datetime.strptime(class_timing, "%H:%M:%S").time()
                    except ValueError:
                        class_time = None  # Handle invalid time formats
                elif isinstance(class_timing, timedelta):
                    class_time = (datetime.min + class_timing).time()
                else:
                    class_time = None  # Handle unexpected data types
                if class_time:
                    timings.append((class_time, cd))

            # Sort timings for easier comparison
            timings.sort(key=lambda x: x[0])

            current_class = None
            next_class = None

            # Retrieve current time
            now = datetime.now()
            current_time = now.time()
            print(f"Current time: {current_time}")  # Debugging statement

            for class_time, cd in timings:
                if class_time:
                    class_start = datetime.combine(now.date(), class_time)
                    class_end = class_start + class_duration
                    print(f"Checking class: {class_time} - Start: {class_start}, End: {class_end}")  # Debugging statement

                    if class_start <= now < class_end:
                        current_class = cd
                        print(f"Current class found: {current_class}")  # Debugging statement
                    elif class_time > current_time and next_class is None:
                        next_class = cd
                        print(f"Next class found: {next_class}")  # Debugging statement

            currentclasstime = current_class['class_timing'] if current_class else None
            nextclasstime = next_class['class_timing'] if next_class else None

            def format_timedelta(td):
                return (datetime.min + td).time().strftime("%H:%M:%S") if td else None

            return render_template('faculty.html', name=faculty['name'], email=faculty['email'],
                                   depart=faculty['department'],
                                   timings=[time.strftime("%H:%M:%S") for time, _ in timings],
                                   classrooms=[cd['classroom'] for _, cd in timings],
                                   batches=[cd['batch_name'] for _, cd in timings],
                                   groups=[cd['grp'] for _, cd in timings],
                                   class_types=[cd['class_type'] for _, cd in timings],
                                   currentclasstime=format_timedelta(currentclasstime),
                                   nextclasstime=format_timedelta(nextclasstime))

    return render_template('faculty.html')


@app.route('/Adminhome')
def admin():
    if 'user_name' in session:
        if 'admin' in session['username']:
            return render_template('Admin.html', name=session['user_name'])
    return redirect(url_for('logout'))

@app.route('/modeldetails')
def modeldetails():

    if 'user_name' in session:
        if 'admin' in session['username']:
            return render_template('modeldetail.html', admin=session['user_name'])
    return redirect(url_for('loginpage'))

@app.route('/retrain')
def retrain():
    print("model training started")
    model = FaceRecognitionModel(train_data_path='Attendify/Students', test_data_path='Attendify/Students')
    model.train_model(epochs=10)
    print(model.OutputNeurons)
    return "Finished Training"


@app.route('/takeattendance', methods=['POST'])
def take_attendance():
    if 'user_name' in session:
        currentclass = request.form['currentclass']
        batches = request.form['Batches']
        classroom = request.form['classroom']
        bl = batches.split(', ')

        cur = mysql.connection.cursor()
        cur.execute("SELECT student_id FROM students WHERE batch IN (%s)" % ','.join(['%s' for _ in bl]), tuple(bl))
        students = cur.fetchall()
        cur.close()
        total_students = len(students)
        student_ids = [student['student_id'] for student in students]
        # print(student_ids)

        return render_template('takeattendance.html', currentclass=currentclass, Batches=batches, total=total_students,
                               classroom=classroom, attdone=False)

    return redirect(url_for('index'))


@app.route('/captureattendance', methods=['POST'])
def captureattendance():
    if 'user_name' in session:
        # Accessing form data
        current_class_data = request.form['currentclass']
        batches_data = request.form['Batches']

        # Get the current date
        current_date = datetime.now().strftime('%Y-%m-%d')

        # Perform calculations to get student enrollment data and attendance data
        # For the purpose of this example, let's assume you have these two lists
        bl = batches_data.split(', ')
        cur = mysql.connection.cursor()
        cur.execute("SELECT student_id FROM students WHERE batch IN (%s)" % ','.join(['%s' for _ in bl]), tuple(bl))
        students = cur.fetchall()
        cur.close()

        student_ids = [student['student_id'] for student in students]
        all_students = student_ids
        url_to_capture = "http://192.168.0.107:8080//shot.jpg"
        classimages = capture_images(url=url_to_capture, interval_seconds=2, num_images=10)

        student_faces = face_cropped_from_list(classimages)
        main_list = recognize_faces(student_faces)
        print(main_list)
        flat_list = [item for sublist in main_list for item in sublist]

        # Count the occurrences of each string
        count_dict = {}
        for item in flat_list:
            count_dict[item] = count_dict.get(item, 0) + 1

        # Create a list of strings that occur more than 4 times
        present_students = [key for key, value in count_dict.items() if value > 2]
        print(present_students)
        # present_students = ['E21CSEU0130']
        total_students_data = len(present_students)

        # Create a CSV file with attendance information for each date
        csv_filename = f"{current_class_data}_{batches_data}_{current_date}.csv"
        csv_filepath = os.path.join("Attendify/attendance_files", csv_filename)

        # Check if the CSV file already exists
        csv_exists = os.path.exists(csv_filepath)

        with open(csv_filepath, mode='a', newline='') as csvfile:
            fieldnames = ['Student', current_date]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if the file is new
            if not csv_exists:
                writer.writeheader()

            # Write data
            for student in all_students:
                attendance_data = 1 if student in present_students else 0
                row_data = {'Student': student, current_date: attendance_data}
                writer.writerow(row_data)

        # Return a response or redirect to a success page
        return render_template('takeattendance.html', currentclass=current_class_data, Batches=batches_data,
                               total=total_students_data, attdone=True, filepath=csv_filepath)
    return redirect(url_for('loginpage'))


@app.route('/showattendance', methods=['POST'])
def showattendance():
    if 'user_name' in session:
        # Path to the CSV file
        current_class_data = request.form['currentclass']
        batches_data = request.form['Batches']
        total_students_data = request.form['total']
        csv_filepath = request.form['filepath']

        # Read CSV file and prepare data for rendering in the template
        attendance_data = {}
        dates = []
        
        if os.path.exists(csv_filepath):
            with open(csv_filepath, mode='r') as csvfile:
                reader = csv.DictReader(csvfile)
                dates = [header for header in reader.fieldnames if header != 'Student']

                for row in reader:
                    student = row['Student']
                    attendance_values = [row[date] for date in dates]
                    attendance_data[student] = attendance_values

        return render_template('showattendance.html', attendance_data=attendance_data, dates=dates, currentclass=current_class_data,Batches =batches_data, total = total_students_data)
    return redirect(url_for('loginpage'))

@app.route('/collectdataset')
def collectdataset():
    return render_template('collectimages.html')

def gen_dataset(enrolment):
    student_folder = os.path.join("Attendify/Students", enrolment)
    print("Student folder:", student_folder)  # Debugging statement

    if not os.path.exists(student_folder):
        os.makedirs(student_folder)
    else:
        existing_files = os.listdir(student_folder)
        for existing_file in existing_files:
            file_path = os.path.join(student_folder, existing_file)
            os.remove(file_path)

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize VideoCapture object
    cap = cv2.VideoCapture(0)  # Change the camera index if needed

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate

    img_id = 0

    while True:
        ret, frame = cap.read()
        faces = face_classifier.detectMultiScale(frame, 1.2, 5)
        for (x, y, w, h) in faces:
            img_id += 1
            face = cv2.resize(frame[y: y+h, x:x+w], (200, 200))
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_path = os.path.join(student_folder, f"{enrolment}.{img_id}.jpg")
            cv2.imwrite(file_path, face_gray)
            print("Saved:", file_path)  # Debugging statement

        if img_id == 500:
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()


@socketio.on('enrolment')
def handle_enrolment(enrolment):
    gen_dataset(enrolment)
    emit('message', 'Dataset Creation Completed')


if __name__ == '__main__':
    socketio.run(app, debug=True)

