import sys
import pickle
from collections import defaultdict
from datetime import datetime, timedelta

import cv2
import cvzone
import face_recognition
import numpy as np
import requests
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QTabWidget,
    QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QTableWidget, QTableWidgetItem, QLineEdit, QMessageBox
)
from pymongo import MongoClient

# Initialize MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["attendance_system"]
students_collection = db["students"]
attendance_collection = db["attendance"]

# Check Internet Connection
def is_internet_available():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

class AttendanceApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Attendance System")
        self.setGeometry(100, 100, 900, 600)

        # 📌 Main Container
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout()

        # 1️⃣ **Title Section**
        title_layout = QHBoxLayout()

        # Logo Image
        self.logo_label = QLabel(self)
        pixmap = QPixmap("resource/logo.png")  # Ensure logo.png exists
        self.logo_label.setPixmap(pixmap.scaled(100, 100))  # Resize logo
        title_layout.addWidget(self.logo_label)

        # Title Text
        self.title_label = QLabel("Face Attendance System")
        self.title_label.setStyleSheet("font-size: 32px; font-weight: bold; padding: 10px;")
        title_layout.addWidget(self.title_label)

        main_layout.addLayout(title_layout)

        # 2️⃣ **Tabs (Dashboard, Student List, Attendance History)**
        self.tabs = QTabWidget()

        # 📷 **Dashboard Tab (Face Recognition)**
        self.dashboard_tab = QWidget()
        self.setup_dashboard()
        self.tabs.addTab(self.dashboard_tab, "Dashboard")

        # 📚 **Student List Tab**
        self.student_list_tab = QWidget()
        self.setup_student_list()
        self.tabs.addTab(self.student_list_tab, "Student List")

        # 📜 **Attendance History Tab**
        self.attendance_history_tab = QWidget()
        self.setup_attendance_history()
        self.tabs.addTab(self.attendance_history_tab, "Attendance History")

        main_layout.addWidget(self.tabs)
        self.central_widget.setLayout(main_layout)

    # 🔴 **Dashboard (Face Recognition) Setup**
    def setup_dashboard(self):
        layout = QHBoxLayout()

        # **Webcam Feed**
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        layout.addWidget(self.video_label)

        # **Student Info**
        info_layout = QVBoxLayout()
        self.student_label = QLabel("Waiting for detection...")
        self.student_label.setStyleSheet("font-size: 18px; color: white; background-color: black; padding: 10px;")
        info_layout.addWidget(self.student_label)

        self.attendance_status_label = QLabel("")
        self.attendance_status_label.setStyleSheet("font-size: 16px; color: green;")
        info_layout.addWidget(self.attendance_status_label)

        # **Exit Button**
        self.exit_button = QPushButton("Exit")
        self.exit_button.setStyleSheet("background-color: red; color: white; font-size: 16px; padding: 10px;")
        self.exit_button.clicked.connect(self.close)
        info_layout.addWidget(self.exit_button)

        layout.addLayout(info_layout)
        self.dashboard_tab.setLayout(layout)

        # Load Face Encodings
        with open("EncodeFile.p", "rb") as file:
            self.encodeListKnown, self.studentIds = pickle.load(file)

        # Setup Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # Start Webcam Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # 📚 **Student List Tab Setup**
    def setup_student_list(self):
        layout = QVBoxLayout()

        # Table for Student List
        self.student_table = QTableWidget()
        self.student_table.setColumnCount(5)
        self.student_table.setHorizontalHeaderLabels(["Student ID", "Name", "Major", "Year", "Started"])
        self.student_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.student_table)

        # Input Fields Section
        input_layout = QVBoxLayout()
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("Student ID")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Name")
        self.major_input = QLineEdit()
        self.major_input.setPlaceholderText("Major")
        self.year_input = QLineEdit()
        self.year_input.setPlaceholderText("Year")
        self.started_input = QLineEdit()
        self.started_input.setPlaceholderText("Started")

        input_layout.addWidget(self.id_input)
        input_layout.addWidget(self.name_input)
        input_layout.addWidget(self.major_input)
        input_layout.addWidget(self.year_input)
        input_layout.addWidget(self.started_input)

        # Buttons Section
        btn_layout = QHBoxLayout()
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_student)
        self.edit_button = QPushButton("Update")
        self.edit_button.clicked.connect(self.edit_student)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_student)
        btn_layout.addWidget(self.add_button)
        btn_layout.addWidget(self.edit_button)
        btn_layout.addWidget(self.delete_button)
        input_layout.addLayout(btn_layout)

        layout.addLayout(input_layout)
        self.student_list_tab.setLayout(layout)
        self.load_students()

    def load_students(self):
        students = students_collection.find()
        self.student_table.setRowCount(0)
        for row, student in enumerate(students):
            self.student_table.insertRow(row)
            self.student_table.setItem(row, 0, QTableWidgetItem(student["_id"]))
            self.student_table.setItem(row, 1, QTableWidgetItem(student.get("name", "Unknown")))
            self.student_table.setItem(row, 2, QTableWidgetItem(student.get("major", "N/A")))
            self.student_table.setItem(row, 3, QTableWidgetItem(student.get("year", "N/A")))
            self.student_table.setItem(row, 4, QTableWidgetItem(student.get("started", "N/A")))

    def add_student(self):
        student_id = self.id_input.text()
        name = self.name_input.text()
        major = self.major_input.text()
        year = self.year_input.text()
        started = self.started_input.text()

        if student_id and name and major and year and started:
            student_data = {
                "_id": student_id,
                "name": name,
                "major": major,
                "year": year,
                "started": started,
                "attendance": []
            }
            students_collection.insert_one(student_data)
            QMessageBox.information(self, "Success", "Student added successfully!")
            self.load_students()
        else:
            QMessageBox.warning(self, "Error", "Please fill all fields!")

    def edit_student(self):
        student_id = self.id_input.text()
        name = self.name_input.text()
        major = self.major_input.text()
        if student_id and name and major:
            students_collection.update_one(
                {"_id": student_id},
                {"$set": {"name": name, "major": major}}
            )
            QMessageBox.information(self, "Success", "Student updated successfully!")
            self.load_students()
        else:
            QMessageBox.warning(self, "Error", "Please enter student ID and details!")

    def delete_student(self):
        student_id = self.id_input.text()
        if student_id:
            students_collection.delete_one({"_id": student_id})
            QMessageBox.information(self, "Success", "Student deleted successfully!")
            self.load_students()
        else:
            QMessageBox.warning(self, "Error", "Please enter Student ID!")

    # 📜 **Attendance History Tab Setup**
    def setup_attendance_history(self):
        layout = QVBoxLayout()
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(["Student ID", "Name", "Major", "Year", "Month", "Attendance Count"])
        layout.addWidget(self.history_table)
        self.attendance_history_tab.setLayout(layout)
        self.load_attendance_history()

    def load_attendance_history(self):
        students = students_collection.find()
        self.history_table.setRowCount(0)
        for student in students:
            student_id = student["_id"]
            attendance = student.get("attendance", [])
            monthly_count = defaultdict(int)
            for timestamp in attendance:
                dt = datetime.fromisoformat(timestamp)
                month_key = dt.strftime("%Y-%m")
                monthly_count[month_key] += 1
            for month, count in monthly_count.items():
                row_position = self.history_table.rowCount()
                self.history_table.insertRow(row_position)
                self.history_table.setItem(row_position, 0, QTableWidgetItem(student_id))
                self.history_table.setItem(row_position, 1, QTableWidgetItem(student.get("name", "N/A")))
                self.history_table.setItem(row_position, 2, QTableWidgetItem(student.get("major", "N/A")))
                self.history_table.setItem(row_position, 3, QTableWidgetItem(student.get("year", "N/A")))
                self.history_table.setItem(row_position, 4, QTableWidgetItem(month))
                self.history_table.setItem(row_position, 5, QTableWidgetItem(str(count)))

    # 📷 **Face Recognition Processing**
    def update_frame(self):
        """ Capture webcam feed and perform face recognition. """
        success, img = self.cap.read()
        if not success:
            return

        # Convert frame to RGB
        small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        # Face recognition
        face_cur_frame = face_recognition.face_locations(small_img)
        encode_cur_frame = face_recognition.face_encodings(small_img, face_cur_frame)

        detected_student_id = None
        student_info_text = "No face detected."
        attendance_status = ""

        for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
            matches = face_recognition.compare_faces(self.encodeListKnown, encode_face)
            face_dis = face_recognition.face_distance(self.encodeListKnown, encode_face)
            match_index = np.argmin(face_dis)

            top, right, bottom, left = [v * 4 for v in face_loc]

            if match_index < len(self.studentIds) and matches[match_index]:
                student_id = self.studentIds[match_index]
                detected_student_id = student_id

                try:
                    # Fetch the student data for the detected student by ID
                    student = students_collection.find_one({"_id": student_id})
                    current_time = datetime.now()

                    if student:
                        attendance = student.get("attendance", [])
                        if attendance:
                            last_attendance_time = datetime.fromisoformat(attendance[-1])
                            time_difference = current_time - last_attendance_time
                            if time_difference < timedelta(minutes=1):
                                attendance_status = "✅ Already Attended"
                            else:
                                attendance.append(current_time.isoformat())
                                students_collection.update_one(
                                    {"_id": student_id},
                                    {"$set": {"attendance": attendance}}
                                )
                                attendance_status = "✅ Attendance Recorded!"
                        else:
                            attendance.append(current_time.isoformat())
                            students_collection.update_one(
                                {"_id": student_id},
                                {"$set": {"attendance": attendance}}
                            )
                            attendance_status = "✅ Attendance Recorded!"

                        # Construct student info text
                        student_info_text = f"ID: {student_id}\n"
                        student_info_text += f"Name: {student.get('name', 'N/A')}\n"
                        student_info_text += f"Major: {student.get('major', 'N/A')}\n"
                        student_info_text += f"Year: {student.get('year', 'N/A')}\n"
                        student_info_text += f"Attendance: {len(attendance)}\n"
                    else:
                        student_info_text = "❌ Student data not found."
                        attendance_status = "❌ Student not found."

                except Exception as e:
                    student_info_text = "❌ Error fetching student data."
                    attendance_status = f"❌ Error: {str(e)}"
                    print(f"Error: {e}")

                # Draw rectangle around the face
                cvzone.cornerRect(img, (left, top, right - left, bottom - top), rt=5, colorR=(0, 255, 0))
            else:
                student_info_text = "❌ Unknown Face Detected"
                attendance_status = "❌ No matching student found."

        # Update UI with the student info and attendance status
        self.student_label.setText(student_info_text)
        self.attendance_status_label.setText(attendance_status)

        # Convert OpenCV image to QPixmap
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        """Clean up resources when closing the app."""
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceApp()
    window.show()
    sys.exit(app.exec())