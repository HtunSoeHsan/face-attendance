# Face Attendance System

A desktop application for marking attendance using facial recognition. Supports both online and offline modes.

---

## Features
- **Face Recognition:** Detects and recognizes faces to mark attendance.
- **Offline Support:** Uses MongoDB for local data storage when offline.
- **Student Management:** Add, update, and delete student records.
- **Attendance History:** Displays attendance records grouped by month.
- **User-Friendly Interface:** Easy-to-use GUI with tabs for Dashboard, Student List, and Attendance History.

---

## Technologies Used
- **Python 3.x**
- **PyQt6** (for GUI)
- **OpenCV** (for webcam feed and image processing)
- **face_recognition** (for facial recognition)
- **MongoDB** (for local data storage)
- **Firestore** (optional, for online data storage)
- **pymongo** (Python library for MongoDB)
- **requests** (for checking internet connectivity)

---

## Setup Instructions

### Prerequisites
1. **Python 3.x** installed on your system.
2. **MongoDB** installed and running locally.
3. **Webcam** for facial recognition.

### Installation Steps
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/htunsoehsan/face-attendance.git
   cd face-attendance
