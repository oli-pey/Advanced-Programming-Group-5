# Advanced-Programming-Group-5

## Default Admin

When the SQLite database is created for the first time, the app seeds a default admin account.

- Username: `admin`
- Password: `admin`

You can override these by setting `DEFAULT_ADMIN_USERNAME` and `DEFAULT_ADMIN_PASSWORD` before starting the app.

erDiagram
    USERS ||--o{ ENTRIES : "has many"
    
    USERS {
        INTEGER id PK
        VARCHAR(100) username UK "Index"
        VARCHAR(255) password_hash
        BOOLEAN is_admin
        DATETIME created_at
    }

    ENTRIES {
        INTEGER id PK "Index"
        INTEGER user_id FK "Index"
        BLOB original_image
        BLOB downsized_image
        TEXT prediction
        VARCHAR(50) model_name
        TEXT probability
        DATETIME created_at
    }


    PROJECT DOCUMENTATION
Digit Recognizer
README / Application Overview

An interactive NiceGUI web application for drawing handwritten digits, classifying them with machine learning models, saving prediction history, and experimenting with custom datasets through a sandbox workflow.
Repository	https://github.com/oli-pey/Advanced-Programming-Group-5
Entry point	main.py
Database	SQLite / mydata.db
Authors	Jules Andreas Kern, Tim Freyvogel, Oliver Gustaf Peyron

Prepared: May 11, 2026
 
Overview
Digit Recognizer is an interactive web-based application that allows users to draw handwritten digits and classify them using machine learning models. The application supports user authentication, prediction history, admin views, and a sandbox area for creating custom datasets and experimenting with image-based classification.
Application type	Web-based digit recognition application
Primary interface	NiceGUI browser interface
Core input method	Drawing canvas for handwritten digits
Machine learning models	CNN model and logistic regression model
Persistence layer	SQLite database managed with SQLAlchemy
Primary users	Students and users experimenting with handwritten digit recognition

Features
Feature Area	Description
Digit recognition	Draw digits directly in the browser and classify handwritten input from 0 to 9.
Model selection	Choose between available AI models, including CNN and logistic regression.
Prediction history	Store original image, processed image, predicted digit, model name, confidence value, and timestamp.
User accounts	Create accounts, log in, and keep prediction history separated by user.
Admin views	Access an admin dashboard and admin-wide prediction history.
Sandbox datasets	Create custom datasets, add classes, upload image samples, and draw samples directly in the browser.
Database persistence	Store users, prediction entries, sandbox datasets, classes, samples, and training-related records in SQLite.

Intended Audience
This application is built for students and users who want to experiment with handwritten digit recognition, machine learning models, and simple image classification workflows through an interactive web interface.
It is also useful for learning how a Python web application can combine authentication, database storage, image preprocessing, and machine learning inference.
Tech Stack
Category	Technologies
Application framework	NiceGUI
Programming language	Python
Database layer	SQLite, SQLAlchemy
Machine learning	PyTorch, Torchvision, NumPy
Image processing	Pillow
Document / SVG utilities	ReportLab, svglib, lxml

Installation
Clone the repository:
git clone https://github.com/oli-pey/Advanced-Programming-Group-5.git
cd Advanced-Programming-Group-5

Create and activate a virtual environment:
python -m venv .venv

On macOS/Linux:
source .venv/bin/activate

On Windows:
.venv\Scripts\activate

Install the required dependencies:
pip install -r requirements.txt

Run the application:
python main.py

The application starts as a NiceGUI web app.
Configuration
The application uses a local SQLite database named:
mydata.db

The database is created automatically when the application starts.
Default Admin Account
When the SQLite database is created for the first time, the app seeds a default admin account:
Username: admin
Password: admin

These values can be overridden with environment variables:
DEFAULT_ADMIN_USERNAME=your_admin_username
DEFAULT_ADMIN_PASSWORD=your_admin_password

The NiceGUI storage secret can also be configured:
STORAGE_SECRET=your_secure_secret

Usage
Login
Open the application and log in with an existing account. New users can create an account from the login page.
Default admin login:
Username: admin
Password: admin

Main Digit Recognition Page
After logging in, users can:
1.	Draw a digit from 0 to 9 on the canvas.
2.	Select an AI model.
3.	Click Predict & Save.
4.	View the predicted digit.
5.	Save the result automatically to the database.
Prediction History
Users can open their history page to view previous predictions. Each prediction stores the original drawing, processed image, predicted digit, model used, confidence value, and creation time.
Admin Area
Admin users can access the admin dashboard and admin prediction history. These pages are intended for reviewing stored prediction data and managing application-level views.
Sandbox Area
The sandbox area allows users to create custom datasets for experimentation. Users can create datasets, add class labels, upload image samples, draw samples manually, store samples by class, and work with training and prediction tools for sandbox datasets.
Project Structure
Advanced-Programming-Group-5/
|
├── main.py
├── routes.py
├── auth.py
├── requirements.txt
|
├── DB/
│   ├── database.py
│   └── database_model.md
|
├── ml/
│   ├── base.py
│   ├── preprocessing.py
│   ├── registry.py
│   ├── result.py
│   |
│   ├── models/
│   │   ├── cnn_model.py
│   │   ├── cnn_mnist.pt
│   │   ├── logreg_model.py
│   │   └── logreg_mnist.pt
│   |
│   └── recognizers/
│       ├── cnn_recognizer.py
│       └── logreg_recognizer.py
|
├── web/
│   ├── index.py
│   ├── history.py
│   ├── admin.py
│   ├── layout.py
│   └── sandbox.py
|
├── sandbox/
├── sandbox_ml/
└── README.md

Main Files and Responsibilities
File / Folder	Responsibility
main.py	Starts the NiceGUI application and registers all routes.
routes.py	Defines web routes for login, logout, main recognition, history, admin pages, and sandbox pages.
auth.py	Handles authentication, password hashing and verification, user lookup, user creation, and session checks.
DB/database.py	Defines the SQLite database connection and SQLAlchemy models.
web/index.py	Implements the main drawing and prediction page, including preprocessing, prediction execution, and result saving.
web/history.py	Displays prediction history for the logged-in user.
web/admin.py	Implements admin-only views and admin-level access to stored prediction information.
web/sandbox.py	Implements the sandbox workflow for datasets, classes, image samples, and experimentation.
ml/registry.py	Registers the available prediction models and returns the selected recognizer.
ml/recognizers/	Contains model-specific recognizer implementations.
ml/models/	Contains model definitions and trained model files.

Database Model Overview
The application uses a relational SQLite database. The core relationship for prediction history is:
USERS 1 --- * ENTRIES

A user can have many prediction entries. A simplified prediction entry contains:
PredictionEntry
- id
- user_id
- original_image
- downsized_image
- prediction
- model_name
- probability
- created_at

Sandbox data extends the database with datasets, classes, samples, training jobs, and trained models.
User Story
As a user, I want to draw handwritten digits in a web application and receive an AI-based prediction, so that I can experiment with digit recognition and understand how different machine learning models classify handwritten input.
This user story guided the application design. The following sections explain how the requirements are fulfilled.
Drawing Digits
Requirement	As a user, I want to draw a digit directly in the browser.
Implementation	The main page provides an interactive drawing canvas where users can draw a digit using mouse or touch input.
Responsible files	web/index.py, main.py, routes.py

Predicting Digits
Requirement	As a user, I want the application to predict which digit I drew.
Implementation	The drawing is converted into an image, processed into a machine-learning-compatible format, and passed to the selected recognizer. The prediction result is displayed to the user.
Responsible files	web/index.py, ml/registry.py, ml/recognizers/, ml/models/, ml/preprocessing.py

Choosing a Model
Requirement	As a user, I want to select which machine learning model should classify my digit.
Implementation	The application provides a model selector with the available models: cnn and logreg.
Responsible files	web/index.py, ml/registry.py, ml/recognizers/cnn_recognizer.py, ml/recognizers/logreg_recognizer.py

Saving Prediction History
Requirement	As a user, I want my predictions to be saved so I can review them later.
Implementation	Each prediction is saved in the SQLite database together with the original image, processed image, model name, prediction, confidence value, and timestamp.
Responsible files	web/index.py, web/history.py, DB/database.py

User Accounts
Requirement	As a user, I want to create an account and log in so my predictions are stored separately from other users.
Implementation	The application supports login, account creation, password hashing, and user-specific sessions.
Responsible files	auth.py, routes.py, DB/database.py

Admin Access
Requirement	As an admin, I want access to admin pages for reviewing application data.
Implementation	Admin-only views are available through dedicated routes. Admin access is checked before rendering admin pages.
Responsible files	web/admin.py, routes.py, auth.py, DB/database.py

Sandbox Datasets
Requirement	As a user, I want to create custom datasets and add labeled image samples for experimentation.
Implementation	The sandbox section allows users to create datasets, add classes, upload image samples, draw samples, and store them in the database.
Responsible files	web/sandbox.py, DB/database.py

License
No formal license specified. Intended for academic use.
Authors
•	Jules Andreas Kern
•	Tim Freyvogel
•	Oliver Gustaf Peyron
