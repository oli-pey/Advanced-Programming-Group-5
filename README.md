# Advanced-Programming-Group-5


# Digit Recognizer

Digit Recognizer is an interactive web-based application that allows users to draw handwritten digits and classify them using machine learning models. The application supports user authentication, prediction history, admin views, and a sandbox area for creating custom datasets and experimenting with image-based classification.

---

## Features

- Draw digits directly in the browser using a mouse or touch input
- Predict handwritten digits from **0 to 9**
- Choose between available AI models:
  - CNN model
  - Logistic regression model
- Store prediction results with:
  - Original image
  - Processed/downscaled image
  - Predicted digit
  - Model name
  - Confidence/probability value
  - Timestamp
- User login and account creation
- Personal prediction history
- Admin dashboard
- Admin-wide prediction history
- Sandbox for creating custom datasets
- Add custom dataset classes
- Upload image samples
- Draw samples directly in the browser
- Store sandbox samples and training-related data
- SQLite-based persistence

---

## Intended Audience

This application is intended for students, educators, and beginner machine learning developers who want to explore handwritten digit recognition through an interactive web interface.

Students can use the application to understand how image classification works in practice. Educators can use it as a teaching tool to demonstrate machine learning concepts, model comparison, image preprocessing, and prediction confidence. Beginner developers can also use the project as an example of how to combine a Python web application with authentication, database storage, image preprocessing, and machine learning inference.

The sandbox functionality also makes the application useful for users who want to experiment with custom image datasets beyond the standard digit recognition workflow.

---

## Tech Stack

- Python
- NiceGUI
- SQLAlchemy
- SQLite
- Pillow
- PyTorch
- Torchvision
- NumPy
- ReportLab
- svglib
- lxml

---

## Installation

Clone the repository:

```bash
git clone https://github.com/oli-pey/Advanced-Programming-Group-5.git
cd Advanced-Programming-Group-5
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python main.py
```

The application starts as a NiceGUI web app.

---

## Configuration

The application uses a local SQLite database:

```text
mydata.db
```

The database is created automatically when the application starts.

### Default Admin Account

When the SQLite database is created for the first time, the app seeds a default admin account:

```text
Username: admin
Password: admin
```

These values can be overridden with environment variables:

```bash
DEFAULT_ADMIN_USERNAME=your_admin_username
DEFAULT_ADMIN_PASSWORD=your_admin_password
```

The NiceGUI storage secret can also be configured:

```bash
STORAGE_SECRET=your_secure_secret
```

---

## Usage

### Login

Open the application and log in with an existing account.

Default admin login:

```text
Username: admin
Password: admin
```

New users can create an account from the login page.

### Main Digit Recognition Page

After logging in, users can:

1. Draw a digit from 0 to 9 on the canvas.
2. Select an AI model.
3. Click **Predict & Save**.
4. View the predicted digit.
5. Save the result automatically to the database.

### Prediction History

Users can open their history page to view previous predictions.

Each prediction stores:

- Original drawing
- Processed image
- Predicted digit
- Model used
- Confidence/probability
- Creation time

### Admin Area

Admin users can access:

- Admin dashboard
- Admin prediction history

These pages are intended for reviewing stored prediction data and managing application-level views.

### Sandbox Area

The sandbox area allows users to create custom datasets for experimentation.

Users can:

- Create datasets
- Add class labels
- Upload image samples
- Draw samples manually
- Store samples by class
- Work with training and prediction tools for sandbox datasets

---

## Project Structure

```text
Advanced-Programming-Group-5/
│
├── main.py
├── routes.py
├── auth.py
├── requirements.txt
│
├── DB/
│   ├── database.py
│   └── database_model.md
│
├── ml/
│   ├── base.py
│   ├── preprocessing.py
│   ├── registry.py
│   ├── result.py
│   │
│   ├── models/
│   │   ├── cnn_model.py
│   │   ├── cnn_mnist.pt
│   │   ├── logreg_model.py
│   │   └── logreg_mnist.pt
│   │
│   └── recognizers/
│       ├── cnn_recognizer.py
│       └── logreg_recognizer.py
│
├── web/
│   ├── index.py
│   ├── history.py
│   ├── admin.py
│   ├── layout.py
│   └── sandbox.py
│
├── sandbox/
├── sandbox_ml/
└── README.md
```
---
## 1. Testing Procedure Overview
The strategy is divided into three distinct layers, moving from high-level user flows to granular code logic and finally to end-to-end component integration.

---

## A. Manual Feature Test (UAT)
The manual feature test ensures the "happy path" is functional and the user experience is seamless. This is the primary verification step before any live demonstration.

### Checklist & User Flow:
- [ ] **Authentication:** Successfully register a new account and login.
- [ ] **Dataset Initialization:** Create a new sandbox dataset.
- [ ] **Schema Definition:** Define and create specific classes for the ML task.
- [ ] **Sample Management:** - Upload sample images.
    - Draw/Annotate samples.
    - Delete samples to ensure cleanup logic works.
- [ ] **Model Training:** Trigger the training process for the defined dataset.
- [ ] **Inference:** Predict outcomes using the newly trained model.
- [ ] **Specialized Testing:** Verify the MNIST digit recognizer specifically.
- [ ] **Persistence:** Restart the application/container and verify that the database state is maintained.

---

## B. Unit Tests
Unit tests focus on isolating small, stateless functions to ensure they produce the correct output for given inputs. These tests do not require the UI or a live database.

### Core Focus Areas:
- **Data Processing:** Label normalization and validation.
- **File System:** Logic for creating storage paths and directory structures.
- **Validation:** Image format validation and dataset integrity checks prior to training.
- **ML Logic:** Model instantiation by type and ensuring prediction results match the expected JSON/Data format.

### Target Modules:
- `sandbox/services.py` (Business logic)
- `sandbox/storage.py` (IO pathing logic)
- `sandbox_ml/models.py` (Architecture definitions)
- `sandbox_ml/training.py` (Validation helpers and data loaders)

---

## C. Integration Tests
Integration tests verify that different modules (Services, Database, and ML Engine) communicate correctly. These tests simulate real-world usage without manual intervention.

### Implementation Requirements:
- **Environment:** Use a temporary **SQLite** database and a dedicated temporary image folder to ensure tests are idempotent and do not pollute production data.
- **End-to-End Flow:**
    1. Programmatically create a dataset, classes, and samples in the test DB.
    2. Execute a "tiny" training run using low-resolution fake images to verify the pipeline.
    3. Save the resulting model to the temporary disk.
    4. Load the saved model and perform a prediction using raw image bytes.

---

## Main Files and Responsibilities

### `main.py`

Starts the NiceGUI application and registers all routes.

### `routes.py`

Defines the web routes for:

- Login
- Logout
- Main digit recognition page
- User history
- Admin dashboard
- Admin history
- Sandbox overview
- Sandbox dataset pages

### `auth.py`

Handles authentication-related functionality, including:

- Password hashing
- Password verification
- User lookup
- User creation
- Session checks

### `DB/database.py`

Defines the SQLite database connection and SQLAlchemy models.

Main stored data includes:

- Users
- Prediction entries
- Sandbox datasets
- Sandbox classes
- Sandbox samples
- Sandbox training jobs
- Sandbox trained models

### `web/index.py`

Implements the main digit drawing and prediction page.

Responsible for:

- Drawing input
- Model selection
- Image preprocessing
- Prediction execution
- Saving prediction results

### `web/history.py`

Displays prediction history for the logged-in user.

### `web/admin.py`

Implements admin-only views and admin-level access to stored prediction information.

### `web/sandbox.py`

Implements the sandbox workflow for creating datasets, adding classes, uploading samples, drawing samples, and experimenting with custom classification data.

### `ml/registry.py`

Registers the available prediction models and returns the selected recognizer.

Available models:

```text
cnn
logreg
```

### `ml/recognizers/`

Contains model-specific recognizer implementations.

### `ml/models/`

Contains model definitions and trained model files.

---

## Database Model Overview

The application uses a relational SQLite database.

Core relationship:

```text
USERS 1 ─── * ENTRIES
```

A user can have many prediction entries.

Simplified prediction entry data:

```text
PredictionEntry
- id
- user_id
- original_image
- downsized_image
- prediction
- model_name
- probability
- created_at
```

Sandbox data extends the database with datasets, classes, samples, training jobs, and trained models.

---

## User Story

As a user, I want to draw handwritten digits in a web application and receive an AI-based prediction, so that I can experiment with digit recognition and understand how different machine learning models classify handwritten input.

This user story guided the application design. The following sections explain how the requirements are fulfilled.

---

## Drawing Digits

### Requirement

As a user, I want to draw a digit directly in the browser.

### Implementation

The main page provides an interactive drawing canvas where users can draw a digit using mouse or touch input.

### Responsible Files

```text
web/index.py
main.py
routes.py
```

---

## Predicting Digits

### Requirement

As a user, I want the application to predict which digit I drew.

### Implementation

The drawing is converted into an image, processed into a machine-learning-compatible format, and passed to the selected recognizer.

The prediction result is displayed to the user.

### Responsible Files

```text
web/index.py
ml/registry.py
ml/recognizers/
ml/models/
ml/preprocessing.py
```

---

## Choosing a Model

### Requirement

As a user, I want to select which machine learning model should classify my digit.

### Implementation

The application provides a model selector with the available models.

Available models:

```text
cnn
logreg
```

### Responsible Files

```text
web/index.py
ml/registry.py
ml/recognizers/cnn_recognizer.py
ml/recognizers/logreg_recognizer.py
```

---

## Saving Prediction History

### Requirement

As a user, I want my predictions to be saved so I can review them later.

### Implementation

Each prediction is saved in the SQLite database together with the original image, processed image, model name, prediction, confidence value, and timestamp.

### Responsible Files

```text
web/index.py
web/history.py
DB/database.py
```

---

## User Accounts

### Requirement

As a user, I want to create an account and log in so my predictions are stored separately from other users.

### Implementation

The application supports login, account creation, password hashing, and user-specific sessions.

### Responsible Files

```text
auth.py
routes.py
DB/database.py
```

---

## Admin Access

### Requirement

As an admin, I want access to admin pages for reviewing application data.

### Implementation

Admin-only views are available through dedicated routes. Admin access is checked before rendering admin pages.

### Responsible Files

```text
web/admin.py
routes.py
auth.py
DB/database.py
```

---

## Sandbox Datasets

### Requirement

As a user, I want to create custom datasets and add labeled image samples for experimentation.

### Implementation

The sandbox section allows users to create datasets, add classes, upload image samples, draw samples, and store them in the database.

### Responsible Files

```text
web/sandbox.py
DB/database.py
```

---

## Conclusion

All main requirements are fulfilled through a modular Python application structure, an interactive NiceGUI web interface, user authentication, SQLite-based persistence, machine learning inference, and a sandbox workflow for custom dataset experimentation.

---

## License

No formal license specified. Intended for academic use.

---

## Authors

Jules Andreas Kern  
Tim Freyvogel  
Oliver Gustaf Peyron
