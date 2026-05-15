# Project Testing Documentation: Digit Recognizer

Following best practices in software development, this document outlines the testing suite for the Digit Recognizer project. The suite consists of 12 test cases designed to verify individual components, data persistence, and end-to-end system integration.

## 1. Unit Tests (6 Test Cases)
*Focus: Isolating logic in preprocessing and authentication modules.*

| ID | Title / Description | Preconditions | Test Steps | Test Data / Input | Expected Result | Actual Result | Status | Comments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TC_001** | Image Resizing Logic | System active. | 1. Pass a 500x500 image to `preprocess_png_bytes`. | 500x500 PNG bytes | Resulting tensor shape is exactly (1, 1, 28, 28). | | | |
| **TC_002** | Pixel Normalization | Image module loaded. | 1. Preprocess an image.<br>2. Check output tensor values. | White pixel (255) | Pixel value is transformed using MNIST mean (0.1307) and std (0.3081). | | | |
| **TC_003** | Color Inversion | Invert flag is set to True. | 1. Input a black digit on a white background. | Black-on-White PNG | Output tensor represents white-on-black (MNIST style). | | | |
| **TC_004** | Password Hashing Uniqueness | `database.py` available. | 1. Hash the string "admin".<br>2. Hash "admin" again. | String: "admin" | Two different hashes are generated due to random salt. | | | |
| **TC_005** | Empty Canvas Guard | Landing page rendered. | 1. Click "Predict & Save" without drawing. | `self.ii.content` = "" | UI notification: "Please draw something first!". | | | |
| **TC_006** | Session Authentication Guard | User not logged in. | 1. Attempt to trigger `process_drawing`. | Empty user storage | UI notification: "Session expired. Please log in.". | | | |

## 2. Database Tests (3 Test Cases)
*Focus: Data persistence and schema relationships in SQLite/SQLAlchemy.*

| ID | Title / Description | Preconditions | Test Steps | Test Data / Input | Expected Result | Actual Result | Status | Comments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TC_007** | Default Admin Seeding | New database instance. | 1. Run `_seed_default_admin_user`. | N/A | One user record exists with `is_admin=True`. | | | |
| **TC_008** | Prediction Persistence | Valid user exists in DB. | 1. Create a `PredictionEntry`.<br>2. Query record by ID. | Digit: "5", Model: "cnn" | Database returns record with matching digit and model name. | | | |
| **TC_009** | Dataset Cascade Deletion | Dataset with classes exists. | 1. Delete a `SandboxDataset`. | Dataset ID: 1 | All linked `SandboxClass` records are automatically deleted. | | | |

## 3. Integration Tests (3 Test Cases)
*Focus: End-to-end workflows between UI, Database, and ML models.*

| ID | Title / Description | Preconditions | Test Steps | Test Data / Input | Expected Result | Actual Result | Status | Comments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TC_010** | End-to-End Prediction Flow | CNN model file exists. | 1. Draw a digit.<br>2. Click Predict & Save. | SVG Path for "3" | UI displays "Prediction (CNN): 3" and DB row is created. | | | |
| **TC_011** | Model Registry Integration | Multiple models available. | 1. Select "logreg" in UI.<br>2. Run prediction. | Model selection: "logreg" | `get_recognizer` returns the correct `LogRegRecognizer` instance. | | | |
| **TC_012** | Sandbox Sample Upload | Sandbox view open. | 1. Upload sample image.<br>2. Save to dataset. | "digit.png" file | Image is saved as `LargeBinary` in `sandbox_samples` table. | | | |
