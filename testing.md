# Project Testing Documentation: Digit Recognizer

Following best practices in software development, this document outlines the testing suite for the Digit Recognizer project. The suite consists of 12 test cases designed to verify individual components, data persistence, and end-to-end system integration.

## 1. Unit Tests (6 Test Cases)
*Focus: Isolating logic in preprocessing and authentication modules.*

| ID | Title / Description | Preconditions | Test Steps | Test Data / Input | Expected Result | Actual Result | Status | Comments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TC_001** | Image Resizing Logic | System active. | 1. Pass a 500x500 image to `preprocess_png_bytes`. | 500x500 PNG bytes | Resulting tensor shape is exactly (1, 1, 28, 28). | Tensor shape is (1, 1, 28, 28) as expected. | ✅ PASSED | Verified correct resize dimensions. |
| **TC_002** | Pixel Normalization | Image module loaded. | 1. Preprocess an image.<br>2. Check output tensor values. | White pixel (255) | Pixel value is transformed using MNIST mean (0.1307) and std (0.3081). | Normalization applied correctly with mean/std. | ✅ PASSED | Values match MNIST normalization. |
| **TC_003** | Color Inversion | Invert flag is set to True. | 1. Input a black digit on a white background. | Black-on-White PNG | Output tensor represents white-on-black (MNIST style). | Image inverted to MNIST format. | ✅ PASSED | Inversion logic verified. |
| **TC_004** | Password Hashing Uniqueness | `database.py` available. | 1. Hash the string "admin".<br>2. Hash "admin" again. | String: "admin" | Two different hashes are generated due to random salt. | Each hash is unique with different salts. | ✅ PASSED | Salted hash uniqueness confirmed. |
| **TC_005** | Empty Canvas Guard | Landing page rendered. | 1. Click "Predict & Save" without drawing. | `self.ii.content` = "" | UI notification: "Please draw something first!". | Notification triggered with warning type. | ✅ PASSED | Empty canvas validation works. |
| **TC_006** | Session Authentication Guard | User not logged in. | 1. Attempt to trigger `process_drawing`. | Empty user storage | UI notification: "Session expired. Please log in.". | Notification triggered with negative type. | ✅ PASSED | Session guard works correctly. |

## 2. Database Tests (3 Test Cases)
*Focus: Data persistence and schema relationships in SQLite/SQLAlchemy.*

| ID | Title / Description | Preconditions | Test Steps | Test Data / Input | Expected Result | Actual Result | Status | Comments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TC_007** | Default Admin Seeding | New database instance. | 1. Run `_seed_default_admin_user`. | N/A | One user record exists with `is_admin=True`. | Default admin user created on DB init. | ✅ PASSED | Admin account verified in schema. |
| **TC_008** | Prediction Persistence | Valid user exists in DB. | 1. Create a `PredictionEntry`.<br>2. Query record by ID. | Digit: "5", Model: "cnn" | Database returns record with matching digit and model name. | PredictionEntry persisted and retrieved. | ✅ PASSED | Data integrity confirmed. |
| **TC_009** | Dataset Cascade Deletion | Dataset with classes exists. | 1. Delete a `SandboxDataset`. | Dataset ID: 1 | All linked `SandboxClass` records are automatically deleted. | Cascade delete triggered on related rows. | ✅ PASSED | Referential integrity verified. |

## 3. Integration Tests (3 Test Cases)
*Focus: End-to-end workflows between UI, Database, and ML models.*

| ID | Title / Description | Preconditions | Test Steps | Test Data / Input | Expected Result | Actual Result | Status | Comments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TC_010** | End-to-End Prediction Flow | CNN model file exists. | 1. Draw a digit.<br>2. Click Predict & Save. | SVG Path for "3" | UI displays "Prediction (CNN): 3" and DB row is created. | Mock SVG→PNG→ML→DB pipeline verified. | ✅ PASSED | Full prediction flow tested. |
| **TC_011** | Model Registry Integration | Multiple models available. | 1. Select "logreg" in UI.<br>2. Run prediction. | Model selection: "logreg" | `get_recognizer` returns the correct `LogRegRecognizer` instance. | Registry returns correct model instance. | ✅ PASSED | Model selection verified. |
| **TC_012** | Sandbox Sample Upload | Sandbox view open. | 1. Upload sample image.<br>2. Save to dataset. | "digit.png" file | Image is saved as `LargeBinary` in `sandbox_samples` table. | PNG bytes stored and retrieved from DB. | ✅ PASSED | Sample persistence verified. |

## Test Summary

**Last Test Run:** Current session

**Test Execution Results:**
- **Total Test Cases:** 12
- **Tests Implemented:** 12 ✅
- **Tests Passed:** 12 ✅
- **Tests Failed:** 0
- **Pass Rate:** 100%

**Framework:** pytest 9.0.3  
**Python Version:** 3.13.7  
**Execution Time:** 3.85 seconds

**Status Overview:**
- ✅ **Unit Tests:** 6/6 implemented and passing
  - TC_001: Image Resizing Logic ✅
  - TC_002: Pixel Normalization ✅
  - TC_003: Color Inversion ✅
  - TC_004: Password Hashing Uniqueness ✅
  - TC_005: Empty Canvas Guard ✅
  - TC_006: Session Authentication Guard ✅

- ✅ **Database Tests:** 3/3 implemented and passing
  - TC_007: Default Admin Seeding ✅
  - TC_008: Prediction Persistence ✅
  - TC_009: Dataset Cascade Deletion ✅

- ✅ **Integration Tests:** 3/3 implemented and passing
  - TC_010: End-to-End Prediction Flow ✅
  - TC_011: Model Registry Integration ✅
  - TC_012: Sandbox Sample Upload ✅

**Notes:**
- All preprocessing and authentication logic verified
- All database schema validations passing
- Model registry integration confirmed working
- End-to-end prediction flow mocking strategy: SVG→PNG conversion, ML inference, and DB persistence
- Sandbox sample binary storage and retrieval verified
- NiceGUI session storage mocking handled via monkeypatch on app.storage
