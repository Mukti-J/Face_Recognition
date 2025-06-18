# Changelog

All significant changes to this project will be documented in this file.

## [Unreleased] - YYYY-MM-DD 
### Added
- Organized output into `Output/Image` and `Output/Video` subdirectories.
- Temporary processed video files are now stored in `Output/Video` and made hidden by default (using file attributes on Windows or prepending a dot `.` on POSIX systems).

### Changed
- Video processing workflow:
    - Videos are processed and saved to a temporary file when the "Start" button is pressed.
    - The "Save" button for videos now copies the already processed temporary file to the user-selected final location, significantly speeding up the save operation and avoiding re-processing.
- Default save directories for images and videos now point to their respective subfolders within `Output`.

### Removed
- Obsolete code and variables related to older video saving mechanisms (e.g., `save_video_thread`, old `TEMP_OUTPUT_DIR` definitions).

## [2025-06-08]

### Added
- Implemented full GUI window interface with proper controls in `gui.py`:
  - Mode selection radio buttons (webcam, image, video)
  - File selection with browsing capability
  - Settings sliders for threshold, confidence, and IoU
  - Status display and Start/Stop controls
- Added threading support for background processing while keeping UI responsive
- Added proper status updates during processing

### Changed
- Completely refactored UI flow - now starts with a window interface instead of dialog boxes
- Separated processing logic from UI code with callback pattern
- Improved directory structure with dedicated Input and Output folders
- Updated all file handling to use the new directory structure

### Improved
- Better user experience with visual feedback during processing
- More intuitive interface with all options visible at once
- Status updates shown in the main window

## [Unreleased]

### Added
- Initial version of `main.py` with the following functions:
  - `load_known_faces()`: Loads known face encodings and names from a pickle file.
  - `recognize_faces(frame, known_encodings, known_names, threshold=0.5)`: Recognizes faces in a given frame using known encodings and a threshold.
  - `main()`: Entry point for the application, handles mode selection (webcam, image, video), loads models, and processes input.
  - `process_frame(frame, model, known_encodings, known_names, threshold, show=False)`: Processes a frame for object detection and face recognition, draws bounding boxes and labels.
- Command line arguments for YOLO confidence threshold (`--yolo_conf`) and NMS IoU threshold (`--yolo_iou`) for more flexible and optimal object detection.
- YOLO inference now uses these parameters for every frame/image/video processed.
- Created new `gui.py` module to handle all GUI-related functionality with the `FaceRecognitionGUI` class
- Added settings dialog capability for adjustable parameters (threshold, confidence, IoU)
- Added proper error handling and user feedback through the GUI
- Sidebar preview panel (400×300) in `gui.py` with letterboxed frame scaling for full-image display.
- `display_frame` callback integration to render frames inside the main GUI instead of separate `cv2.imshow` windows.
- Support for `stop_event` in `process_request` to gracefully terminate webcam and video streams from the GUI.
- Two-decimal numeric entry fields for face threshold, YOLO confidence, and IOU parameters with real-time sync.
- “+”/“-” buttons next to each entry to increment/decrement values by 0.1 directly from the GUI.

### Changed
- The `main` function now wraps YOLO inference in a `yolo_infer` function that applies the user-specified confidence and IoU thresholds.
- `process_frame` now uses `matplotlib` for static image preview, enabling zoom and pan. Keeps `cv2.imshow` for webcam/video.
- Updated `process_frame` to always use the YOLO class name from the results object for labeling detected objects.
- Refactored `main.py` to use the new GUI module for better separation of concerns
- Improved user experience with more consistent dialog boxes and prompts
- Refactored `process_request` and `process_frame` in `main.py` to accept a GUI `display_fn` and `stop_event`, removing direct calls to `cv2.imshow` when GUI is active.
- Updated `gui.py` to bind `DoubleVar` sliders to `StringVar` entries, ensuring consistent two-decimal formatting and bi-directional control.

### Fixed
- Prevented preview image clipping by implementing letterbox-style scaling in the GUI display function.

### Removed
- No removals yet.

---

> Please update this changelog with every function addition, update, or significant change in the project.
