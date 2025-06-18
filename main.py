# Main.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")
import argparse
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
import face_recognition
import os
import time
import shutil # Import shutil for file operations
from utils import draw_box
from gui import FaceRecognitionGUI

# Import ctypes only if on Windows for setting hidden attribute
if os.name == 'nt':
    import ctypes
    FILE_ATTRIBUTE_HIDDEN = 0x02

KNOWN_FACES_ENCODINGS = 'known_faces_encodings.pkl'
OUTPUT_DIR_BASE = os.path.join(os.getcwd(), 'Output')
VIDEO_TEMP_OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, 'Video') # New temp and final video directory


def load_known_faces():
    if not os.path.exists(KNOWN_FACES_ENCODINGS):
        print('Known faces encodings not found. Please run encode_faces.py first.')
        return [], []
    with open(KNOWN_FACES_ENCODINGS, 'rb') as f:
        data = pickle.load(f)
    return data['encodings'], data['names']

def recognize_faces(frame, known_encodings, known_names, threshold=0.2):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=1-threshold)
        name = 'Unknown'
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        results.append(((left, top, right, bottom), name))
    return results

def process_request(mode, source, settings, display_fn=None, stop_event=None, save_video_to_path=None):
    """Process the detection request. For video, save_video_to_path determines output location."""
    # Load face encodings
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        print('No known faces loaded. Exiting.')
        return None # Return None for consistency

    # Extract settings
    threshold = settings['threshold']
    yolo_conf = settings['yolo_conf']
    yolo_iou = settings['yolo_iou']
    
    # Initialize YOLO model
    model = YOLO('models/yolov8s.pt')  # Load YOLO model

    def yolo_infer(frame):
        return model(frame, conf=yolo_conf, iou=yolo_iou)

    if mode == 'realtime' or mode == 'webcam':  # Handle both naming conventions
        cap = cv2.VideoCapture(0)
        while True:
            if stop_event and stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame, yolo_infer, known_encodings, known_names, threshold)
            if display_fn:
                display_fn(frame)
                time.sleep(0.03)
            else:
                cv2.imshow('Detection & Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        if not display_fn:
            cv2.destroyAllWindows()
        return None # Return None for consistency
    
    elif mode == 'image':
        if not isinstance(source, str) or not source:
            print('Image not found.')
            return None 
        frame = cv2.imread(str(source))
        if frame is None:
            print('Image not found.')
            return None 
        frame = process_frame(frame, yolo_infer, known_encodings, known_names, threshold)
        if display_fn:
            display_fn(frame)
        else:
            cv2.imshow('Image Preview', frame)
            cv2.waitKey(0)
            cv2.destroyWindow('Image Preview')
        return None # Image processing displays frame, doesn't save a file here. GUI handles saving current_frame_to_save.
    elif mode == 'video':
        if not isinstance(source, str) or not source:
            print('Video not found.')
            return None
        
        cap = cv2.VideoCapture(str(source))
        output_file_path_for_writer = None # Path used by VideoWriter

        # Ensure the target directory for videos (temp or final) exists
        os.makedirs(VIDEO_TEMP_OUTPUT_DIR, exist_ok=True)

        if save_video_to_path: 
            output_file_path_for_writer = save_video_to_path
        else: 
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            base_name = os.path.basename(str(source)) if source else "unknown_video"
            # Generate a non-hidden name first for VideoWriter
            temp_filename_base = f"temp_processed_{timestamp}_{base_name}"
            output_file_path_for_writer = os.path.join(VIDEO_TEMP_OUTPUT_DIR, temp_filename_base)

        if not output_file_path_for_writer: 
            print("Error: Output path for video not determined.")
            cap.release()
            return None

        if hasattr(cv2.VideoWriter, 'fourcc'): # Corrected attribute access
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        else:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(output_file_path_for_writer, fourcc, fps, (width, height))
        print(f"Processing video. Output will be temporarily at: {output_file_path_for_writer}")

        while True:
            if stop_event and stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame, yolo_infer, known_encodings, known_names, threshold)
            
            if out: # Write frame if VideoWriter is initialized
                out.write(frame)

            if display_fn:
                display_fn(frame)
                time.sleep(0.03) # Keep for live preview consistency
            else: # CLI fallback, though less relevant if GUI is primary
                cv2.imshow('Video Preview', frame) # Changed window name for clarity
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        
        final_output_path = output_file_path_for_writer # Path to be returned, possibly modified for hidden attribute

        if out:
            out.release()
            if stop_event and stop_event.is_set():
                print(f'Video processing stopped. Incomplete output at {final_output_path}')
            else:
                print(f'Video processed and saved to {final_output_path}')
                # Attempt to make the temporary file hidden
                try:
                    if os.name == 'nt': # Windows
                        ret = ctypes.windll.kernel32.SetFileAttributesW(str(final_output_path), FILE_ATTRIBUTE_HIDDEN)
                        if ret:
                            print(f"Successfully set hidden attribute for: {final_output_path}")
                        else:
                            print(f"Failed to set hidden attribute for: {final_output_path}. Error code: {ctypes.GetLastError()}")
                    elif os.name == 'posix': # Unix-like (Linux, macOS)
                        dir_name = os.path.dirname(final_output_path)
                        base_name = os.path.basename(final_output_path)
                        if not base_name.startswith('.'):
                            hidden_base_name = "." + base_name
                            hidden_path = os.path.join(dir_name, hidden_base_name)
                            os.rename(final_output_path, hidden_path)
                            final_output_path = hidden_path # Update path to the new hidden name
                            print(f"Renamed temporary file to hidden: {final_output_path}")
                except Exception as e:
                    print(f"Error making temporary file hidden: {e}")
        
        if not display_fn:
            cv2.destroyAllWindows()
        
        # Return the path of the video file created (temp or final)
        # Only return if processing was not stopped, or if we decide to keep partial files
        if stop_event and stop_event.is_set():
            return None 
        return final_output_path
    
    return None # Default return for other modes or if no file path is generated

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='Real-time Object Detection and Face Recognition')
    parser.add_argument('--threshold', type=float, default=0.5, help='Face recognition threshold (0-1)')
    args = parser.parse_args()

    # Initialize GUI with the processing callback
    gui = FaceRecognitionGUI(process_callback=process_request)
    
    # Run the GUI
    gui.run()

def process_frame(frame, model, known_encodings, known_names, threshold):
    """Detect, recognize faces, and annotate frame without displaying"""
    results = model(frame)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for box, cls in zip(boxes, classes):
            label = results[0].names[cls]  # Use class names from the results object
            draw_box(frame, box, label)
            if label == 'person':
                x1, y1, x2, y2 = box
                face_roi = frame[y1:y2, x1:x2]
                faces = recognize_faces(face_roi, known_encodings, known_names, threshold)
                for (l, t, r, b), name in faces:
                    abs_box = (x1+l, y1+t, x1+r, y1+b)
                    draw_box(frame, abs_box, name, color=(255,0,0))
    return frame

if __name__ == '__main__':
    main()
