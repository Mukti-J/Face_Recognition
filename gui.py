# gui.py
import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk # Keep tkinter for some basic functionalities if needed, or remove if not.
import os
import cv2
import threading
from PIL import Image, ImageTk
import shutil # Import shutil

class FaceRecognitionGUI:
    def __init__(self, process_callback=None):
        self.INPUT_DIR = os.path.join(os.getcwd(), 'Input')
        self.BASE_OUTPUT_DIR = os.path.join(os.getcwd(), 'Output')
        self.IMAGE_OUTPUT_DIR = os.path.join(self.BASE_OUTPUT_DIR, 'Image')
        self.VIDEO_OUTPUT_DIR = os.path.join(self.BASE_OUTPUT_DIR, 'Video') # For both temp and final videos

        os.makedirs(self.INPUT_DIR, exist_ok=True)
        os.makedirs(self.IMAGE_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.VIDEO_OUTPUT_DIR, exist_ok=True) # Ensure video dir exists for temp files

        self.source = None
        self.mode = None
        self.process_callback = process_callback
        self.processing_thread = None
        self.webcam_active = False
        self.last_processed_video_temp_path = None # To store path of temp processed video
        
        # Main window setup
        ctk.set_appearance_mode("dark")  # Modes: "System" (default), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "dark-blue", "green"

        self.root = ctk.CTk()
        self.root.title("Detection APP")
        self.root.geometry("1000x680") # Adjusted geometry to better fit the design
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.file_var = ctk.StringVar() # Initialize file_var
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header Frame (for title and theme button)
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.pack(fill=ctk.X, pady=(0, 20))

        title_label = ctk.CTkLabel(header_frame, text="Detection APP", font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(side=ctk.LEFT, padx=20, pady=10)

        self.theme_button = ctk.CTkButton(header_frame, text="☀️", command=self.toggle_theme, width=40, height=40, corner_radius=10)
        self.theme_button.pack(side=ctk.RIGHT, padx=20, pady=10)

        # Left Panel (for mode selection, settings, and version)
        left_panel = ctk.CTkFrame(self.main_frame, width=300, corner_radius=10)
        left_panel.pack(side=ctk.LEFT, fill=ctk.Y, padx=10, pady=10)
        left_panel.pack_propagate(False) # Prevent resizing

        # Mode selection frame
        mode_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        mode_frame.pack(fill=ctk.X, pady=10, padx=10)
        
        self.mode_var = ctk.StringVar(value="webcam")
        
        # Mode buttons
        self.realtime_button = ctk.CTkButton(mode_frame, text="Real-time",
                                             command=lambda: self.mode_var.set("webcam"),
                                             font=ctk.CTkFont(size=14, weight="bold"),
                                             fg_color=("#3E4756", "#3E4756"), # Darker shade for selected
                                             hover_color=("#4A5463", "#4A5463"),
                                             corner_radius=8, width=100, height=35)
        self.realtime_button.pack(side=ctk.LEFT, padx=5)

        self.image_button = ctk.CTkButton(mode_frame, text="Image",
                                          command=lambda: self.mode_var.set("image"),
                                          font=ctk.CTkFont(size=14, weight="bold"),
                                          fg_color=("#555B66", "#555B66"),
                                          hover_color=("#4A5463", "#4A5463"),
                                          corner_radius=8, width=100, height=35)
        self.image_button.pack(side=ctk.LEFT, padx=5)

        self.video_button = ctk.CTkButton(mode_frame, text="Video",
                                          command=lambda: self.mode_var.set("video"),
                                          font=ctk.CTkFont(size=14, weight="bold"),
                                          fg_color=("#555B66", "#555B66"),
                                          hover_color=("#4A5463", "#4A5463"),
                                          corner_radius=8, width=100, height=35)
        self.video_button.pack(side=ctk.LEFT, padx=5)

        # Input file selection (initially hidden for webcam mode)
        self.file_frame = ctk.CTkFrame(left_panel, corner_radius=10)
        self.file_frame.pack(fill=ctk.X, pady=10, padx=10) # Packed conditionally in update_ui_for_mode

        ctk.CTkLabel(self.file_frame, text="Selected File:", font=ctk.CTkFont(weight="bold")).pack(anchor=ctk.W, padx=10, pady=(10,0))
        
        # Create a container frame for the entry and button
        file_input_frame = ctk.CTkFrame(self.file_frame, fg_color="transparent")
        file_input_frame.pack(fill=ctk.X, padx=10, pady=(0,10))
        
        # Add the entry to the container frame
        self.file_path_entry = ctk.CTkEntry(file_input_frame, placeholder_text="No file selected", state="readonly")
        self.file_path_entry.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(0,5))
        
        # Add the browse button to the container frame
        self.browse_button = ctk.CTkButton(file_input_frame, text="Browse", command=self.browse_file, corner_radius=8, width=80)
        self.browse_button.pack(side=ctk.RIGHT)
        
        # Settings frame
        settings_frame = ctk.CTkFrame(left_panel, corner_radius=10)
        settings_frame.pack(fill=ctk.X, pady=10, padx=10)

        settings_title = ctk.CTkLabel(settings_frame, text="Detection Settings", font=ctk.CTkFont(weight="bold"))
        settings_title.pack(pady=(10, 5))
        
        # Create settings
        self.threshold_var = ctk.DoubleVar(value=0.5)
        # StringVar for formatted threshold display and two-way sync
        self.threshold_strvar = ctk.StringVar(value=f"{self.threshold_var.get():.2f}")
        self.threshold_var.trace_add('write', lambda *args: self.threshold_strvar.set(f"{self.threshold_var.get():.2f}"))
        self.threshold_strvar.trace_add('write', self.on_threshold_strvar_change)
        self.yolo_conf_var = ctk.DoubleVar(value=0.5)
        # StringVar for formatted YOLO confidence display and sync
        self.yolo_conf_strvar = ctk.StringVar(value=f"{self.yolo_conf_var.get():.2f}")
        self.yolo_conf_var.trace_add('write', lambda *args: self.yolo_conf_strvar.set(f"{self.yolo_conf_var.get():.2f}"))
        self.yolo_conf_strvar.trace_add('write', self.on_yolo_conf_strvar_change)
        self.yolo_iou_var = ctk.DoubleVar(value=0.45)
        # StringVar for formatted YOLO IoU display and sync
        self.yolo_iou_strvar = ctk.StringVar(value=f"{self.yolo_iou_var.get():.2f}")
        self.yolo_iou_var.trace_add('write', lambda *args: self.yolo_iou_strvar.set(f"{self.yolo_iou_var.get():.2f}"))
        self.yolo_iou_strvar.trace_add('write', self.on_yolo_iou_strvar_change)
        
        ctk.CTkLabel(settings_frame, text="Face Recognition Threshold:").pack(anchor=ctk.W, padx=10, pady=(10,0))
        threshold_slider = ctk.CTkSlider(settings_frame, from_=0, to=1, variable=self.threshold_var)
        threshold_slider.pack(fill=ctk.X, padx=10)
        threshold_input_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        threshold_input_frame.pack(fill=ctk.X, padx=10, pady=(0,10))
        ctk.CTkButton(threshold_input_frame, text="-", width=30, command=self.decrement_threshold).pack(side=ctk.LEFT)
        ctk.CTkEntry(threshold_input_frame, textvariable=self.threshold_strvar, width=50, justify=ctk.CENTER).pack(side=ctk.LEFT, expand=True, fill=ctk.X, padx=5)
        ctk.CTkButton(threshold_input_frame, text="+", width=30, command=self.increment_threshold).pack(side=ctk.RIGHT)

        ctk.CTkLabel(settings_frame, text="YOLO Confidence:").pack(anchor=ctk.W, padx=10, pady=(10,0))
        yolo_conf_slider = ctk.CTkSlider(settings_frame, from_=0, to=1, variable=self.yolo_conf_var)
        yolo_conf_slider.pack(fill=ctk.X, padx=10)
        yolo_conf_input_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        yolo_conf_input_frame.pack(fill=ctk.X, padx=10, pady=(0,10))
        ctk.CTkButton(yolo_conf_input_frame, text="-", width=30, command=self.decrement_yolo_conf).pack(side=ctk.LEFT)
        ctk.CTkEntry(yolo_conf_input_frame, textvariable=self.yolo_conf_strvar, width=50, justify=ctk.CENTER).pack(side=ctk.LEFT, expand=True, fill=ctk.X, padx=5)
        ctk.CTkButton(yolo_conf_input_frame, text="+", width=30, command=self.increment_yolo_conf).pack(side=ctk.RIGHT)

        ctk.CTkLabel(settings_frame, text="YOLO IoU Threshold:").pack(anchor=ctk.W, padx=10, pady=(10,0))
        yolo_iou_slider = ctk.CTkSlider(settings_frame, from_=0, to=1, variable=self.yolo_iou_var)
        yolo_iou_slider.pack(fill=ctk.X, padx=10)
        yolo_iou_input_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        yolo_iou_input_frame.pack(fill=ctk.X, padx=10, pady=(0,10))
        ctk.CTkButton(yolo_iou_input_frame, text="-", width=30, command=self.decrement_yolo_iou).pack(side=ctk.LEFT)
        ctk.CTkEntry(yolo_iou_input_frame, textvariable=self.yolo_iou_strvar, width=50, justify=ctk.CENTER).pack(side=ctk.LEFT, expand=True, fill=ctk.X, padx=5)
        ctk.CTkButton(yolo_iou_input_frame, text="+", width=30, command=self.increment_yolo_iou).pack(side=ctk.RIGHT)
        
        # Version label
        version_label = ctk.CTkLabel(left_panel, text="V 1.0", font=ctk.CTkFont(size=12), text_color="gray")
        version_label.pack(side=ctk.BOTTOM, pady=10)

        # Right Panel (for preview and control buttons)
        right_panel = ctk.CTkFrame(self.main_frame, corner_radius=10)
        right_panel.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=10, pady=10)

        # Preview area
        self.preview_frame = ctk.CTkFrame(right_panel, fg_color="#222222", corner_radius=10)
        self.preview_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)
        self.preview_frame.pack_propagate(False)
        self.preview_width = 600  # Adjusted for new layout
        self.preview_height = 400 # Adjusted for new layout
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="", font=ctk.CTkFont(size=24, weight="bold"), text_color="#666666")
        self.preview_label.pack(expand=True)

        # Status frame
        self.status_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        self.status_frame.pack(fill=ctk.X, pady=(0, 10), padx=10)
        
        self.status_var = ctk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(self.status_frame, textvariable=self.status_var,
                                   font=ctk.CTkFont(size=14), text_color="#00B0F0") # A shade of blue
        self.status_label.pack(side=ctk.LEFT, padx=10)

        # Control buttons
        control_button_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        control_button_frame.pack(pady=10)
        
        self.start_button = ctk.CTkButton(control_button_frame, text="Start",
                                         command=self.start_processing, corner_radius=8,
                                         font=ctk.CTkFont(size=16, weight="bold"), width=100, height=40)
        self.start_button.pack(side=ctk.LEFT, padx=5)
        
        self.stop_button = ctk.CTkButton(control_button_frame, text="Stop",
                                        command=self.stop_processing, state=ctk.DISABLED, corner_radius=8,
                                        font=ctk.CTkFont(size=16, weight="bold"), width=100, height=40)
        self.stop_button.pack(side=ctk.LEFT, padx=5)

        self.save_button = ctk.CTkButton(control_button_frame, text="Save",
                                        command=self.save_output, corner_radius=8,
                                        font=ctk.CTkFont(size=16, weight="bold"), width=100, height=40)
        self.save_button.pack(side=ctk.LEFT, padx=5)

        # Update UI based on mode selection
        self.mode_var.trace_add('write', self.update_ui_for_mode)
        self.update_ui_for_mode()
        
    def on_closing(self):
        """Handle the window closing event"""
        if hasattr(self, 'stop_event') and self.stop_event:
            self.stop_event.set()
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1)

        # Clean up any lingering temp file from the new location
        if self.last_processed_video_temp_path and os.path.exists(self.last_processed_video_temp_path):
            # Ensure we are only deleting files from our designated video output/temp folder
            if os.path.dirname(self.last_processed_video_temp_path) == self.VIDEO_OUTPUT_DIR:
                try:
                    os.remove(self.last_processed_video_temp_path)
                    print(f"Cleaned up temp file on closing: {self.last_processed_video_temp_path}")
                except OSError as e:
                    print(f"Error cleaning up temp file {self.last_processed_video_temp_path} on closing: {e}")
            else:
                print(f"Skipping cleanup of temp file not in video output dir: {self.last_processed_video_temp_path}")
        
        self.root.destroy()

    def run(self):
        """Run the main application"""
        self.root.mainloop()
        
    def update_ui_for_mode(self, *args):
        """Update UI elements based on the selected mode"""
        mode = self.mode_var.get()
        
        self.update_mode_buttons(mode)

        if mode == "webcam":
            self.file_frame.pack_forget() # Hides the file input section
        else:
            self.file_frame.pack(fill=ctk.X, pady=10, padx=10) # Shows the file input section

    def update_mode_buttons(self, current_mode):
        buttons = {
            "webcam": self.realtime_button,
            "image": self.image_button,
            "video": self.video_button
        }
        for mode, button in buttons.items():
            if mode == current_mode:
                button.configure(fg_color=("#3E4756", "#3E4756")) # Darker shade for selected
            else:
                button.configure(fg_color=("#555B66", "#555B66")) # Original shade

    def toggle_theme(self):
        current_mode = ctk.get_appearance_mode()
        if current_mode == "Dark":
            ctk.set_appearance_mode("Light")
            self.theme_button.configure(text="🌙")
        else:
            ctk.set_appearance_mode("Dark")
            self.theme_button.configure(text="☀️")
            
    def browse_file(self):
        """Open a file dialog to select input file"""
        mode = self.mode_var.get()
        
        if mode == "image":
            file_path = filedialog.askopenfilename(
                initialdir=self.INPUT_DIR,
                title="Select Image",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
            )
        elif mode == "video":
            file_path = filedialog.askopenfilename(
                initialdir=self.INPUT_DIR,
                title="Select Video",
                filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
            )
        else:
            return
            
        if file_path:
                 self.source = file_path
                 self.file_path_entry.configure(state="normal") # Enable editing
                 self.file_path_entry.delete(0, ctk.END) # Clear current text
                 self.file_path_entry.insert(0, file_path) # Insert new path
                 self.file_path_entry.configure(state="readonly") # Disable editing
            
    def get_mode(self):
        """Get the current mode"""
        return self.mode_var.get()
    
    def select_image(self):
        """Show dialog for selecting an image file"""
        root = tk.Tk()
        root.withdraw()
        self.source = filedialog.askopenfilename(
            initialdir=self.INPUT_DIR,
            title='Select Image',
            filetypes=[('Image Files', '*.jpg *.jpeg *.png')]
        )
        if not self.source:
            messagebox.showinfo('Info', 'No image selected. Exiting.')
            return None
        return self.source
    
    def select_video(self):
        """Show dialog for selecting a video file"""
        root = tk.Tk()
        root.withdraw()
        self.source = filedialog.askopenfilename(
            initialdir=self.INPUT_DIR,
            title='Select Video',
            filetypes=[('Video Files', '*.mp4 *.avi *.mov *.mkv')]
        )
        if not self.source:
            messagebox.showinfo('Info', 'No video selected. Exiting.')
            return None
        return self.source
    
    def show_error(self, message):
        """Show error message"""
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('Error', message)
    
    def get_output_path(self, source_path):
        """Generate output file path based on input path"""
        return os.path.join(self.BASE_OUTPUT_DIR, f'output_{os.path.basename(source_path)}')
    
    def show_result_message(self, output_path):
        """Show message about saved output"""
        print(f'Output saved to {output_path}')
        
    def save_output(self):
        """Save the current preview frame (image/webcam) or copy processed video to a user-chosen location."""
        mode = self.mode_var.get()

        if mode == "image" or mode == "webcam":
            if hasattr(self, 'current_frame_to_save') and self.current_frame_to_save is not None:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")],
                    initialdir=self.IMAGE_OUTPUT_DIR, # Default to Output/Image
                    title="Save Output Image"
                )
                if file_path:
                    try:
                        rgb_image = cv2.cvtColor(self.current_frame_to_save, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_image)
                        pil_image.save(file_path)
                        messagebox.showinfo("Success", f"Output saved to {file_path}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to save image: {e}")
            else:
                messagebox.showinfo("Info", "No frame to save. Please start processing first.")
        
        elif mode == "video":
            if not self.last_processed_video_temp_path or not os.path.exists(self.last_processed_video_temp_path):
                messagebox.showerror("Error", "No processed video found to save. Please process a video first using 'Start'.")
                return

            original_basename = os.path.basename(self.source) if self.source else "video.mp4"
            suggested_filename = f"output_{original_basename}"
            
            # Refine suggested_filename if temp file has the specific prefix
            temp_file_basename = os.path.basename(self.last_processed_video_temp_path)
            if temp_file_basename.startswith("temp_processed_"):
                 parts = temp_file_basename.split('_', 2)
                 if len(parts) > 2 and parts[2]: # Ensure there's a name part after timestamp
                     suggested_filename = f"output_{parts[2]}"
                 elif original_basename != "video.mp4": # Fallback to original if parsing temp fails but source exists
                     suggested_filename = f"output_{original_basename}"
                 # else keep the generic "output_video.mp4" or "output_original_basename"


            final_save_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")],
                initialdir=self.VIDEO_OUTPUT_DIR, # Default to Output/Video
                title="Save Processed Video As",
                initialfile=suggested_filename
            )

            if final_save_path:
                try:
                    self.status_var.set(f"Copying video to {os.path.basename(final_save_path)}...")
                    self.root.update_idletasks() # Update GUI
                    shutil.copy(self.last_processed_video_temp_path, final_save_path)
                    messagebox.showinfo("Success", f"Video saved to {final_save_path}")
                    self.status_var.set(f"Video saved to {os.path.basename(final_save_path)}")
                    # Optionally, you might want to delete the temp file now or keep it until next processing/app close
                    # For now, temp file is cleaned up on next video process or app close.
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save video: {e}")
                    self.status_var.set("Error saving video.")
        else:
            messagebox.showerror("Error", "Save function not implemented for this mode.")

    # _execute_video_save and _finalize_save_operation are no longer needed and should be removed.

    def show_settings_dialog(self, default_threshold=0.5):
        """Show settings dialog for adjusting parameters"""
        # This function is likely deprecated as settings are now directly in the main GUI
        pass

    def get_source(self):
        """Get the selected source file path"""
        return self.source
    
    def get_settings(self):
        """Get the current settings"""
        return {
            'threshold': self.threshold_var.get(),
            'yolo_conf': self.yolo_conf_var.get(),
            'yolo_iou': self.yolo_iou_var.get()
        }
    
    def start_processing(self):
        """Start processing the selected mode and file"""
        mode = self.mode_var.get()
        
        if mode in ["image", "video"] and not self.source:
            messagebox.showerror("Error", "Please select an input file first.")
            return
        
        # Clean up previous temp video file if starting new video processing
        if mode == "video":
            if self.last_processed_video_temp_path and os.path.exists(self.last_processed_video_temp_path):
                # Ensure we are only deleting files from our designated video output/temp folder
                if os.path.dirname(self.last_processed_video_temp_path) == self.VIDEO_OUTPUT_DIR:
                    try:
                        os.remove(self.last_processed_video_temp_path)
                        print(f"Removed old temp file: {self.last_processed_video_temp_path}")
                    except OSError as e:
                        print(f"Error removing old temp file {self.last_processed_video_temp_path}: {e}")
                else:
                    print(f"Skipping removal of old temp file not in video output dir: {self.last_processed_video_temp_path}")
            self.last_processed_video_temp_path = None # Reset before new processing
            
        self.stop_event = threading.Event()
        
        # Update UI state
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set(f"Processing {mode}...")
        self.save_button.configure(state=ctk.DISABLED)
        
        # Get settings
        settings = self.get_settings()
        
        # Start processing in a separate thread with display and stop controls
        self.processing_thread = threading.Thread(
            target=self.run_processing,
            args=(mode, self.source, settings, self.display_frame, self.stop_event),
            daemon=True
        )
        self.processing_thread.start()
    
    def run_processing(self, mode, source, settings, display_fn, stop_event):
        """Run the processing function provided at initialization"""
        processed_output_path = None 
        if self.process_callback:
            try:
                # For video, process_request will save to VIDEO_OUTPUT_DIR/temp_...
                processed_output_path = self.process_callback(mode, source, settings, display_fn, stop_event, None) 
            finally:
                if mode == "video": 
                    if not stop_event.is_set() and processed_output_path and os.path.exists(processed_output_path):
                        # Ensure the processed_output_path is within the VIDEO_OUTPUT_DIR
                        if os.path.dirname(processed_output_path) == self.VIDEO_OUTPUT_DIR:
                            self.last_processed_video_temp_path = processed_output_path
                            self.root.after(100, self.update_ui_for_file_completion)
                        else:
                            print(f"Error: Processed video path {processed_output_path} is not in the expected directory {self.VIDEO_OUTPUT_DIR}.")
                            self.last_processed_video_temp_path = None
                            self.root.after(100, self.update_ui_after_stop) # Treat as error/stop
                    else: 
                        if processed_output_path and os.path.exists(processed_output_path) and os.path.dirname(processed_output_path) == self.VIDEO_OUTPUT_DIR:
                             print(f"Video processing stopped/failed. Temp file (possibly incomplete): {processed_output_path}")
                        self.last_processed_video_temp_path = None 
                        self.root.after(100, self.update_ui_after_stop)
                elif mode == "image" and not stop_event.is_set(): # Image processing
                    self.root.after(100, self.update_ui_for_file_completion)
                else: # Webcam processing or other interrupted states
                    self.root.after(100, self.update_ui_after_stop)

    def stop_processing(self):
        """Stop the current main processing"""
        if hasattr(self, 'stop_event') and self.stop_event:
            self.stop_event.set()

    def update_ui_after_stop(self):
        """Update UI after stopping processing (user action or webcam end)"""
        self.stop_button.configure(state=ctk.DISABLED)
        self.start_button.configure(state=ctk.NORMAL)
        self.status_var.set("Processing Stopped.")
        
        # Enable save button if there's something to save
        if self.mode_var.get() == "video" and self.last_processed_video_temp_path and os.path.exists(self.last_processed_video_temp_path):
            # Typically, if stopped, last_processed_video_temp_path would be None or point to an incomplete file.
            # Let's assume we don't want to save incomplete stopped videos.
            self.save_button.configure(state=ctk.DISABLED) # Disable for video if stopped
        elif self.mode_var.get() in ["image", "webcam"] and hasattr(self, 'current_frame_to_save') and self.current_frame_to_save is not None:
            self.save_button.configure(state=ctk.NORMAL)
        else:
            self.save_button.configure(state=ctk.DISABLED)


    def update_ui_for_file_completion(self):
        """Update UI after image/video processing completes naturally."""
        self.stop_button.configure(state=ctk.DISABLED)
        self.start_button.configure(state=ctk.NORMAL)
        self.status_var.set("Done.")
        
        if self.mode_var.get() == "video" and self.last_processed_video_temp_path and os.path.exists(self.last_processed_video_temp_path):
            self.save_button.configure(state=ctk.NORMAL)
        elif self.mode_var.get() == "image" and hasattr(self, 'current_frame_to_save') and self.current_frame_to_save is not None:
            self.save_button.configure(state=ctk.NORMAL)
        else:
            self.save_button.configure(state=ctk.DISABLED) # Should not happen if completed successfully with output

    def on_threshold_strvar_change(self, *args):
        """Update threshold_var when threshold_strvar changes"""
        try:
            value = float(self.threshold_strvar.get())
            if 0.0 <= value <= 1.0:
                self.threshold_var.set(value)
            else:
                # Optionally, revert to last valid or show error
                self.threshold_strvar.set(f"{self.threshold_var.get():.2f}")
        except ValueError:
            # Revert to last valid if input is not a number
            self.threshold_strvar.set(f"{self.threshold_var.get():.2f}")

    def on_yolo_conf_strvar_change(self, *args):
        """Update yolo_conf_var when yolo_conf_strvar changes"""
        try:
            value = float(self.yolo_conf_strvar.get())
            if 0.0 <= value <= 1.0:
                self.yolo_conf_var.set(value)
            else:
                self.yolo_conf_strvar.set(f"{self.yolo_conf_var.get():.2f}")
        except ValueError:
            self.yolo_conf_strvar.set(f"{self.yolo_conf_var.get():.2f}")

    def on_yolo_iou_strvar_change(self, *args):
        """Update yolo_iou_var when yolo_iou_strvar changes"""
        try:
            value = float(self.yolo_iou_strvar.get())
            if 0.0 <= value <= 1.0:
                self.yolo_iou_var.set(value)
            else:
                self.yolo_iou_strvar.set(f"{self.yolo_iou_var.get():.2f}")
        except ValueError:
            self.yolo_iou_strvar.set(f"{self.yolo_iou_var.get():.2f}")

    def display_frame(self, frame):
        """Display the given frame in the preview area."""
        if frame is None:
            self.preview_label.configure(image=None)
            self.preview_label.configure(text="No video stream")
            self.current_frame_to_save = None
            return

        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)

        # Resize image to fit preview area while maintaining aspect ratio
        img_width, img_height = pil_image.size
        ratio = min(self.preview_width / img_width, self.preview_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage and update label
        self.photo_image = ImageTk.PhotoImage(image=pil_image)
        self.preview_label.configure(image=self.photo_image, text="")
        self.preview_label_image = self.photo_image  # Keep a reference!
        self.current_frame_to_save = frame

    def decrement_threshold(self):
        """Decrement the threshold value"""
        new_value = max(self.threshold_var.get() - 0.1, 0.0)
        self.threshold_var.set(new_value)

    def increment_threshold(self):
        """Increment the threshold value"""
        new_value = min(self.threshold_var.get() + 0.1, 1.0)
        self.threshold_var.set(new_value)
    
    def decrement_yolo_conf(self):
        """Decrease YOLO confidence by 0.1"""
        val = max(self.yolo_conf_var.get() - 0.1, 0.0)
        self.yolo_conf_var.set(val)

    def increment_yolo_conf(self):
        """Increase YOLO confidence by 0.1"""
        val = min(self.yolo_conf_var.get() + 0.1, 1.0)
        self.yolo_conf_var.set(val)

    def decrement_yolo_iou(self):
        """Decrease YOLO IoU by 0.1"""
        val = max(self.yolo_iou_var.get() - 0.1, 0.0)
        self.yolo_iou_var.set(val)

    def increment_yolo_iou(self):
        """Increase YOLO IoU by 0.1"""
        val = min(self.yolo_iou_var.get() + 0.1, 1.0)
        self.yolo_iou_var.set(val)
