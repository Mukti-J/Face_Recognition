import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import cv2
import threading
from PIL import Image, ImageTk

class FaceRecognitionGUI:
    def __init__(self, process_callback=None):
        self.INPUT_DIR = os.path.join(os.getcwd(), 'Input')
        self.OUTPUT_DIR = os.path.join(os.getcwd(), 'Output')
        os.makedirs(self.INPUT_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.source = None
        self.mode = None
        self.process_callback = process_callback
        self.processing_thread = None
        self.webcam_active = False
        
        # Main window setup
        self.root = tk.Tk()
        self.root.title("Face Recognition System")
        self.root.geometry("950x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(self.main_frame, text="Face Recognition and Object Detection", 
                               font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Mode selection frame
        mode_frame = ttk.LabelFrame(self.main_frame, text="Select Mode")
        mode_frame.pack(fill=tk.X, pady=10)
        
        # Mode buttons
        self.mode_var = tk.StringVar(value="webcam")
        
        modes_frame = ttk.Frame(mode_frame)
        modes_frame.pack(pady=10)
        
        ttk.Radiobutton(modes_frame, text="Webcam", variable=self.mode_var, 
                       value="webcam").grid(row=0, column=0, padx=20)
        ttk.Radiobutton(modes_frame, text="Image", variable=self.mode_var, 
                       value="image").grid(row=0, column=1, padx=20)
        ttk.Radiobutton(modes_frame, text="Video", variable=self.mode_var, 
                       value="video").grid(row=0, column=2, padx=20)
        
        # Input file selection
        self.file_frame = ttk.LabelFrame(self.main_frame, text="Input File")
        self.file_frame.pack(fill=tk.X, pady=10)
        
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(self.file_frame, textvariable=self.file_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.X)
        
        browse_button = ttk.Button(self.file_frame, text="Browse", command=self.browse_file)
        browse_button.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(self.main_frame, text="Detection Settings")
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Create settings
        self.threshold_var = tk.DoubleVar(value=0.5)
        # StringVar for formatted threshold display and two-way sync
        self.threshold_strvar = tk.StringVar(value=f"{self.threshold_var.get():.2f}")
        self.threshold_var.trace_add('write', lambda *args: self.threshold_strvar.set(f"{self.threshold_var.get():.2f}"))
        self.threshold_strvar.trace_add('write', self.on_threshold_strvar_change)
        self.yolo_conf_var = tk.DoubleVar(value=0.5)
        # StringVar for formatted YOLO confidence display and sync
        self.yolo_conf_strvar = tk.StringVar(value=f"{self.yolo_conf_var.get():.2f}")
        self.yolo_conf_var.trace_add('write', lambda *args: self.yolo_conf_strvar.set(f"{self.yolo_conf_var.get():.2f}"))
        self.yolo_conf_strvar.trace_add('write', self.on_yolo_conf_strvar_change)
        self.yolo_iou_var = tk.DoubleVar(value=0.45)
        # StringVar for formatted YOLO IoU display and sync
        self.yolo_iou_strvar = tk.StringVar(value=f"{self.yolo_iou_var.get():.2f}")
        self.yolo_iou_var.trace_add('write', lambda *args: self.yolo_iou_strvar.set(f"{self.yolo_iou_var.get():.2f}"))
        self.yolo_iou_strvar.trace_add('write', self.on_yolo_iou_strvar_change)
        
        ttk.Label(settings_frame, text="Face recognition threshold:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        threshold_slider = ttk.Scale(settings_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                    length=200, variable=self.threshold_var)
        threshold_slider.grid(row=0, column=1, pady=5)
        # Entry and +/- buttons for threshold
        threshold_entry = ttk.Entry(settings_frame, textvariable=self.threshold_strvar, width=5)
        threshold_entry.grid(row=0, column=2, padx=(5,2))
        ttk.Button(settings_frame, text="-", width=2, command=self.decrement_threshold).grid(row=0, column=3, padx=2)
        ttk.Button(settings_frame, text="+", width=2, command=self.increment_threshold).grid(row=0, column=4, padx=(2,5))

        ttk.Label(settings_frame, text="YOLO confidence:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        yolo_conf_slider = ttk.Scale(settings_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                    length=200, variable=self.yolo_conf_var)
        yolo_conf_slider.grid(row=1, column=1, pady=5)
        # Entry and +/- buttons for YOLO confidence
        ttk.Entry(settings_frame, textvariable=self.yolo_conf_strvar, width=5).grid(row=1, column=2, padx=(5,2))
        ttk.Button(settings_frame, text="-", width=2, command=self.decrement_yolo_conf).grid(row=1, column=3, padx=2)
        ttk.Button(settings_frame, text="+", width=2, command=self.increment_yolo_conf).grid(row=1, column=4, padx=(2,5))

        ttk.Label(settings_frame, text="YOLO IoU threshold:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        yolo_iou_slider = ttk.Scale(settings_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                   length=200, variable=self.yolo_iou_var)
        yolo_iou_slider.grid(row=2, column=1, pady=5)
        # Entry and +/- buttons for YOLO IoU
        ttk.Entry(settings_frame, textvariable=self.yolo_iou_strvar, width=5).grid(row=2, column=2, padx=(5,2))
        ttk.Button(settings_frame, text="-", width=2, command=self.decrement_yolo_iou).grid(row=2, column=3, padx=2)
        ttk.Button(settings_frame, text="+", width=2, command=self.increment_yolo_iou).grid(row=2, column=4, padx=(2,5))
        
        # Status frame
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, pady=10)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var, 
                                   font=("Arial", 10), foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Control buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Start Processing", 
                                    command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                    command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Sidebar preview area with fixed dimensions
        self.preview_frame = ttk.LabelFrame(self.main_frame, text="Preview", width=400, height=300)
        self.preview_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        # Prevent preview_frame from resizing to children
        self.preview_frame.propagate(False)
        # Dimensions for scaling frames
        self.preview_width = 400
        self.preview_height = 300
        # Label for displaying frames
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # Update UI based on mode selection
        self.mode_var.trace_add('write', self.update_ui_for_mode)
        self.update_ui_for_mode()
        
    def run(self):
        """Run the main application"""
        self.root.mainloop()
        
    def update_ui_for_mode(self, *args):
        """Update UI elements based on the selected mode"""
        mode = self.mode_var.get()
        
        if mode == "webcam":
            self.file_frame.pack_forget()
        else:
            self.file_frame.pack(fill=tk.X, pady=10, after=self.main_frame.winfo_children()[1])
            
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
            self.file_var.set(file_path)
            
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
        return os.path.join(self.OUTPUT_DIR, f'output_{os.path.basename(source_path)}')
    
    def show_result_message(self, output_path):
        """Show message about saved output"""
        print(f'Output saved to {output_path}')
        
    def show_settings_dialog(self, default_threshold=0.5):
        """Show settings dialog for adjusting parameters"""
        # This could be expanded with sliders for confidence and IoU thresholds
        root = tk.Tk()
        root.title("Detection Settings")
        root.geometry("400x250")
        
        settings = {
            'threshold': tk.DoubleVar(value=default_threshold),
            'yolo_conf': tk.DoubleVar(value=0.5),
            'yolo_iou': tk.DoubleVar(value=0.45)
        }
        
        # Create the main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create sliders
        ttk.Label(main_frame, text="Face recognition threshold:").grid(row=0, column=0, sticky=tk.W, pady=5)
        threshold_slider = ttk.Scale(main_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL, 
                                    length=200, variable=settings['threshold'])
        threshold_slider.grid(row=0, column=1, pady=5)
        ttk.Label(main_frame, textvariable=settings['threshold']).grid(row=0, column=2, padx=5)
        
        ttk.Label(main_frame, text="YOLO confidence:").grid(row=1, column=0, sticky=tk.W, pady=5)
        yolo_conf_slider = ttk.Scale(main_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL, 
                                    length=200, variable=settings['yolo_conf'])
        yolo_conf_slider.grid(row=1, column=1, pady=5)
        ttk.Label(main_frame, textvariable=settings['yolo_conf']).grid(row=1, column=2, padx=5)
        
        ttk.Label(main_frame, text="YOLO IoU threshold:").grid(row=2, column=0, sticky=tk.W, pady=5)
        yolo_iou_slider = ttk.Scale(main_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL, 
                                   length=200, variable=settings['yolo_iou'])
        yolo_iou_slider.grid(row=2, column=1, pady=5)
        ttk.Label(main_frame, textvariable=settings['yolo_iou']).grid(row=2, column=2, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="OK", command=root.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Run the dialog
        root.mainloop()
        
        # Return the settings
        return {
            'threshold': settings['threshold'].get(),
            'yolo_conf': settings['yolo_conf'].get(),
            'yolo_iou': settings['yolo_iou'].get()
        }
    
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
        
        # Validate input for image/video modes
        if mode in ["image", "video"] and not self.source:
            messagebox.showerror("Error", "Please select an input file first.")
            return
            
        # Initialize stop event
        self.stop_event = threading.Event()
        
        # Update UI state
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set(f"Processing {mode}...")
        
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
        if self.process_callback:
            self.process_callback(mode, source, settings, display_fn, stop_event)
            # Update UI after processing is complete
            self.root.after(100, self.update_ui_after_stop)
    
    def stop_processing(self):
        """Stop the current processing"""
        # Signal to stop processing
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        if self.processing_thread and self.processing_thread.is_alive():
             # Signal to stop processing
             cv2.destroyAllWindows()
             # Update UI
             self.root.after(100, self.update_ui_after_stop)
    
    def update_ui_after_stop(self):
        """Update UI after stopping processing"""
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Ready")
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_processing()
        self.root.destroy()

    def display_frame(self, frame):
        """Convert frame to ImageTk and display in the GUI"""
        # Convert BGR to RGB and create a PIL image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # Letterbox-scale to fit preview panel
        orig_w, orig_h = img.size
        ratio = min(self.preview_width / orig_w, self.preview_height / orig_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Prepare background and paste centered
        background = Image.new('RGB', (self.preview_width, self.preview_height), (0, 0, 0))
        x_off = (self.preview_width - new_w) // 2
        y_off = (self.preview_height - new_h) // 2
        background.paste(img_resized, (x_off, y_off))
        # Convert to PhotoImage and display
        imgtk = ImageTk.PhotoImage(background)
        # Store reference to prevent GC and update label
        self.imgtk = imgtk
        self.preview_label.config(image=self.imgtk)

    def decrement_threshold(self):
        """Decrement the threshold value"""
        new_value = max(self.threshold_var.get() - 0.1, 0.0)
        self.threshold_var.set(new_value)

    def increment_threshold(self):
        """Increment the threshold value"""
        new_value = min(self.threshold_var.get() + 0.1, 1.0)
        self.threshold_var.set(new_value)

    def on_threshold_strvar_change(self, *args):
        """Update threshold_var when the user edits the entry directly"""
        val = self.threshold_strvar.get()
        try:
            f = float(val)
        except ValueError:
            return
        # clamp between 0.0 and 1.0
        f = max(min(f, 1.0), 0.0)
        # update slider variable without triggering recursion
        if abs(self.threshold_var.get() - f) > 1e-6:
            self.threshold_var.set(f)
    
    def decrement_yolo_conf(self):
        """Decrease YOLO confidence by 0.1"""
        val = max(self.yolo_conf_var.get() - 0.1, 0.0)
        self.yolo_conf_var.set(val)

    def increment_yolo_conf(self):
        """Increase YOLO confidence by 0.1"""
        val = min(self.yolo_conf_var.get() + 0.1, 1.0)
        self.yolo_conf_var.set(val)

    def on_yolo_conf_strvar_change(self, *args):
        """Sync entry edits back to YOLO confidence var"""
        try:
            f = float(self.yolo_conf_strvar.get())
        except ValueError:
            return
        f = max(min(f, 1.0), 0.0)
        if abs(self.yolo_conf_var.get() - f) > 1e-6:
            self.yolo_conf_var.set(f)

    def decrement_yolo_iou(self):
        """Decrease YOLO IoU by 0.1"""
        val = max(self.yolo_iou_var.get() - 0.1, 0.0)
        self.yolo_iou_var.set(val)

    def increment_yolo_iou(self):
        """Increase YOLO IoU by 0.1"""
        val = min(self.yolo_iou_var.get() + 0.1, 1.0)
        self.yolo_iou_var.set(val)

    def on_yolo_iou_strvar_change(self, *args):
        """Sync entry edits back to YOLO IoU var"""
        try:
            f = float(self.yolo_iou_strvar.get())
        except ValueError:
            return
        f = max(min(f, 1.0), 0.0)
        if abs(self.yolo_iou_var.get() - f) > 1e-6:
            self.yolo_iou_var.set(f)
