# main.py
import cv2
import torch
from ultralytics import YOLO
import time
from datetime import datetime
from notifications import NotificationSystem
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

class AccidentDetectionSystem:
    def __init__(self, model_path, notification_config):
        self.model = YOLO(model_path)
        self.notification_system = NotificationSystem(notification_config)
        
    def analyze_severity(self, detection):
        confidence = detection.conf[0]
        
        if confidence > 0.95:
            return 5
        elif confidence > 0.85:
            return 4
        elif confidence > 0.75:
            return 3
        elif confidence > 0.65:
            return 2
        else:
            return 1

    def process_video(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video source!")
            return

        last_notification_time = 0
        notification_cooldown = 30

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = self.model(frame)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls == 0 and conf > 0.5:
                        current_time = time.time()
                        
                        if current_time - last_notification_time > notification_cooldown:
                            severity = self.analyze_severity(box)
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            location = "Camera 1"
                            
                            self.notification_system.notify_police(location, severity, timestamp)
                            
                            if severity > 3:
                                self.notification_system.notify_ambulance(location, severity, timestamp)
                            
                            last_notification_time = current_time

                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Accident {conf:.2f}", (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Accident Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

class AccidentDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Accident Detection System")
        self.root.geometry("400x300")
        
        # Style configuration
        style = ttk.Style()
        style.configure('TButton', padding=10)
        style.configure('TLabel', padding=10, font=('Helvetica', 12))
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Accident Detection System", 
                              font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=20)
        
        # Buttons
        webcam_btn = ttk.Button(main_frame, text="Use Webcam", 
                               command=self.start_webcam)
        webcam_btn.pack(fill=tk.X, pady=10)
        
        video_btn = ttk.Button(main_frame, text="Select Video File", 
                              command=self.select_video)
        video_btn.pack(fill=tk.X, pady=10)
        
        quit_btn = ttk.Button(main_frame, text="Quit", 
                             command=self.root.quit)
        quit_btn.pack(fill=tk.X, pady=10)

    def start_webcam(self):
        self.root.withdraw()  # Hide the main window
        twilio_config = {
            'account_sid': 'your_account_sid',
            'auth_token': 'your_auth_token',
            'from_number': '+1234567890',
            'police_number': '+1234567890',
            'ambulance_number': '+1234567890'
        }
        
        system = AccidentDetectionSystem("runs/detect/train/weights/best.pt", twilio_config)
        system.process_video(0)
        self.root.deiconify()  # Show the main window again

    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            self.root.withdraw()  # Hide the main window
            twilio_config = {
                'account_sid': 'your_account_sid',
                'auth_token': 'your_auth_token',
                'from_number': '+1234567890',
                'police_number': '+1234567890',
                'ambulance_number': '+1234567890'
            }
            
            system = AccidentDetectionSystem("runs/detect/train/weights/best.pt", twilio_config)
            system.process_video(file_path)
            self.root.deiconify()  # Show the main window again

    def run(self):
        self.root.mainloop()

def main():
    app = AccidentDetectionGUI()
    app.run()

if __name__ == "__main__":
    main()