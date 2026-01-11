import cv2
import os
from ultralytics import YOLO
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class NotificationSystem:
    def __init__(self, email_config):
        self.sender_email = email_config['sender_email']
        self.sender_password = email_config['sender_password']
        self.police_email = email_config['police_email']
        self.ambulance_email = email_config['ambulance_email']

    def send_email(self, to_email, subject, body):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            print(f"Alert email sent successfully to {to_email}")
            return True
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            return False

    def notify_police(self, location, severity, incident_details, timestamp):
        subject = f"Traffic Accident Alert - Severity Level: {severity}"
        body = f"""
TRAFFIC ACCIDENT DETECTED

Location: {location}
Time: {timestamp}
Severity Level: {severity}/5

Incident Details:
{incident_details}

This is an automated alert from the Traffic Accident Detection System.
Immediate response may be required.
"""
        return self.send_email(self.police_email, subject, body)

    def notify_ambulance(self, location, severity, incident_details, timestamp):
        subject = f"URGENT: Medical Assistance Required - Severity Level: {severity}"
        body = f"""
URGENT: SEVERE TRAFFIC ACCIDENT DETECTED

Location: {location}
Time: {timestamp}
Severity Level: {severity}/5

Incident Details:
{incident_details}

IMMEDIATE MEDICAL ASSISTANCE REQUIRED
This is an automated alert from the Traffic Accident Detection System.
"""
        return self.send_email(self.ambulance_email, subject, body)

class SeverityAnalyzer:
    def __init__(self):
        self.last_detections = []
        self.max_history = 15  # Increased history for better pattern recognition
        self.detection_threshold = 0.25  # Further lowered threshold
        self.consecutive_detections = 0
        self.detection_window = []  # Rolling window of recent detections
        self.window_size = 5  # Number of frames to consider

    def analyze_severity(self, detection, frame):
        confidence = float(detection.conf[0])
        box = detection.xyxy[0].cpu().numpy()
        
        # Track consecutive detections for reliability
        self.detection_window.append(1)
        if len(self.detection_window) > self.window_size:
            self.detection_window.pop(0)
        
        frame_area = frame.shape[0] * frame.shape[1]
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        size_factor = box_area / frame_area

        # Enhanced motion analysis
        motion_factors = []
        if self.last_detections:
            for prev_box in self.last_detections[-3:]:  # Consider last 3 frames
                motion_factors.append(self._calculate_motion(box, prev_box))
        
        motion_factor = max(motion_factors) if motion_factors else 0
        
        # Multiple detection zones
        severity_zones = self._analyze_zones(frame, box)
        
        self.last_detections.append(box)
        if len(self.last_detections) > self.max_history:
            self.last_detections.pop(0)

        # Enhanced severity calculation
        severity_score = self._calculate_severity_score(
            confidence, 
            size_factor, 
            motion_factor,
            severity_zones,
            sum(self.detection_window) / len(self.detection_window)  # Detection stability
        )
        
        details = self._generate_incident_details(
            confidence, 
            size_factor, 
            motion_factor, 
            severity_score,
            severity_zones
        )
        
        return severity_score, details

    def _analyze_zones(self, frame, box):
        # Divide frame into zones and check impact areas
        height, width = frame.shape[:2]
        zones = {
            'center': 0,
            'edges': 0,
            'critical': 0
        }
        
        box_center_x = (box[0] + box[2]) / 2
        box_center_y = (box[1] + box[3]) / 2
        
        # Center zone check
        center_zone = (
            width * 0.3 <= box_center_x <= width * 0.7 and
            height * 0.3 <= box_center_y <= height * 0.7
        )
        zones['center'] = 1 if center_zone else 0
        
        # Edge zones check
        edge_zone = (
            box_center_x <= width * 0.1 or
            box_center_x >= width * 0.9 or
            box_center_y <= height * 0.1 or
            box_center_y >= height * 0.9
        )
        zones['edges'] = 1 if edge_zone else 0
        
        # Critical zones (intersections, crosswalks, etc.)
        # This would need to be customized based on the camera location
        critical_zone = (
            width * 0.4 <= box_center_x <= width * 0.6 and
            height * 0.4 <= box_center_y <= height * 0.6
        )
        zones['critical'] = 1 if critical_zone else 0
        
        return zones

    def _calculate_severity_score(self, confidence, size_factor, motion_factor, zones, detection_stability):
        # Enhanced severity calculation with multiple factors
        base_severity = (
            confidence * 0.25 +  # Reduced confidence weight
            size_factor * 0.35 +  # Increased size importance
            min(motion_factor / 100, 1.0) * 0.25 +  # Motion consideration
            (zones['center'] * 0.05 +  # Zone weights
             zones['edges'] * 0.05 +
             zones['critical'] * 0.05)
        )
        
        # Apply detection stability factor
        stability_bonus = detection_stability * 0.15
        
        # Scale to 1-5 with increased sensitivity
        severity = base_severity + stability_bonus
        return min(max(round(severity * 7), 1), 5)  # Increased multiplier for higher sensitivity

    def _calculate_motion(self, current_box, previous_box):
        # Enhanced motion calculation
        current_center = ((current_box[0] + current_box[2])/2, (current_box[1] + current_box[3])/2)
        previous_center = ((previous_box[0] + previous_box[2])/2, (previous_box[1] + previous_box[3])/2)
        
        distance = np.sqrt((current_center[0] - previous_center[0])**2 + 
                         (current_center[1] - previous_center[1])**2)
        
        # Add size change detection
        current_size = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        previous_size = (previous_box[2] - previous_box[0]) * (previous_box[3] - previous_box[1])
        size_change = abs(current_size - previous_size) / max(current_size, previous_size)
        
        return distance * (1 + size_change)  # Combine motion and size change

    def _generate_incident_details(self, confidence, size_factor, motion_factor, severity_score, zones):
        size_percentage = size_factor * 100
        motion_intensity = min(motion_factor / 100, 1.0) * 100
        
        zone_analysis = []
        if zones['center']: zone_analysis.append("Central Area Impact")
        if zones['edges']: zone_analysis.append("Edge Area Impact")
        if zones['critical']: zone_analysis.append("Critical Zone Impact")
        
        details = f"""
Severity Assessment Details:
- Detection Confidence: {confidence:.2%}
- Accident Size: {size_percentage:.1f}% of frame
- Motion Intensity: {motion_intensity:.1f}%
- Impact Zones: {', '.join(zone_analysis) if zone_analysis else 'Standard Zone'}
- Overall Severity Score: {severity_score}/5

Response Priority:
- {'CRITICAL - Immediate Response Required' if severity_score >= 4 else 'HIGH PRIORITY' if severity_score >= 3 else 'Standard Response Protocol'}
- {'Multiple Units Required' if size_percentage > 25 else 'Standard Unit Response'}
- {'Additional Medical Support Advised' if motion_intensity > 70 else ''}

Detection Confidence:
- {'WARNING: Low Confidence Detection' if confidence < 0.4 else 'High Confidence Detection'}
"""
        return details

class CollisionDetector:
    def __init__(self):
        self.previous_frames = []
        self.max_frames = 5
        self.intersection_threshold = 0.2
        self.motion_threshold = 10
        self.size_change_threshold = 0.3

    def update(self, current_boxes, frame):
        """
        Update detector with current frame's vehicle boxes
        Returns: (collision_detected, collision_info)
        """
        collision_detected = False
        collision_info = None
        
        # Store current frame info
        self.previous_frames.append(current_boxes)
        if len(self.previous_frames) > self.max_frames:
            self.previous_frames.pop(0)
            
        if len(self.previous_frames) >= 2:
            # Check for collisions between vehicles
            collision_detected, collision_info = self._detect_collision(current_boxes, frame)
            
        return collision_detected, collision_info
    
    def _detect_collision(self, current_boxes, frame):
        """
        Check for collisions between all pairs of vehicles
        """
        for i, box1 in enumerate(current_boxes):
            for j, box2 in enumerate(current_boxes[i+1:], i+1):
                # Check multiple collision conditions
                intersection = self._check_intersection(box1, box2)
                motion = self._check_motion(box1, box2)
                size_change = self._check_size_change(box1, box2)
                
                # Combine evidence of collision
                collision_score = (
                    intersection * 0.4 +
                    motion * 0.4 +
                    size_change * 0.2
                )
                
                if collision_score > 0.6:  # Threshold for collision detection
                    severity = self._calculate_collision_severity(box1, box2, frame)
                    return True, {
                        'boxes': (box1, box2),
                        'severity': severity,
                        'score': collision_score
                    }
        return False, None

    def _check_intersection(self, box1, box2):
        """Calculate intersection over union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = intersection / (box1_area + box2_area - intersection)
        return iou

    def _check_motion(self, box1, box2):
        """Check for rapid motion between vehicles"""
        if len(self.previous_frames) < 2:
            return 0.0
            
        prev_boxes = self.previous_frames[-2]
        
        # Find corresponding boxes in previous frame
        prev_box1 = self._find_closest_box(box1, prev_boxes)
        prev_box2 = self._find_closest_box(box2, prev_boxes)
        
        if prev_box1 is None or prev_box2 is None:
            return 0.0
            
        # Calculate relative motion
        motion1 = self._calculate_motion(box1, prev_box1)
        motion2 = self._calculate_motion(box2, prev_box2)
        
        relative_motion = abs(motion1 - motion2) / self.motion_threshold
        return min(relative_motion, 1.0)

    def _check_size_change(self, box1, box2):
        """Check for sudden changes in vehicle size (indicating impact)"""
        if len(self.previous_frames) < 2:
            return 0.0
            
        prev_boxes = self.previous_frames[-2]
        prev_box1 = self._find_closest_box(box1, prev_boxes)
        prev_box2 = self._find_closest_box(box2, prev_boxes)
        
        if prev_box1 is None or prev_box2 is None:
            return 0.0
            
        # Calculate size changes
        size_change1 = self._calculate_size_change(box1, prev_box1)
        size_change2 = self._calculate_size_change(box2, prev_box2)
        
        return max(size_change1, size_change2)

    def _find_closest_box(self, current_box, previous_boxes):
        """Find the closest box in previous frame"""
        if not previous_boxes:
            return None
            
        min_dist = float('inf')
        closest_box = None
        
        current_center = (
            (current_box[0] + current_box[2]) / 2,
            (current_box[1] + current_box[3]) / 2
        )
        
        for box in previous_boxes:
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            dist = np.sqrt((current_center[0] - center[0])**2 + 
                         (current_center[1] - center[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_box = box
                
        return closest_box

    def _calculate_motion(self, current_box, previous_box):
        """Calculate motion between frames"""
        current_center = (
            (current_box[0] + current_box[2]) / 2,
            (current_box[1] + current_box[3]) / 2
        )
        previous_center = (
            (previous_box[0] + previous_box[2]) / 2,
            (previous_box[1] + previous_box[3]) / 2
        )
        
        return np.sqrt((current_center[0] - previous_center[0])**2 + 
                      (current_center[1] - previous_center[1])**2)

    def _calculate_size_change(self, current_box, previous_box):
        """Calculate change in box size between frames"""
        current_size = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        previous_size = (previous_box[2] - previous_box[0]) * (previous_box[3] - previous_box[1])
        
        size_change = abs(current_size - previous_size) / max(current_size, previous_size)
        return min(size_change / self.size_change_threshold, 1.0)

    def _calculate_collision_severity(self, box1, box2, frame):
        """Calculate collision severity (1-5 scale)"""
        # Calculate multiple severity factors
        intersection = self._check_intersection(box1, box2)
        
        # Size of impact area
        impact_area = (min(box1[2], box2[2]) - max(box1[0], box2[0])) * \
                     (min(box1[3], box2[3]) - max(box1[1], box2[1]))
        frame_area = frame.shape[0] * frame.shape[1]
        size_factor = min(impact_area / frame_area * 10, 1.0)
        
        # Motion severity
        motion_severity = 0
        if len(self.previous_frames) >= 2:
            prev_boxes = self.previous_frames[-2]
            prev_box1 = self._find_closest_box(box1, prev_boxes)
            prev_box2 = self._find_closest_box(box2, prev_boxes)
            if prev_box1 is not None and prev_box2 is not None:
                motion1 = self._calculate_motion(box1, prev_box1)
                motion2 = self._calculate_motion(box2, prev_box2)
                motion_severity = min((motion1 + motion2) / (2 * self.motion_threshold), 1.0)
        
        # Combined severity score
        severity = (intersection * 0.4 +
                   size_factor * 0.3 +
                   motion_severity * 0.3)
        
        # Convert to 1-5 scale
        return min(max(int(severity * 5) + 1, 1), 5)

class AccidentDetectionSystem:
    def __init__(self, model_path, email_config=None):
        # Initialize models
        self.vehicle_model = YOLO('yolov8n.pt')  # Vehicle detection
        self.accident_model = YOLO(model_path)    # Your accident model
        
        # Set detection thresholds
        self.vehicle_model.conf = 0.3
        self.accident_model.conf = 0.25
        
        # Vehicle classes (from COCO dataset)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Initialize collision detector
        self.collision_detector = CollisionDetector()
        
        # Initialize notification system if email config is provided
        self.notification_system = None
        if email_config:
            self.notification_system = NotificationSystem(email_config)
        
        # Debug mode for visualization
        self.debug = True

    def process_video(self, video_path):
        """
        Process video from file
        video_path: path to the video file
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Error: Could not open video file: {video_path}")

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            print(f"Processing video: {video_path}")
            print(f"Total frames: {total_frames}")
            print(f"FPS: {fps}")

            frame_count = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1
                if frame_count % 10 == 0:  # Progress update every 10 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}%")

                # 1. Detect vehicles
                vehicle_results = self.vehicle_model(frame)
                vehicle_boxes = []
                
                # Process vehicle detections
                for result in vehicle_results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        if cls in self.vehicle_classes:
                            box_coords = box.xyxy[0].cpu().numpy()
                            vehicle_boxes.append(box_coords)
                            
                            if self.debug:
                                # Draw vehicle boxes in green
                                cv2.rectangle(frame, 
                                            (int(box_coords[0]), int(box_coords[1])), 
                                            (int(box_coords[2]), int(box_coords[3])), 
                                            (0, 255, 0), 2)
                                vehicle_type = self._get_vehicle_type(cls)
                                cv2.putText(frame, f"{vehicle_type}", 
                                          (int(box_coords[0]), int(box_coords[1])-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 2. Detect collisions
                collision_detected, collision_info = self.collision_detector.update(vehicle_boxes, frame)
                
                # 3. If collision detected, check for accident
                if collision_detected:
                    box1, box2 = collision_info['boxes']
                    severity = collision_info['severity']
                    
                    if self.debug:
                        # Draw collision area in red
                        x1 = min(box1[0], box2[0])
                        y1 = min(box1[1], box2[1])
                        x2 = max(box1[2], box2[2])
                        y2 = max(box1[3], box2[3])
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    (0, 0, 255), 2)
                        cv2.putText(frame, f"Collision - Severity: {severity}/5", 
                                  (int(x1), int(y1)-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    # Get ROI for accident detection
                    roi = self._get_roi(frame, box1, box2)
                    if roi is not None and roi.size > 0:
                        # Run accident detection on ROI
                        accident_results = self.accident_model(roi)
                        for result in accident_results:
                            for box in result.boxes:
                                if float(box.conf[0]) > 0.25:
                                    print(f"\nACCIDENT DETECTED! Severity: {severity}/5")
                                    
                                    # Send notifications if notification system is initialized
                                    if self.notification_system:
                                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        location = self._extract_location_from_path(video_path)
                                        incident_details = (f"Vehicle collision detected with severity level {severity}/5\n"
                                                         f"Frame: {frame_count}/{total_frames}")
                                        
                                        if severity >= 4:
                                            self.notification_system.notify_ambulance(
                                                location, severity, incident_details, timestamp)
                                        self.notification_system.notify_police(
                                            location, severity, incident_details, timestamp)

                # Show frame with processing results
                cv2.imshow('Accident Detection', frame)
                
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing stopped by user")
                    break

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            print("\nVideo processing completed")

    def _get_roi(self, frame, box1, box2):
        """Get region of interest around potential collision"""
        x1 = max(0, int(min(box1[0], box2[0])) - 20)
        y1 = max(0, int(min(box1[1], box2[1])) - 20)
        x2 = min(frame.shape[1], int(max(box1[2], box2[2])) + 20)
        y2 = min(frame.shape[0], int(max(box1[3], box2[3])) + 20)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return frame[y1:y2, x1:x2]
    
    def _get_vehicle_type(self, class_id):
        """Convert class ID to vehicle type string"""
        vehicle_types = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }
        return vehicle_types.get(class_id, "Vehicle")
    
    def _extract_location_from_path(self, video_path):
        """Extract location information from video path"""
        # You can customize this based on your video naming convention
        filename = os.path.basename(video_path)
        location = os.path.splitext(filename)[0]
        return location if location else "Unknown Location"

# Usage
if __name__ == "__main__":
    # Initialize system with path to your trained accident detection model
    system = AccidentDetectionSystem("runs/detect/train/weights/best.pt")
    
class VehicleTracker:
    def __init__(self):
        self.previous_boxes = []
        self.velocity_threshold = 30  # pixels per frame
        self.proximity_threshold = 50  # pixels
        
    def update(self, current_boxes):
        self.previous_boxes = current_boxes
        
    def detect_potential_accidents(self):
        potential_areas = []
        
        # Check for vehicles in close proximity
        for i, box1 in enumerate(self.previous_boxes):
            for j, box2 in enumerate(self.previous_boxes[i+1:], i+1):
                if self._check_proximity(box1, box2):
                    # Create a bounding box that encompasses both vehicles
                    area = self._create_region_of_interest(box1, box2)
                    potential_areas.append(area)
        
        return potential_areas
    
    def _check_proximity(self, box1, box2):
        # Calculate centers
        center1 = ((box1[0] + box1[2])/2, (box1[1] + box1[3])/2)
        center2 = ((box2[0] + box2[2])/2, (box2[1] + box2[3])/2)
        
        # Calculate distance between centers
        distance = np.sqrt((center1[0] - center2[0])**2 + 
                         (center1[1] - center2[1])**2)
        
        return distance < self.proximity_threshold
    
    def _create_region_of_interest(self, box1, box2):
        # Create a region that includes both vehicles plus some margin
        margin = 20  # pixels
        x1 = min(box1[0], box2[0]) - margin
        y1 = min(box1[1], box2[1]) - margin
        x2 = max(box1[2], box2[2]) + margin
        y2 = max(box1[3], box2[3]) + margin
        
        return [x1, y1, x2, y2]

class AccidentDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Accident Detection System")
        self.root.geometry("500x400")
        
        style = ttk.Style()
        style.configure('TButton', padding=10)
        style.configure('TLabel', padding=10, font=('Helvetica', 12))
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, 
                              text="Traffic Accident Detection System",
                              font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=20)
        
        desc_label = ttk.Label(main_frame,
                              text="Select a video file to analyze for traffic accidents\nSystem will automatically alert emergency services when accidents occur",
                              justify=tk.CENTER)
        desc_label.pack(pady=10)
        
        video_btn = ttk.Button(main_frame,
                              text="Select Video File",
                              command=self.select_video)
        video_btn.pack(fill=tk.X, pady=10)
        
        quit_btn = ttk.Button(main_frame,
                             text="Exit System",
                             command=self.root.quit)
        quit_btn.pack(fill=tk.X, pady=10)

    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            self.root.withdraw()
            email_config = {
                'sender_email': 'ganashreear@gmail.com',
                'sender_password': 'bqzm lylc uynl gjwo',
                'police_email': 'amoghmn2004@gmail.com',
                'ambulance_email': 'nulln5780@gmail.com'
            }
            
            system = AccidentDetectionSystem("runs/detect/train/weights/best.pt", email_config)
            system.process_video(file_path)
            self.root.deiconify()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AccidentDetectionGUI()
    app.run()