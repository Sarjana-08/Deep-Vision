#!/usr/bin/env python3
"""
Flask-Based Real-Time Crowd Monitoring Dashboard
Features:
- Live video feed with crowd detection
- Real-time density heatmaps
- Alert management
- Email configuration
- Historical data visualization
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import cv2
import numpy as np
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from collections import deque

# Import custom modules
from crowd_detector import AdvancedCrowdDetector
from email_alerts import EmailAlertManager

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'threshold': 500,
    'alert_enabled': True,
    'fps_target': 5,
    'smooth_window': 5,
}

# ============================================================================
# GLOBAL STATE
# ============================================================================

class MonitoringState:
    def __init__(self):
        self.detector = AdvancedCrowdDetector(smooth_window=CONFIG['smooth_window'])
        self.alert_manager = EmailAlertManager()
        self.latest_frame = None
        self.latest_heatmap = None
        self.latest_count = 0
        self.latest_detection = None
        self.alert_active = False
        self.alert_sent_time = None
        self.video_source = None
        self.is_recording = False
        self.history = deque(maxlen=1000)
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_time = 0

state = MonitoringState()

# ============================================================================
# WEBCAM/VIDEO CAPTURE THREAD
# ============================================================================

def capture_video_feed(source=0):
    """Capture video and process for crowd detection"""
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"✗ Cannot open video source: {source}")
        return
    
    frame_skip = 0
    skip_count = int(30 / CONFIG['fps_target'])
    
    print(f"✓ Video capture started (source: {source})")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Failed to read frame")
            break
        
        # Resize for processing
        frame = cv2.resize(frame, (640, 480))
        state.latest_frame = frame.copy()
        
        frame_skip += 1
        if frame_skip >= skip_count:
            frame_skip = 0
            
            process_start = time.time()
            
            # Detect crowd
            detection = state.detector.estimate_crowd_count(frame)
            state.latest_detection = detection
            state.latest_count = detection['count']
            
            # Create heatmap
            state.latest_heatmap = state.detector.create_density_heatmap(
                frame, state.latest_count
            )
            
            # Check threshold for alerts
            if CONFIG['alert_enabled'] and state.latest_count > CONFIG['threshold']:
                state.alert_active = True
                
                # Send email alerts
                current_time = time.time()
                if state.alert_sent_time is None or (current_time - state.alert_sent_time) > 300:
                    state.alert_manager.send_alerts_batch(
                        state.latest_count,
                        CONFIG['threshold']
                    )
                    state.alert_sent_time = current_time
            else:
                state.alert_active = False
            
            # Store history
            state.history.append({
                'timestamp': datetime.now().isoformat(),
                'count': state.latest_count,
                'threshold': CONFIG['threshold'],
                'alert': state.alert_active,
                'confidence': detection['confidence']
            })
            
            # Calculate FPS
            state.frame_count += 1
            state.processing_time = time.time() - process_start
            elapsed = time.time() - state.start_time
            state.fps = state.frame_count / elapsed if elapsed > 0 else 0

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Get/update configuration"""
    if request.method == 'GET':
        return jsonify({
            'threshold': CONFIG['threshold'],
            'alert_enabled': CONFIG['alert_enabled'],
            'fps_target': CONFIG['fps_target'],
            'smooth_window': CONFIG['smooth_window']
        })
    
    elif request.method == 'POST':
        data = request.json
        
        if 'threshold' in data:
            CONFIG['threshold'] = float(data['threshold'])
        if 'alert_enabled' in data:
            CONFIG['alert_enabled'] = bool(data['alert_enabled'])
        if 'fps_target' in data:
            CONFIG['fps_target'] = int(data['fps_target'])
        
        return jsonify({'status': 'success', 'config': CONFIG})

@app.route('/api/status')
def api_status():
    """Get current monitoring status"""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'crowd_count': round(state.latest_count, 2),
        'threshold': CONFIG['threshold'],
        'alert_active': state.alert_active,
        'confidence': round(state.latest_detection['confidence'] * 100) if state.latest_detection else 0,
        'fps': round(state.fps, 1),
        'processing_time_ms': round(state.processing_time * 1000, 1)
    })

@app.route('/api/frame')
def api_frame():
    """Get latest frame with annotations"""
    if state.latest_frame is None:
        return jsonify({'error': 'No frame available'}), 404
    
    frame = state.latest_frame.copy()
    
    # Add annotations
    color = (0, 0, 255) if state.alert_active else (0, 255, 0)
    
    cv2.putText(frame, f"Count: {state.latest_count:.0f}", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(frame, f"Threshold: {CONFIG['threshold']}", (20, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if state.alert_active:
        cv2.putText(frame, "ALERT!", (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    cv2.putText(frame, f"FPS: {state.fps:.1f}", (20, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = buffer.tobytes()
    
    return Response(frame_b64, mimetype='image/jpeg')

@app.route('/api/heatmap')
def api_heatmap():
    """Get density heatmap"""
    if state.latest_heatmap is None:
        return jsonify({'error': 'No heatmap available'}), 404
    
    _, buffer = cv2.imencode('.jpg', state.latest_heatmap)
    heatmap_b64 = buffer.tobytes()
    
    return Response(heatmap_b64, mimetype='image/jpeg')

@app.route('/api/history')
def api_history():
    """Get historical data"""
    limit = request.args.get('limit', 100, type=int)
    return jsonify(list(state.history)[-limit:])

@app.route('/api/detection-details')
def api_detection_details():
    """Get detailed detection information"""
    if state.latest_detection is None:
        return jsonify({'error': 'No detection data'}), 404
    
    return jsonify({
        'count': round(state.latest_detection['count'], 2),
        'cascade_count': state.latest_detection['cascade_count'],
        'contour_count': state.latest_detection['contour_count'],
        'density_count': round(state.latest_detection['density_count'], 2),
        'model_count': round(state.latest_detection['model_count'], 2) if state.latest_detection['model_count'] else None,
        'confidence': round(state.latest_detection['confidence'] * 100, 1),
        'timestamp': state.latest_detection['timestamp'].isoformat()
    })

# ============================================================================
# EMAIL ALERTS API
# ============================================================================

@app.route('/api/alerts/recipients', methods=['GET', 'POST', 'DELETE'])
def api_recipients():
    """Manage alert recipients"""
    if request.method == 'GET':
        recipients = state.alert_manager.get_recipients(enabled_only=False)
        return jsonify([{'email': r[0], 'name': r[1]} for r in recipients])
    
    elif request.method == 'POST':
        data = request.json
        email = data.get('email')
        name = data.get('name', '')
        
        if not email:
            return jsonify({'error': 'Email required'}), 400
        
        success = state.alert_manager.add_recipient(email, name)
        return jsonify({
            'status': 'success' if success else 'error',
            'message': f"Recipient {'added' if success else 'failed to add'}"
        })
    
    elif request.method == 'DELETE':
        data = request.json
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'Email required'}), 400
        
        success = state.alert_manager.remove_recipient(email)
        return jsonify({
            'status': 'success' if success else 'error',
            'message': f"Recipient {'removed' if success else 'failed to remove'}"
        })

@app.route('/api/alerts/toggle/<email>/<int:enabled>', methods=['PUT'])
def api_toggle_recipient(email, enabled):
    """Toggle recipient alerts"""
    success = state.alert_manager.toggle_recipient(email, bool(enabled))
    return jsonify({
        'status': 'success' if success else 'error'
    })

@app.route('/api/alerts/history')
def api_alert_history():
    """Get alert history"""
    limit = request.args.get('limit', 50, type=int)
    history = state.alert_manager.get_alert_history(limit)
    
    return jsonify([{
        'email': h[0],
        'count': round(h[1], 1),
        'threshold': round(h[2], 1),
        'timestamp': h[3],
        'status': h[4]
    } for h in history])

@app.route('/api/alerts/test/<email>', methods=['POST'])
def api_test_alert(email):
    """Send test alert to recipient"""
    success = state.alert_manager.send_alert_email(email, 750, 500)
    return jsonify({
        'status': 'sent' if success else 'failed',
        'message': f"Test alert {'sent' if success else 'failed'} to {email}"
    })

@app.route('/api/alerts/config', methods=['GET', 'POST'])
def api_smtp_config():
    """Get/update SMTP configuration"""
    if request.method == 'GET':
        return jsonify(state.alert_manager.smtp_config)
    
    elif request.method == 'POST':
        data = request.json
        
        try:
            # Update config file
            config_path = Path('smtp_config.json')
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Reload
            state.alert_manager.smtp_config = data
            
            return jsonify({
                'status': 'success',
                'message': 'SMTP configuration updated'
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400

# ============================================================================
# STARTUP & INITIALIZATION
# ============================================================================

def initialize_app():
    """Initialize application"""
    print("\n" + "="*80)
    print("FLASK CROWD MONITORING DASHBOARD - INITIALIZATION")
    print("="*80 + "\n")
    
    print("[1/3] Loading crowd detector...")
    print(f"  ✓ Detector initialized with {CONFIG['smooth_window']} frame smoothing")
    
    print("\n[2/3] Loading email alert system...")
    recipients = state.alert_manager.get_recipients()
    print(f"  ✓ Email alerts ready ({len(recipients)} recipients)")
    
    print("\n[3/3] Starting video capture thread...")
    capture_thread = threading.Thread(
        target=capture_video_feed,
        args=(0,),  # 0 = default webcam
        daemon=True
    )
    capture_thread.start()
    
    print("\n" + "="*80)
    print("✓ DASHBOARD READY")
    print("="*80)
    print("\nAccess dashboard at: http://localhost:5000")
    print("API Documentation at: http://localhost:5000/api/docs\n")

if __name__ == '__main__':
    initialize_app()
    app.run(debug=False, host='localhost', port=5000, threaded=True)
