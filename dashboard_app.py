#!/usr/bin/env python3
"""
Production-Grade Streamlit Real-Time Crowd Monitoring Dashboard
Deployment-Ready with Database & Persistent Storage
Features:
- Live video feed with crowd detection
- Real-time density heatmaps
- Alert management with database persistence
- Email configuration
- Historical data visualization
- Advanced statistics and analytics
- Configuration persistence
"""

import streamlit as st
import cv2
import numpy as np
import json
import threading
import time
import sqlite3
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
from contextlib import contextmanager

# Import custom modules
from crowd_detector import AdvancedCrowdDetector
from email_alerts import EmailAlertManager

# Try to import yt-dlp for YouTube support
try:
    import yt_dlp
    YOUTUBE_SUPPORT = True
except ImportError:
    YOUTUBE_SUPPORT = False

# ============================================================================
# DATABASE SETUP
# ============================================================================

DB_PATH = Path(__file__).parent / 'crowd_monitor.db'

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_database():
    """Initialize database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Detection records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                crowd_count REAL,
                cascade_count INTEGER,
                contour_count INTEGER,
                density_count REAL,
                confidence REAL,
                method TEXT,
                threshold REAL,
                alert_active BOOLEAN DEFAULT 0
            )
        ''')
        
        # Alert recipients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_recipients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE,
                name TEXT,
                enabled BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_alert DATETIME
            )
        ''')
        
        # Alert history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipient_email TEXT,
                crowd_count REAL,
                threshold REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT
            )
        ''')
        
        # Configuration table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS configuration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()

def save_detection_record(crowd_count, cascade_count, contour_count, density_count, confidence, method, threshold, alert_active):
    """Save detection record to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detection_records 
                (crowd_count, cascade_count, contour_count, density_count, confidence, method, threshold, alert_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (crowd_count, cascade_count, contour_count, density_count, confidence, method, threshold, alert_active))
            conn.commit()
    except Exception as e:
        print(f"Database error: {e}")

def save_alert_history(recipient_email, crowd_count, threshold, status):
    """Log alert to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alert_history (recipient_email, crowd_count, threshold, status)
                VALUES (?, ?, ?, ?)
            ''', (recipient_email, crowd_count, threshold, status))
            
            # Update last alert time
            cursor.execute('''
                UPDATE alert_recipients SET last_alert = CURRENT_TIMESTAMP
                WHERE email = ?
            ''', (recipient_email,))
            
            conn.commit()
    except Exception as e:
        print(f"Alert logging error: {e}")

def get_detection_history(hours=24, limit=1000):
    """Retrieve detection history from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            cursor.execute('''
                SELECT * FROM detection_records
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (cutoff_time.isoformat(), limit))
            
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]
    except Exception as e:
        print(f"History retrieval error: {e}")
        return []

def get_alert_history(hours=24, limit=50):
    """Retrieve alert history from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            cursor.execute('''
                SELECT * FROM alert_history
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (cutoff_time.isoformat(), limit))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        print(f"Alert history error: {e}")
        return []

def save_config(key, value):
    """Save configuration to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO configuration (key, value)
                VALUES (?, ?)
            ''', (key, str(value)))
            conn.commit()
    except Exception as e:
        print(f"Config save error: {e}")

def load_config(key, default=None):
    """Load configuration from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM configuration WHERE key = ?', (key,))
            row = cursor.fetchone()
            if row:
                return row[0]
    except Exception as e:
        print(f"Config load error: {e}")
    return default

# ============================================================================
# VIDEO SOURCE HANDLERS
# ============================================================================

def get_youtube_video_url(youtube_url):
    """Extract downloadable URL from YouTube link"""
    if not YOUTUBE_SUPPORT:
        return None
    
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info.get('url')
    except Exception as e:
        st.error(f"❌ YouTube error: {e}")
        return None

def process_uploaded_video(uploaded_file):
    """Save uploaded video file to temporary location and return path"""
    try:
        # Create temp directory
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save uploaded file
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return temp_path
    except Exception as e:
        st.error(f"❌ Upload error: {e}")
        return None

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Crowd Monitoring Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database on startup
init_database()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'detector' not in st.session_state:
    st.session_state.detector = AdvancedCrowdDetector(smooth_window=5)
    st.session_state.alert_manager = EmailAlertManager()
    st.session_state.latest_frame = None
    st.session_state.latest_heatmap = None
    st.session_state.latest_count = 0
    st.session_state.latest_detection = None
    st.session_state.alert_active = False
    st.session_state.alert_sent_time = None
    st.session_state.history = deque(maxlen=1000)
    st.session_state.fps = 0
    st.session_state.frame_count = 0
    st.session_state.start_time = time.time()
    st.session_state.processing_time = 0
    
    # Load persisted settings from database
    st.session_state.threshold = int(float(load_config('threshold', 500)))
    st.session_state.alert_enabled = load_config('alert_enabled', 'true').lower() == 'true'
    st.session_state.fps_target = int(load_config('fps_target', 5))
    st.session_state.smooth_window = int(load_config('smooth_window', 5))
    st.session_state.show_recipients = False
    st.session_state.show_analytics = False
    st.session_state.last_alert_email_time = {}

# ============================================================================
# VIDEO CAPTURE & PROCESSING
# ============================================================================

def capture_and_process_frame(cap, detector, config):
    """Capture and process a single frame"""
    ret, frame = cap.read()
    
    if not ret:
        return None
    
    # Resize for processing
    frame = cv2.resize(frame, (640, 480))
    
    process_start = time.time()
    
    # Detect crowd
    detection = detector.estimate_crowd_count(frame)
    count = detection['count']
    
    # Create heatmap
    heatmap = detector.create_density_heatmap(frame, detection)
    
    # Check threshold for alerts
    alert_active = config['alert_enabled'] and count > config['threshold']
    
    processing_time = time.time() - process_start
    
    return {
        'frame': frame,
        'heatmap': heatmap,
        'count': count,
        'detection': detection,
        'alert_active': alert_active,
        'processing_time': processing_time
    }

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.title("⚙️ Configuration")

# Threshold settings
old_threshold = st.session_state.threshold
st.session_state.threshold = st.sidebar.slider(
    "Crowd Threshold",
    min_value=100,
    max_value=2000,
    value=st.session_state.threshold,
    step=50,
    help="Alert will trigger when crowd count exceeds this value"
)
if st.session_state.threshold != old_threshold:
    save_config('threshold', st.session_state.threshold)

# Alert toggle
old_alert = st.session_state.alert_enabled
st.session_state.alert_enabled = st.sidebar.checkbox(
    "Enable Alerts",
    value=st.session_state.alert_enabled,
    help="Enable/disable email notifications"
)
if st.session_state.alert_enabled != old_alert:
    save_config('alert_enabled', st.session_state.alert_enabled)

# FPS target
old_fps = st.session_state.fps_target
st.session_state.fps_target = st.sidebar.slider(
    "Target FPS",
    min_value=1,
    max_value=30,
    value=st.session_state.fps_target,
    step=1,
    help="Target frames per second for processing"
)
if st.session_state.fps_target != old_fps:
    save_config('fps_target', st.session_state.fps_target)

# Smooth window
old_smooth = st.session_state.smooth_window
st.session_state.smooth_window = st.sidebar.slider(
    "Smoothing Window",
    min_value=1,
    max_value=20,
    value=st.session_state.smooth_window,
    step=1,
    help="Number of frames to average for smooth detection"
)
if st.session_state.smooth_window != old_smooth:
    save_config('smooth_window', st.session_state.smooth_window)
    st.session_state.detector = AdvancedCrowdDetector(smooth_window=st.session_state.smooth_window)

st.sidebar.markdown("---")

# Email configuration
st.sidebar.subheader("📧 Email Alerts")

if st.sidebar.button("📋 Manage Recipients"):
    st.session_state.show_recipients = not st.session_state.show_recipients

# Analytics
st.sidebar.subheader("📊 Analytics")

if st.sidebar.button("📈 View Analytics"):
    st.session_state.show_analytics = not st.session_state.show_analytics

# Alert history in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistics")

# Get statistics
history_data = get_detection_history()
if history_data:
    peak_count = max([h['crowd_count'] for h in history_data], default=0)
    confidence_values = [h['confidence'] for h in history_data]
    avg_confidence = np.mean(confidence_values) if confidence_values else 0
    total_alerts = sum(1 for h in history_data if h['alert_active'])
else:
    peak_count = 0
    avg_confidence = 0
    total_alerts = 0

st.sidebar.metric("Peak Count", round(peak_count, 1))
st.sidebar.metric("Total Alerts", total_alerts)
st.sidebar.metric("Avg Confidence", round(avg_confidence * 100, 1))

# Uptime
uptime_seconds = time.time() - st.session_state.start_time
uptime_minutes = int(uptime_seconds / 60)
uptime_hours = uptime_minutes // 60
uptime_mins = uptime_minutes % 60
st.sidebar.metric("Uptime", f"{uptime_hours}h {uptime_mins}m")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.title("👥 Real-Time Crowd Monitoring Dashboard")

# Create placeholders for real-time updates
col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_count = st.empty()
with col2:
    metric_threshold = st.empty()
with col3:
    metric_fps = st.empty()
with col4:
    metric_processing = st.empty()

# Video display area
st.markdown("---")
col_feed, col_heatmap = st.columns(2)

with col_feed:
    st.subheader("📹 Live Feed")
    frame_placeholder = st.empty()

with col_heatmap:
    st.subheader("🔥 Density Heatmap")
    heatmap_placeholder = st.empty()

# Alert indicator
st.markdown("---")
alert_placeholder = st.empty()

# History chart
st.markdown("---")
st.subheader("📈 Detection History")
history_placeholder = st.empty()

# Details section
st.markdown("---")
st.subheader("🔍 Detection Details")
details_col1, details_col2, details_col3 = st.columns(3)

with details_col1:
    detail_cascade = st.empty()
with details_col2:
    detail_contour = st.empty()
with details_col3:
    detail_model = st.empty()

# ============================================================================
# VIDEO PROCESSING LOOP
# ============================================================================

st.subheader("🎬 Video Source Selection")

# Video source tabs
tab_webcam, tab_upload, tab_youtube = st.tabs(["📹 Webcam", "📤 Upload Video", "🎥 YouTube"])

with tab_webcam:
    col_source = st.columns([1])
    with col_source[0]:
        video_source = st.selectbox(
            "Select Webcam",
            options=[0, 1, 2],
            index=0,
            help="Select webcam device (0 = default)"
        )
    
    if st.button("▶️ Start Webcam Feed", key="start_webcam"):
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            st.error(f"❌ Cannot open webcam: {video_source}")
        else:
            st.success("✓ Webcam started")
            
            frame_skip = 0
            skip_count = int(30 / st.session_state.fps_target)
            
            stop_monitoring = st.button("⏹️ Stop Webcam", key="stop_webcam")
            
            while not stop_monitoring:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Failed to read frame")
                    break
                
                # Resize for processing
                frame = cv2.resize(frame, (640, 480))
                st.session_state.latest_frame = frame.copy()
                
                frame_skip += 1
                if frame_skip >= skip_count:
                    frame_skip = 0
                    
                    # Process frame
                    result = capture_and_process_frame(
                        cap,
                        st.session_state.detector,
                        {
                            'threshold': st.session_state.threshold,
                            'alert_enabled': st.session_state.alert_enabled,
                            'fps_target': st.session_state.fps_target
                        }
                    )
                    
                    if result:
                        st.session_state.latest_frame = result['frame']
                        st.session_state.latest_heatmap = result['heatmap']
                        st.session_state.latest_count = result['count']
                        st.session_state.latest_detection = result['detection']
                        st.session_state.alert_active = result['alert_active']
                        st.session_state.processing_time = result['processing_time']
                        
                        # Send alerts if needed
                        if st.session_state.alert_active:
                            current_time = time.time()
                            try:
                                from email_alerts import EmailAlertManager
                                alert_mgr = st.session_state.alert_manager
                                
                                # Get all enabled recipients from database
                                with get_db_connection() as conn:
                                    cursor = conn.cursor()
                                    cursor.execute('SELECT email, name FROM alert_recipients WHERE enabled = 1')
                                    recipients = cursor.fetchall()
                                
                                for recipient in recipients:
                                    email = recipient[0]
                                    last_alert_time = st.session_state.last_alert_email_time.get(email, 0)
                                    
                                    # Send alert only every 5 minutes per recipient
                                    if current_time - last_alert_time > 300:
                                        success = alert_mgr.send_alert_email(
                                            email,
                                            st.session_state.latest_count,
                                            st.session_state.threshold
                                        )
                                        
                                        # Log the alert
                                        save_alert_history(
                                            email,
                                            st.session_state.latest_count,
                                            st.session_state.threshold,
                                            'sent' if success else 'failed'
                                        )
                                        
                                        if success:
                                            st.session_state.last_alert_email_time[email] = current_time
                            except Exception as e:
                                print(f"Alert error: {e}")
                        
                        # Store history
                        st.session_state.history.append({
                            'timestamp': datetime.now().isoformat(),
                            'count': st.session_state.latest_count,
                            'threshold': st.session_state.threshold,
                            'alert': st.session_state.alert_active,
                            'confidence': result['detection']['confidence']
                        })
                        
                        # Save to database
                        save_detection_record(
                            st.session_state.latest_count,
                            result['detection']['cascade_count'],
                            result['detection']['contour_count'],
                            result['detection']['density_count'],
                            result['detection']['confidence'],
                            result['detection']['method'],
                            st.session_state.threshold,
                            st.session_state.alert_active
                        )
                        
                        # Update metrics
                        with metric_count:
                            st.metric("Crowd Count", f"{st.session_state.latest_count:.0f}")
                        
                        with metric_threshold:
                            st.metric("Threshold", st.session_state.threshold)
                        
                        st.session_state.frame_count += 1
                        elapsed = time.time() - st.session_state.start_time
                        st.session_state.fps = st.session_state.frame_count / elapsed if elapsed > 0 else 0
                        
                        with metric_fps:
                            st.metric("FPS", f"{st.session_state.fps:.1f}")
                        
                        with metric_processing:
                            st.metric("Processing (ms)", f"{st.session_state.processing_time * 1000:.1f}")
                        
                        # Display frame with annotations
                        frame_display = st.session_state.latest_frame.copy()
                        color = (0, 0, 255) if st.session_state.alert_active else (0, 255, 0)
                        cv2.putText(frame_display, f"Count: {st.session_state.latest_count:.0f}", (20, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                        cv2.putText(frame_display, f"Threshold: {st.session_state.threshold}", (20, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if st.session_state.alert_active:
                            cv2.putText(frame_display, "ALERT!", (20, 150),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cv2.putText(frame_display, f"FPS: {st.session_state.fps:.1f}", (20, 200),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        
                        frame_display_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                        with frame_placeholder:
                            st.image(frame_display_rgb, use_container_width=True)
                        
                        # Display heatmap
                        if st.session_state.latest_heatmap is not None:
                            heatmap_rgb = cv2.cvtColor(st.session_state.latest_heatmap, cv2.COLOR_BGR2RGB)
                            with heatmap_placeholder:
                                st.image(heatmap_rgb, use_container_width=True)
                        
                        # Alert indicator
                        with alert_placeholder:
                            if st.session_state.alert_active:
                                st.error("🚨 ALERT: Crowd threshold exceeded!")
                            else:
                                st.success("✓ Normal conditions")
                        
                        # History chart
                        if len(st.session_state.history) > 0:
                            history_df = pd.DataFrame(list(st.session_state.history))
                            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                            
                            with history_placeholder:
                                st.line_chart(history_df[['count', 'threshold']].rename(
                                    columns={'count': 'Crowd Count', 'threshold': 'Threshold'}
                                ))
                        
                        # Detection details
                        if st.session_state.latest_detection:
                            with detail_cascade:
                                st.metric("Cascade Count", st.session_state.latest_detection.get('cascade_count', 0))
                            with detail_contour:
                                st.metric("Contour Count", st.session_state.latest_detection.get('contour_count', 0))
                            with detail_model:
                                model_count = st.session_state.latest_detection.get('model_count')
                                st.metric("Model Count", f"{model_count:.0f}" if model_count else "N/A")
                        
                        time.sleep(0.033)  # ~30 FPS
            
            cap.release()
            st.info("✓ Webcam stopped")

with tab_upload:
    st.write("Upload a video file to analyze crowd detection")
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'])
    
    if uploaded_file is not None:
        # Save uploaded file
        temp_path = process_uploaded_video(uploaded_file)
        
        if temp_path and st.button("▶️ Start Video Analysis", key="start_upload"):
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                st.error("❌ Cannot open uploaded video")
            else:
                st.success("✓ Video started")
                
                frame_skip = 0
                skip_count = int(30 / st.session_state.fps_target)
                
                stop_monitoring = st.button("⏹️ Stop Analysis", key="stop_upload")
                
                while not stop_monitoring:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.info("✓ Video analysis complete")
                        break
                    
                    frame = cv2.resize(frame, (640, 480))
                    st.session_state.latest_frame = frame.copy()
                    
                    frame_skip += 1
                    if frame_skip >= skip_count:
                        frame_skip = 0
                        
                        result = capture_and_process_frame(
                            cap,
                            st.session_state.detector,
                            {
                                'threshold': st.session_state.threshold,
                                'alert_enabled': st.session_state.alert_enabled,
                                'fps_target': st.session_state.fps_target
                            }
                        )
                        
                        if result:
                            st.session_state.latest_frame = result['frame']
                            st.session_state.latest_heatmap = result['heatmap']
                            st.session_state.latest_count = result['count']
                            st.session_state.latest_detection = result['detection']
                            st.session_state.alert_active = result['alert_active']
                            st.session_state.processing_time = result['processing_time']
                            
                            # Store history
                            st.session_state.history.append({
                                'timestamp': datetime.now().isoformat(),
                                'count': st.session_state.latest_count,
                                'threshold': st.session_state.threshold,
                                'alert': st.session_state.alert_active,
                                'confidence': result['detection']['confidence']
                            })
                            
                            # Save to database
                            save_detection_record(
                                st.session_state.latest_count,
                                result['detection']['cascade_count'],
                                result['detection']['contour_count'],
                                result['detection']['density_count'],
                                result['detection']['confidence'],
                                result['detection']['method'],
                                st.session_state.threshold,
                                st.session_state.alert_active
                            )
                            
                            with metric_count:
                                st.metric("Crowd Count", f"{st.session_state.latest_count:.0f}")
                            
                            with metric_threshold:
                                st.metric("Threshold", st.session_state.threshold)
                            
                            st.session_state.frame_count += 1
                            elapsed = time.time() - st.session_state.start_time
                            st.session_state.fps = st.session_state.frame_count / elapsed if elapsed > 0 else 0
                            
                            with metric_fps:
                                st.metric("FPS", f"{st.session_state.fps:.1f}")
                            
                            with metric_processing:
                                st.metric("Processing (ms)", f"{st.session_state.processing_time * 1000:.1f}")
                            
                            frame_display = st.session_state.latest_frame.copy()
                            color = (0, 0, 255) if st.session_state.alert_active else (0, 255, 0)
                            cv2.putText(frame_display, f"Count: {st.session_state.latest_count:.0f}", (20, 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                            cv2.putText(frame_display, f"Threshold: {st.session_state.threshold}", (20, 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            if st.session_state.alert_active:
                                cv2.putText(frame_display, "ALERT!", (20, 150),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                            
                            frame_display_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                            with frame_placeholder:
                                st.image(frame_display_rgb, use_container_width=True)
                            
                            if st.session_state.latest_heatmap is not None:
                                heatmap_rgb = cv2.cvtColor(st.session_state.latest_heatmap, cv2.COLOR_BGR2RGB)
                                with heatmap_placeholder:
                                    st.image(heatmap_rgb, use_container_width=True)
                            
                            with alert_placeholder:
                                if st.session_state.alert_active:
                                    st.error("🚨 ALERT: Crowd threshold exceeded!")
                                else:
                                    st.success("✓ Normal conditions")
                            
                            time.sleep(0.033)
                
                cap.release()
                try:
                    os.remove(temp_path)
                except:
                    pass
                st.info("✓ Video analysis stopped")
    else:
        st.info("👆 Upload a video file to begin analysis")

with tab_youtube:
    if YOUTUBE_SUPPORT:
        st.write("Enter a YouTube link to analyze crowd detection")
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste YouTube link (must be public)"
        )
        
        if youtube_url and st.button("▶️ Start YouTube Analysis", key="start_youtube"):
            with st.spinner("Fetching YouTube video..."):
                video_url = get_youtube_video_url(youtube_url)
            
            if video_url:
                cap = cv2.VideoCapture(video_url)
                
                if not cap.isOpened():
                    st.error("❌ Cannot open YouTube video")
                else:
                    st.success("✓ YouTube video started")
                    
                    frame_skip = 0
                    skip_count = int(30 / st.session_state.fps_target)
                    
                    stop_monitoring = st.button("⏹️ Stop YouTube", key="stop_youtube")
                    
                    while not stop_monitoring:
                        ret, frame = cap.read()
                        
                        if not ret:
                            st.info("✓ YouTube video analysis complete")
                            break
                        
                        frame = cv2.resize(frame, (640, 480))
                        st.session_state.latest_frame = frame.copy()
                        
                        frame_skip += 1
                        if frame_skip >= skip_count:
                            frame_skip = 0
                            
                            result = capture_and_process_frame(
                                cap,
                                st.session_state.detector,
                                {
                                    'threshold': st.session_state.threshold,
                                    'alert_enabled': st.session_state.alert_enabled,
                                    'fps_target': st.session_state.fps_target
                                }
                            )
                            
                            if result:
                                st.session_state.latest_frame = result['frame']
                                st.session_state.latest_heatmap = result['heatmap']
                                st.session_state.latest_count = result['count']
                                st.session_state.latest_detection = result['detection']
                                st.session_state.alert_active = result['alert_active']
                                st.session_state.processing_time = result['processing_time']
                                
                                st.session_state.history.append({
                                    'timestamp': datetime.now().isoformat(),
                                    'count': st.session_state.latest_count,
                                    'threshold': st.session_state.threshold,
                                    'alert': st.session_state.alert_active,
                                    'confidence': result['detection']['confidence']
                                })
                                
                                save_detection_record(
                                    st.session_state.latest_count,
                                    result['detection']['cascade_count'],
                                    result['detection']['contour_count'],
                                    result['detection']['density_count'],
                                    result['detection']['confidence'],
                                    result['detection']['method'],
                                    st.session_state.threshold,
                                    st.session_state.alert_active
                                )
                                
                                with metric_count:
                                    st.metric("Crowd Count", f"{st.session_state.latest_count:.0f}")
                                
                                with metric_threshold:
                                    st.metric("Threshold", st.session_state.threshold)
                                
                                st.session_state.frame_count += 1
                                elapsed = time.time() - st.session_state.start_time
                                st.session_state.fps = st.session_state.frame_count / elapsed if elapsed > 0 else 0
                                
                                with metric_fps:
                                    st.metric("FPS", f"{st.session_state.fps:.1f}")
                                
                                with metric_processing:
                                    st.metric("Processing (ms)", f"{st.session_state.processing_time * 1000:.1f}")
                                
                                frame_display = st.session_state.latest_frame.copy()
                                color = (0, 0, 255) if st.session_state.alert_active else (0, 255, 0)
                                cv2.putText(frame_display, f"Count: {st.session_state.latest_count:.0f}", (20, 50),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                                cv2.putText(frame_display, f"Threshold: {st.session_state.threshold}", (20, 100),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                if st.session_state.alert_active:
                                    cv2.putText(frame_display, "ALERT!", (20, 150),
                                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                                
                                frame_display_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                                with frame_placeholder:
                                    st.image(frame_display_rgb, use_container_width=True)
                                
                                if st.session_state.latest_heatmap is not None:
                                    heatmap_rgb = cv2.cvtColor(st.session_state.latest_heatmap, cv2.COLOR_BGR2RGB)
                                    with heatmap_placeholder:
                                        st.image(heatmap_rgb, use_container_width=True)
                                
                                with alert_placeholder:
                                    if st.session_state.alert_active:
                                        st.error("🚨 ALERT: Crowd threshold exceeded!")
                                    else:
                                        st.success("✓ Normal conditions")
                                
                                time.sleep(0.033)
                    
                    cap.release()
                    st.info("✓ YouTube analysis stopped")
    else:
        st.warning("⚠️ YouTube support requires yt-dlp. Install with: pip install yt-dlp")
        st.info("After installing, refresh the page to enable YouTube support")

# ============================================================================
# MANAGE RECIPIENTS TAB
# ============================================================================

if st.session_state.get('show_recipients', False):
    st.markdown("---")
    st.subheader("📧 Manage Alert Recipients")
    
    # Add new recipient
    col_email, col_name, col_add = st.columns([2, 2, 1])
    
    with col_email:
        new_email = st.text_input("Email Address", key="new_email")
    with col_name:
        new_name = st.text_input("Name", key="new_name")
    with col_add:
        if st.button("➕ Add"):
            if new_email and '@' in new_email:
                try:
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            'INSERT INTO alert_recipients (email, name) VALUES (?, ?)',
                            (new_email, new_name or new_email)
                        )
                        conn.commit()
                    st.success(f"✓ Added {new_email}")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error(f"❌ Email already exists")
            else:
                st.error("❌ Valid email required")
    
    # List recipients
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, email, name, enabled FROM alert_recipients ORDER BY created_at DESC')
            recipients = cursor.fetchall()
        
        if recipients:
            st.write("**Current Recipients:**")
            for recipient in recipients:
                rec_id, email, name, enabled = recipient
                col_email_list, col_name_list, col_toggle, col_test, col_remove = st.columns([2, 2, 1, 1, 1])
                
                with col_email_list:
                    st.write(email)
                with col_name_list:
                    st.write(name or "—")
                with col_toggle:
                    new_enabled = st.checkbox(
                        "Active",
                        value=bool(enabled),
                        key=f"toggle_{rec_id}"
                    )
                    if new_enabled != bool(enabled):
                        with get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                'UPDATE alert_recipients SET enabled = ? WHERE id = ?',
                                (new_enabled, rec_id)
                            )
                            conn.commit()
                        st.rerun()
                
                with col_test:
                    if st.button("✉️ Test", key=f"test_{rec_id}"):
                        success = st.session_state.alert_manager.send_alert_email(email, 750, 500)
                        if success:
                            save_alert_history(email, 750, 500, 'sent')
                            st.success("Test sent!")
                        else:
                            st.error("Test failed!")
                
                with col_remove:
                    if st.button("🗑️", key=f"remove_{rec_id}"):
                        with get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute('DELETE FROM alert_recipients WHERE id = ?', (rec_id,))
                            conn.commit()
                        st.success("Removed!")
                        st.rerun()
        else:
            st.info("No recipients configured yet")
    
    except Exception as e:
        st.error(f"Error managing recipients: {e}")

# ============================================================================
# ANALYTICS TAB
# ============================================================================

if st.session_state.get('show_analytics', False):
    st.markdown("---")
    st.subheader("📊 Advanced Analytics & History")
    
    # Time period selection
    col_period, col_limit = st.columns(2)
    
    with col_period:
        period_hours = st.selectbox(
            "Time Period",
            options=[1, 6, 12, 24, 48, 72],
            format_func=lambda x: f"Last {x} hours"
        )
    
    with col_limit:
        limit = st.selectbox(
            "Limit Records",
            options=[50, 100, 250, 500, 1000],
            format_func=lambda x: f"{x} records"
        )
    
    # Retrieve historical data
    history_data = get_detection_history(hours=period_hours, limit=limit)
    
    if history_data:
        # Convert to DataFrame for analysis
        df = pd.DataFrame(history_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Summary statistics
        st.write("**Summary Statistics**")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Peak Count", f"{df['crowd_count'].max():.0f}")
        with col3:
            st.metric("Avg Count", f"{df['crowd_count'].mean():.0f}")
        with col4:
            st.metric("Min Count", f"{df['crowd_count'].min():.0f}")
        with col5:
            alert_count = sum(1 for _, row in df.iterrows() if row['alert_active'])
            st.metric("Alerts Triggered", alert_count)
        
        # Crowd count trend
        st.write("**Crowd Count Trend**")
        chart_data = df[['timestamp', 'crowd_count', 'threshold']].set_index('timestamp')
        st.line_chart(chart_data)
        
        # Detection methods distribution
        st.write("**Detection Methods Distribution**")
        col_cascade, col_contour, col_density = st.columns(3)
        
        with col_cascade:
            cascade_avg = df['cascade_count'].mean()
            st.metric("Avg Cascade Detections", f"{cascade_avg:.0f}")
        
        with col_contour:
            contour_avg = df['contour_count'].mean()
            st.metric("Avg Contour Detections", f"{contour_avg:.0f}")
        
        with col_density:
            density_avg = df['density_count'].mean()
            st.metric("Avg Density Count", f"{density_avg:.1f}")
        
        # Confidence distribution
        st.write("**Confidence Distribution**")
        confidence_data = df[['timestamp', 'confidence']].set_index('timestamp')
        st.line_chart(confidence_data)
        
        # Alert history
        st.write("**Alert History**")
        alert_history = get_alert_history(hours=period_hours, limit=limit)
        
        if alert_history:
            alert_df = pd.DataFrame(alert_history)
            alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
            
            # Group by recipient and status
            st.write("**Alerts by Recipient**")
            recipient_stats = alert_df.groupby('recipient_email').agg({
                'crowd_count': ['count', 'mean', 'max'],
                'status': lambda x: (x == 'sent').sum()
            }).round(1)
            
            st.dataframe(recipient_stats)
            
            # Recent alerts
            st.write("**Recent Alerts (Last 20)**")
            recent_alerts = alert_df.sort_values('timestamp', ascending=False).head(20)
            st.dataframe(recent_alerts[['timestamp', 'recipient_email', 'crowd_count', 'threshold', 'status']])
        else:
            st.info("No alert history yet")
        
        # Export data
        st.write("**Export Data**")
        col_csv, col_json = st.columns(2)
        
        with col_csv:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col_json:
            json_data = df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        st.info("No historical data available yet. Start monitoring to collect data.")
