import os
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import cv2
from ultralytics import YOLO
import sqlite3
from datetime import datetime, timedelta
import os
import threading
import time
from PIL import Image
from detection_logic import InstanceDetector, ComplianceChecker, SnapshotManager
from person_tracker import PersonViolationTracker, build_detections, draw_violation_box
from database import Database
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'N/A'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

model = None
camera = None
streaming = False
stream_lock = threading.Lock()
dev_mode = False
db = Database()
instance_detector = InstanceDetector()
compliance_checker = ComplianceChecker()
snapshot_manager = SnapshotManager()
person_tracker = PersonViolationTracker()

SETTINGS_FILE = 'settings.json'
DEFAULT_SETTINGS = {
    'required_ppe': {
        'helmet': True,
        'vest': True,
        'mask': False
    },
    'non_compliance_delay': 3,
    'instance_reset_timeout': 5,
    'detection_mode': 'single',  # 'single' or 'multi'
    'person_cooldown': 60  # seconds before the SAME person is recorded again
}

def load_settings():
    """Load settings from file or return defaults"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading settings: {e}")
    return DEFAULT_SETTINGS.copy()

def save_settings_to_file(settings):
    """Save settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

current_settings = load_settings()

def load_model():
    """Load YOLO model"""
    global model
    model_path = '../Model-Training/Outputs/runs/detect/yolov8s_ppe_css_80_epochs/weights/best.pt'
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        model = YOLO('yolov8n.pt')

def generate_frames():
    """Generate video frames with instance detection"""
    global camera, streaming, model, current_settings
    
    last_alert_time = 0
    ALERT_COOLDOWN = current_settings['non_compliance_delay']
    last_snapshot_time = 0
    SNAPSHOT_INTERVAL = current_settings['instance_reset_timeout']
    
    while streaming:
        try:
            with stream_lock:
                if camera is None or not camera.isOpened():
                    break
                
                success, frame = camera.read()
            
            if not success or frame is None:
                time.sleep(0.1)
                continue
            
            results = model.track(frame, persist=True, verbose=False)
            result = results[0]

            detections = build_detections(result)
            current_time = time.time()
            person_results = person_tracker.process(detections, now=current_time)

            annotated_frame = result.plot()

            any_violation = False
            for pr in person_results:
                if pr['is_violation']:
                    any_violation = True
                    draw_violation_box(annotated_frame, pr)

            has_person = len(person_results) > 0
            is_compliant = not any_violation

            # one record per person per cooldown window, saved with the red box on that person
            for pr in person_results:
                if pr['should_record']:
                    snapshot = annotated_frame.copy()
                    snapshot_path = snapshot_manager.save_snapshot(snapshot, pr['record_id'])
                    if snapshot_path:
                        db.log_instance_snapshot(
                            instance_id=pr['record_id'],
                            missing_ppe=pr['missing_ppe'],
                            detected_ppe=pr['detected_ppe'],
                            snapshot_path=snapshot_path,
                            person_id=pr['track_id']
                        )
                        db.log_alert(
                            "NON_COMPLIANCE",
                            f"Person {pr['track_id']} missing: {', '.join(pr['missing_ppe'])}",
                            snapshot_path
                        )
                        socketio.emit('alert', {
                            'timestamp': datetime.now().isoformat(),
                            'type': 'NON_COMPLIANCE',
                            'description': f"Person {pr['track_id']} missing {', '.join(pr['missing_ppe'])}",
                            'person_id': pr['track_id'],
                            'record_id': pr['record_id'],
                            'dev_mode': dev_mode
                        })

            if any_violation:
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], annotated_frame.shape[0]),
                             (0, 0, 255), 20)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.8, overlay, 0.2, 0)

                alert_text = "DEV MODE - TESTING" if dev_mode else "NON-COMPLIANT DETECTED"
                cv2.putText(annotated_frame, alert_text,
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            socketio.emit('detection_update', {
                'timestamp': datetime.now().isoformat(),
                'is_compliant': is_compliant,
                'person_count': len(person_results),
                'violations': [
                    {'person_id': pr['track_id'], 'missing_ppe': pr['missing_ppe']}
                    for pr in person_results if pr['is_violation']
                ],
                'detection_details': {
                    'has_person': has_person,
                    'is_compliant': is_compliant,
                    'missing_ppe': sorted({m for pr in person_results for m in pr['missing_ppe']}),
                    'detected_ppe': sorted({d for pr in person_results for d in pr['detected_ppe']})
                },
                'dev_mode': dev_mode
            })
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        except GeneratorExit:
            break
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/review')
def review():
    """Review page for instances and snapshots"""
    return render_template('review.html')

@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html')

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings"""
    return jsonify(current_settings)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings"""
    global current_settings
    try:
        new_settings = request.json
        current_settings = new_settings

        instance_detector.update_settings(new_settings)
        person_tracker.update_settings(new_settings)
        
        if save_settings_to_file(new_settings):
            return jsonify({'status': 'success', 'message': 'Settings saved'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save settings to file'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    """Reset settings to defaults"""
    global current_settings
    try:
        current_settings = DEFAULT_SETTINGS.copy()
        instance_detector.update_settings(current_settings)
        person_tracker.update_settings(current_settings)
        
        if save_settings_to_file(current_settings):
            return jsonify({'status': 'success', 'message': 'Settings reset to defaults', 'settings': current_settings})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save settings to file'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    try:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed: {e}")
        return '', 500

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start video streaming"""
    global camera, streaming, model
    
    try:
        if model is None:
            load_model()
        
        with stream_lock:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        streaming = True
        return jsonify({'status': 'success', 'message': 'Stream started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop video streaming"""
    global camera, streaming
    
    try:
        streaming = False
        time.sleep(0.3)
        
        with stream_lock:
            if camera is not None:
                camera.release()
                camera = None
        
        return jsonify({'status': 'success', 'message': 'Stream stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/toggle_dev_mode', methods=['POST'])
def toggle_dev_mode():
    """Toggle dev/testing mode"""
    global dev_mode
    dev_mode = not dev_mode
    return jsonify({'status': 'success', 'dev_mode': dev_mode, 
                   'message': f'Dev mode {"enabled" if dev_mode else "disabled"}'})

@app.route('/stats')
def get_stats():
    """Get detection statistics"""
    stats = db.get_statistics()
    stats['dev_mode'] = dev_mode
    return jsonify(stats)

@app.route('/api/instances')
def get_instances():
    """Get all detection instances"""
    try:
        sort_by = request.args.get('sort', 'first_detected')
        sort_order = request.args.get('order', 'desc')
        
        instances = db.get_all_instances(sort_by, sort_order)
        return jsonify(instances)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/instance/<instance_id>/snapshots')
def get_instance_snapshots(instance_id):
    """Get all snapshots for a specific instance"""
    try:
        data = db.get_instance_snapshots(instance_id)
        if data:
            return jsonify(data)
        return jsonify({'error': 'Instance not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_snapshot/<path:filename>')
def download_snapshot(filename):
    """Download snapshot"""
    try:
        filepath = filename if os.path.isabs(filename) else os.path.join('snapshots', filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    """Serve snapshot for viewing"""
    try:
        filepath = os.path.join('snapshots', filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/jpeg')
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_instance/<instance_id>', methods=['DELETE'])
def delete_instance(instance_id):
    """Delete an instance and its snapshot"""
    try:
        success = db.delete_instance(instance_id)
        if success:
            return jsonify({'status': 'success', 'message': 'Instance deleted'})
        return jsonify({'error': 'Failed to delete instance'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        db.init_db()
        load_model()
        instance_detector.update_settings(current_settings)
        person_tracker.update_settings(current_settings)
        socketio.run(app, debug=True, host='localhost', port=3333, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"Fatal error starting app: {e}")
