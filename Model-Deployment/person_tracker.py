"""
Per-person PPE violation tracking.

Assigns a persistent id to every person (via the model's built-in tracker),
associates the negative PPE detections (NO-Hardhat / NO-Safety Vest / NO-Mask)
with the person they overlap, and produces one violation "record" per person
per cooldown window (default 60s). The same person is not recorded again until
the cooldown elapses; if they are still violating after it, a new record opens.

Shared by the Flask app (Model-Deployment/app.py) and the video script
(detect_video.py).
"""
import time
from datetime import datetime

# required-PPE key  ->  model class names (lower-case) that mean "missing" / "present"
PPE_NEGATIVE = {
    'helmet': ['no-hardhat', 'no-helmet'],
    'vest':   ['no-safety vest', 'no-vest'],
    'mask':   ['no-mask', 'no-face-mask'],
}
PPE_POSITIVE = {
    'helmet': ['hardhat', 'helmet'],
    'vest':   ['safety vest', 'vest'],
    'mask':   ['mask'],
}


def _center_inside(box, person):
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return person[0] <= cx <= person[2] and person[1] <= cy <= person[3]


def _overlap_over_box_area(box, person):
    """Intersection area divided by the (PPE) box area — how much of the item sits on the person."""
    x1 = max(box[0], person[0]); y1 = max(box[1], person[1])
    x2 = min(box[2], person[2]); y2 = min(box[3], person[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    return inter / area


def _associated(item_box, person_box):
    """Is this PPE detection on this person?"""
    return _center_inside(item_box, person_box) or _overlap_over_box_area(item_box, person_box) > 0.2


class PersonViolationTracker:
    """Tracks each person's compliance and decides when to record a new violation."""

    def __init__(self, cooldown=60, required_ppe=None):
        self.cooldown = cooldown
        self.required_ppe = required_ppe or {'helmet': True, 'vest': True, 'mask': False}
        self._last_record_time = {}   # track_id -> last time a record was saved
        self._serial = 0

    def update_settings(self, settings):
        if not settings:
            return
        if 'required_ppe' in settings:
            self.required_ppe = settings['required_ppe']
        if 'person_cooldown' in settings:
            try:
                self.cooldown = float(settings['person_cooldown'])
            except (TypeError, ValueError):
                pass

    def _evaluate_person(self, person_box, ppe_items):
        """Return (missing_ppe, detected_ppe) for one person based on nearby PPE detections."""
        missing, detected = [], []
        for ppe_type, is_required in self.required_ppe.items():
            if not is_required:
                continue
            neg_names = PPE_NEGATIVE.get(ppe_type, [])
            pos_names = PPE_POSITIVE.get(ppe_type, [])
            has_neg = any(
                _associated(it['bbox'], person_box) and any(n in it['class'] for n in neg_names)
                for it in ppe_items
            )
            has_pos = any(
                _associated(it['bbox'], person_box)
                and not it['class'].startswith('no-')
                and any(p in it['class'] for p in pos_names)
                for it in ppe_items
            )
            if has_neg:
                missing.append(ppe_type)
            elif has_pos:
                detected.append(ppe_type)
        return missing, detected

    def process(self, detections, now=None):
        """
        detections: list of dicts {class (lower-case), bbox [x1,y1,x2,y2], track_id or None}.
        Returns one result dict per tracked person:
            {track_id, bbox, missing_ppe, detected_ppe, is_violation, should_record, record_id}
        """
        now = time.time() if now is None else now
        persons = [d for d in detections
                   if d['class'] == 'person' and d.get('track_id') is not None]
        ppe_items = [d for d in detections if d['class'] != 'person']

        results = []
        for p in persons:
            tid = p['track_id']
            missing, detected = self._evaluate_person(p['bbox'], ppe_items)
            is_violation = len(missing) > 0
            should_record = False
            record_id = None

            if is_violation:
                last = self._last_record_time.get(tid)
                if last is None or (now - last) >= self.cooldown:
                    should_record = True
                    self._last_record_time[tid] = now
                    self._serial += 1
                    record_id = f"{datetime.now().strftime('%m_%d_%Y')}_P{tid}_{self._serial}"

            results.append({
                'track_id': tid,
                'bbox': p['bbox'],
                'missing_ppe': missing,
                'detected_ppe': detected,
                'is_violation': is_violation,
                'should_record': should_record,
                'record_id': record_id,
            })
        return results


def draw_violation_box(frame, person, color=(0, 0, 255)):
    """Draw a red box + label around a violating person (full-frame evidence)."""
    import cv2
    x1, y1, x2, y2 = [int(v) for v in person['bbox']]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    tid = person.get('track_id')
    label = f"ID {tid} | missing: {', '.join(person['missing_ppe'])}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ytxt = y1 - 10 if y1 - 10 > th else y1 + th + 10
    cv2.rectangle(frame, (x1, ytxt - th - 6), (x1 + tw + 6, ytxt + 4), color, -1)
    cv2.putText(frame, label, (x1 + 3, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def build_detections(result):
    """Convert an ultralytics result (from model.track) into the detection dicts this module expects."""
    detections = []
    boxes = getattr(result, 'boxes', None)
    if boxes is None:
        return detections
    names = result.names
    ids = boxes.id
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        track_id = None
        if ids is not None:
            try:
                track_id = int(ids[i].item())
            except Exception:
                track_id = None
        detections.append({
            'class': names[cls].lower(),
            'bbox': [x1, y1, x2, y2],
            'confidence': float(box.conf[0]),
            'track_id': track_id,
        })
    return detections
