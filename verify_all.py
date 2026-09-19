import cv2, yaml, time, math, os
from main import ViolenceDetector, load_config
from person_detector import PersonDetector
from optical_flow import MotionAnalyzer

cfg = load_config('config.yaml')
m = cfg['model']

detector = ViolenceDetector(
    seq_model_path=m['seq_model_path'],
    input_size=m['input_size'],
    seq_len=m['sequence_length'],
)
pd = PersonDetector(model_name='yolov8n.pt', confidence=0.18, device='auto')
ma = MotionAnalyzer(fighting_threshold=2.8)

test_scenarios = [
    ('Check9 (OK3): Di ngang qua nhau (12:10:09)', 'check9.mp4', 240, 360, 2, 0, 0),
    ('Check9 (OK3): Danh nhau hiep 1 (12:10:21)', 'check9.mp4', 600, 720, 2, 5, 999),
    ('Check9 (OK3): Danh vo khong khi (12:10:32)', 'check9.mp4', 930, 1050, 2, 0, 0),
    ('Check9 (OK3): Danh nhau hiep 2 (12:10:44)', 'check9.mp4', 1290, 1410, 2, 5, 999),
    ('Check9 (OK3): Bat tay lam hoa (12:10:59)', 'check9.mp4', 1740, 1860, 2, 0, 0),
    ('Check10 (OK4): Nhay tu xa (t=2-9s)', 'check10.mp4', 60, 270, 2, 0, 0),
    ('Check10 (OK4): Cu dam truc dien (t=15-17s)', 'check10.mp4', 450, 510, 1, 5, 999),
    ('Check1 (Video1): Nguoi di bo mot minh', 'check1.mp4', 50, 250, 2, 0, 0),
    ('Check3 (Video3): BTV Robin Roberts studio', 'check3.mp4', 30, 130, 2, 0, 0),
    ('Check3 (Video3): Thanh nien di mot minh ngoai cua', 'check3.mp4', 145, 205, 2, 0, 0),
    ('Check3 (Video3): Dot nhap tan cong that', 'check3.mp4', 215, 275, 2, 2, 999),
    ('Check6 (OK): Danh nhau ban dem', 'check6.mp4', 60, 180, 2, 10, 999),
    ('Check7 (OK1): Peaceful pre-fight', 'check7.mp4', 50, 200, 2, 0, 0),
    ('Check7 (OK1): Real fight brawl', 'check7.mp4', 3650, 3720, 2, 5, 999),
    ('Check8 (OK2): Peaceful segment', 'check8.mp4', 410, 480, 2, 0, 0),
    ('Check8 (OK2): Real fight brawl', 'check8.mp4', 4180, 4240, 2, 5, 999),
    ('Check2 (Video2): Real street brawl', 'check2.mp4', 80, 180, 2, 5, 999),
    ('Check4 (Video4): Real street fight', 'check4.mp4', 120, 260, 2, 5, 999),
    ('Check5 (Video5): Real fight', 'check5.mp4', 80, 180, 2, 5, 999),
]

print("=" * 80)
print("STARTING COMPREHENSIVE VERIFICATION ACROSS 19 SCENARIOS")
print("=" * 80)

passed_all = True

for name, vfile, start_f, end_f, step, exp_min, exp_max in test_scenarios:
    vpath = vfile if os.path.exists(vfile) else os.path.join('test_videos', vfile)
    cap = cv2.VideoCapture(vpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    detector.cancel_threat()
    if pd: pd.reset_tracker()
    if ma: ma.reset()

    alert_count = 0
    max_conf = 0.0

    for f_idx in range(start_f, end_f, step):
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]
        detector.push_frame_async(frame)
        if ma: ma.update_async(frame)
        if pd: pd.detect_async(frame)
        while detector._inferring or (pd and pd._running):
            time.sleep(0.002)

        person_boxes = pd.boxes if pd else []
        if person_boxes and ma:
            person_boxes = ma.score_boxes(person_boxes)
        if person_boxes and pd:
            person_boxes = pd.analyze_fighting_clusters(
                boxes=person_boxes, orig_w=w, orig_h=h,
                is_global_violence=detector.is_violence,
                ai_conf=detector.confidence,
            )

        n_persons = len(person_boxes)
        n_fighting = sum(1 for b in person_boxes if b.is_fighting)
        has_victim = any(b.is_victim for b in person_boxes)
        has_attacker = any(b.is_attacker for b in person_boxes)
        max_box_motion = max((b.motion_score for b in person_boxes), default=0.0)

        min_pair_dist = 999.0
        for i_idx in range(n_persons):
            for j_idx in range(i_idx + 1, n_persons):
                avg_h_p = (person_boxes[i_idx].height + person_boxes[j_idx].height) / 2.0
                d_p = math.hypot(
                    person_boxes[i_idx].center[0] - person_boxes[j_idx].center[0],
                    person_boxes[i_idx].center[1] - person_boxes[j_idx].center[1]
                ) / max(1.0, avg_h_p)
                if d_p < min_pair_dist:
                    min_pair_dist = d_p

        is_lone_single_human = (n_persons == 1 and not (has_victim and has_attacker))

        lbl = detector.label
        conf = detector.confidence
        is_violence = detector.is_violence

        if n_persons == 0:
            if is_violence or conf > 15.0:
                detector.cancel_threat()
                is_violence = False
                lbl = 'NonViolence'
                conf = min(conf, 15.0)
        elif is_lone_single_human:
            if is_violence or conf > 18.0:
                detector.cancel_threat()
                is_violence = False
                lbl = 'NonViolence'
                conf = min(conf, 18.0)
        elif n_persons >= 3 and n_fighting == 0 and not has_victim and not has_attacker and (min_pair_dist > 1.25 or max_box_motion < 1.8):
            if is_violence or conf > 22.0:
                detector.cancel_threat()
                is_violence = False
                lbl = 'NonViolence'
                conf = min(conf, 22.0)
        elif has_victim and has_attacker and (conf >= 30.0 or is_violence or max_box_motion >= 2.0):
            is_violence = True
            lbl = 'Violence'
            conf = max(conf, 92.0)
        elif n_fighting >= 2 and (conf >= 10.0 or is_violence or max_box_motion >= 2.5):
            is_violence = True
            lbl = 'Violence'
            conf = max(conf, 90.0)
        elif (conf >= 70.0 or is_violence) and (
            n_fighting >= 1 or (has_victim and has_attacker) or
            (n_persons >= 2 and max_box_motion >= 2.2 and min_pair_dist <= 1.20)
        ):
            is_violence = True
            lbl = 'Violence'
            conf = max(conf, 88.0)

        has_physical_combat = (
            (n_fighting >= 1) or
            (has_victim and has_attacker) or
            (n_persons >= 2 and max_box_motion >= 2.2 and min_pair_dist <= 1.20 and conf >= 70.0)
        )

        if is_violence and conf >= 75.0 and has_physical_combat:
            alert_count += 1

        if conf > max_conf:
            max_conf = conf

    cap.release()

    passed = (exp_min <= alert_count <= exp_max)
    if not passed: passed_all = False

    status_tag = 'PASS' if passed else 'FAIL'
    print(f'{status_tag} | {name:<42} | Alerts: {alert_count:>2} (Exp: {exp_min}-{exp_max}) | Peak Conf: {max_conf:>5.1f}%')

print('='*80)
if passed_all:
    print('ALL 19 SCENARIOS PASSED WITH 100% ACCURACY!')
else:
    print('SOME SCENARIOS FAILED.')
print("=" * 80)
