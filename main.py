import os
import cv2
import time
import torch
import argparse
import numpy as np

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

import requests
import threading
from collections import deque
from datetime import datetime

# 전역 변수로 상태 관리
fall_buffer = deque(maxlen=5)  # 최근 5프레임 저장
lying_buffer = deque(maxlen=5)  # 최근 5프레임 저장
last_alert_time = None

#source = '../Data/test_video/test7.mp4'
#source = '../Data/falldata/Home/Videos/video (2).avi'  # hard detect
#source = './Data/falldata/Home/Videos/video_1.avi'
source = 'D:\develop_hanium\hanium.mp4'
out_path = 'D:\develop_hanium\hanium_out.mp4'
#source = 2
api_url = 'http://218.148.5.40:13000'
#api_url = 'http://218.148.5.40:13000'


def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

def check_and_alert_fall(action, confidence, cooldown_sec=10, threshold=0.2):
    """
    낙상 감지 및 알림 함수
    
    Args:
        is_fall: 현재 프레임 낙상 감지 여부 (bool)
        confidence: 현재 프레임 신뢰도 (0.0~1.0)
        cooldown_sec: 알림 후 대기 시간 (초)
        threshold: 평균 신뢰도 임계값
    
    Returns:
        bool: 알림 발송 여부
    """
    global fall_buffer, lying_buffer, last_alert_time
    
    print('check')
    # 1. 현재 프레임 결과 저장
    fall_buffer.append({'fall': action=='Fall', 'conf': confidence})
    lying_buffer.append({'lying': action=='Lying', 'conf': confidence})
    
    # 2. 버퍼가 다 안 찼으면 대기
    if action == 'Fall':
        if len(fall_buffer) < 5:
            return False
        # 3. 쿨다운 체크
        if last_alert_time:
            elapsed = (datetime.now() - last_alert_time).total_seconds()
            if elapsed < cooldown_sec:
                return False
        
        # 4. 연속 감지 및 평균 신뢰도 확인
        all_fall = all(f['fall'] for f in fall_buffer)
        avg_conf = sum(f['conf'] for f in fall_buffer) / 5
        if all_fall and avg_conf >= threshold:
            # 5. API 호출
            print('낙상 감지! 알림 발송 중...1')
            thread = threading.Thread(target=send_alert, args=(avg_conf, action))
            thread.daemon = True
            thread.start()

            last_alert_time = datetime.now()
            fall_buffer.clear()
    else:
        if len(lying_buffer) < 5:
            return False
        # 3. 쿨다운 체크
        if last_alert_time:
            elapsed = (datetime.now() - last_alert_time).total_seconds()
            if elapsed < cooldown_sec:
                return False
        
        # 4. 연속 감지 및 평균 신뢰도 확인
        all_lying = all(f['lying'] for f in lying_buffer)
        avg_conf = sum(f['conf'] for f in lying_buffer) / 5
        if all_lying and avg_conf >= threshold:
            # 5. API 호출
            print('낙상 위험! 알림 발송 중...1')
            thread = threading.Thread(target=send_alert, args=(avg_conf, action))
            thread.daemon = True
            thread.start()

            last_alert_time = datetime.now()
            fall_buffer.clear()
    
    return False
    
    
    

def send_alert(confidence, action):
    print('send_alert')
    """백그라운드 API 호출'
    // 7. 새 사고/경보 등록
        app.post('/api/fall-incident', async (req, res) => {
        const { patient_id, room_id, incident_type, severity, message } = req.body;

        🚨 incident_type: 사고 유형
        이 컬럼은 발생한 낙상 관련 사건의 종류를 구분합니다.
        'accident' (사고): 실제로 환자가 넘어져 다친 실제 낙상 사고가 발생했음을 의미합니다.
        'alert' (알림/경보): 환자가 넘어지려고 하거나 위험한 상황이 감지되어 시스템이 경보를 울린 경우를 의미합니다. 실제 사고로 이어지지는 않았을 수 있습니다.
        'near_miss' (아차사고): 환자가 넘어질 뻔했지만, 직원이나 보조기구 등의 도움으로 다치기 직전에 사고를 피한 경우를 의미합니다.

        🩹 severity: 심각도
        사고가 발생했을 경우, 그 피해 정도가 얼마나 심각한지를 나타냅니다. DEFAULT 'moderate'는 별도로 값을 지정하지 않으면 기본으로 '보통' 등급이 저장된다는 뜻입니다.
        'minor' (경미): 가벼운 찰과상이나 타박상 등 치료가 거의 필요 없는 사소한 부상입니다.
        'moderate' (보통): 치료가 필요한 중간 수준의 부상입니다. (예: 깁스 필요)
        'severe' (심각): 입원 치료가 필요하거나 수술이 필요한 중대한 부상입니다.
        'critical' (위중): 생명이 위독할 수 있는 매우 치명적인 부상입니다.

        🔔 alert_level: 경보 수준
        시스템이 감지한 위험 상황이나 발생한 사고에 대해 어느 정도 수준의 대응이 필요한지를 나타냅니다. DEFAULT 'medium'은 기본값이 '중간' 수준이라는 의미입니다.
        'low' (낮음): 단순 기록이나 참고만 필요한 낮은 수준의 경보입니다.
        'medium' (중간): 담당 간호사나 직원의 확인 및 조치가 필요한 경보입니다.
        'high' (높음): 여러 의료진의 즉각적인 도움이 필요한 긴급 상황 경보입니다.
    """

    accident_json = { 
            'patient_id': '3', 
            'room_id': '8', 
            'incident_type' : 'accident', 
            'severity' : 'moderate', 
            'message' : '203호 이유빈 환자 낙상사고 - 즉각 대응 필요' 
        }
    alert_json = { 
            'patient_id': '3', 
            'room_id': '8', 
            'incident_type' : 'alert', 
            'severity' : 'minor', 
            'message' : '203호 이유빈 환자 낙상위험 - 주의 감찰' 
        }

    json = accident_json
    if(action == 'Fall'):
        json = accident_json
    else:
        json = alert_json
    
    try:
        response = requests.post(
            f'{api_url}/api/fall-incident',
            json=json,
            timeout=5
        )
        if response.status_code == 200:
            print(f"✅ 낙상 알림 발송 (신뢰도: {confidence:.2%})")
        else:
            print(f"❌ 알림 실패, 상태 코드: {response.status_code}")
    except Exception as e:
        print(f"❌ 알림 실패: {e}")

if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default=out_path,
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.')
    args = par.parse_args()

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG(device=device)

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()
    #frame_size = cam.frame_size
    #scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    outvid = False
    if args.save_out != '':
        outvid = True
        fps_val = cam.fps if hasattr(cam, 'fps') and cam.fps > 0 else 30
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))
    print("line 97")
    fps_time = 0
    f = 0
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        image = frame.copy()

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det
        #print("fps " + str(f))
        detections = []  # List of Detections object for tracking.
        if detected is not None:
            #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

            # VISUALIZE.
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                    check_and_alert_fall('Fall', out[0].max()) # 
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)
                    check_and_alert_fall('Lying', out[0].max()) #

            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)
        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        fps_time = time.time()
        frame = frame[:, :, ::-1]
        fps_time = time.time()
        if outvid:
            writer.write(frame)

        # cv2.imwrite(f"frames/frame_{f}.jpg", frame) # 이미지저장
        cv2.imshow('frame', frame) # 분석화면 보기
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Clear resource.

    print("# CAM STOP!!!")
    cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
