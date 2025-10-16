
import requests
import threading
from collections import deque
from datetime import datetime
import time

# 전역 변수로 상태 관리
fall_buffer = deque(maxlen=5)  # 최근 5프레임 저장
last_alert_time = None
def check_and_alert_fall(is_fall, confidence, cooldown_sec=10, threshold=0.7):
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
    global fall_buffer, last_alert_time
    
    print('check')
    # 1. 현재 프레임 결과 저장
    fall_buffer.append({'fall': is_fall, 'conf': confidence})
    
    # 2. 버퍼가 다 안 찼으면 대기
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
        thread = threading.Thread(target=send_alert, args=(avg_conf,))
        thread.daemon = True
        thread.start()

        last_alert_time = datetime.now()
        fall_buffer.clear()

    return False

def send_alert(confidence):
    print('send_alert')
    """백그라운드 API 호출"""
    try:
        response = requests.post(
            'http://your-server.com/api/fall-alert',
            json={
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence,
                'location': 'Camera_01'
            },
            timeout=5
        )
        if response.status_code == 200:
            print(f"✅ 낙상 알림 발송 (신뢰도: {confidence:.2%})")
        else:
            print(f"❌ 알림 실패, 상태 코드: {response.status_code}")
    except Exception as e:
        print(f"❌ 알림 실패: {e}")

for idx in range(4):
    print(f"반복횟수 : {idx+1}번째 체크! {sum(f['conf'] for f in fall_buffer) / 5}");
    check_and_alert_fall(True, 0.8)


print(f"반복횟수 : 5번째 체크! {sum(f['conf'] for f in fall_buffer) / 5}");
check_and_alert_fall(True, 0.6)

#time.sleep(5)  # 쿨다운 시간 대기
print(f"반복횟수 : 6번째 체크! {sum(f['conf'] for f in fall_buffer) / 5}");
check_and_alert_fall(True, 0.8)

time.sleep(2)  # 쿨다운 시간 대기
print(f"반복횟수 : 7번째 체크! {sum(f['conf'] for f in fall_buffer) / 5}");
check_and_alert_fall(True, 0.8)

time.sleep(2)  # 쿨다운 시간 대기
print(f"반복횟수 : 8번째 체크! {sum(f['conf'] for f in fall_buffer) / 5}");
check_and_alert_fall(True, 0.8)

time.sleep(2)  # 쿨다운 시간 대기
print(f"반복횟수 : 9번째 체크! {sum(f['conf'] for f in fall_buffer) / 5}");
check_and_alert_fall(True, 0.8)

time.sleep(2)  # 쿨다운 시간 대기
print(f"반복횟수 : 10번째 체크! {sum(f['conf'] for f in fall_buffer) / 5}");
check_and_alert_fall(True, 0.8)

time.sleep(2)  # 쿨다운 시간 대기
print(f"반복횟수 : 11번째 체크! {sum(f['conf'] for f in fall_buffer) / 5}");
check_and_alert_fall(True, 0.8)

time.sleep(2)  # 쿨다운 시간 대기
print(f"반복횟수 : 12번째 체크! {sum(f['conf'] for f in fall_buffer) / 5}");
check_and_alert_fall(True, 0.8)