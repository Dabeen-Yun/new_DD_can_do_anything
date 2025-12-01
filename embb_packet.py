import pandas as pd
import numpy as np
import math
import os
import random

# ---------------------------------------------------------
# 1. 설정
# ---------------------------------------------------------
OUTPUT_DIR = "./data"
OUTPUT_FILE = "flight_paths.csv"
NUM_POINTS_PER_ROUTE = 300  # 각 경로당 생성할 데이터 포인트(항공기) 수 (밀도 조절)
PATH_WIDTH_KM = 50  # 항공로의 너비 (좌표가 너무 일직선이지 않게 산포)

# ---------------------------------------------------------
# 2. 주요 공항 좌표 (Lat, Lon) - 트래픽이 많은 허브 위주
# ---------------------------------------------------------
airports = {
    # 아시아
    "ICN": (37.4602, 126.4407),  # 서울/인천
    "NRT": (35.7720, 140.3929),  # 도쿄/나리타
    "HKG": (22.3080, 113.9185),  # 홍콩
    "SIN": (1.3644, 103.9915),  # 싱가포르
    "DXB": (25.2532, 55.3657),  # 두바이

    # 유럽
    "LHR": (51.4700, -0.4543),  # 런던
    "CDG": (49.0097, 2.5479),  # 파리
    "FRA": (50.0379, 8.5622),  # 프랑크푸르트

    # 북미
    "JFK": (40.6413, -73.7781),  # 뉴욕
    "LAX": (33.9416, -118.4085),  # 로스앤젤레스
    "SFO": (37.6213, -122.3790),  # 샌프란시스코
    "ORD": (41.9742, -87.9073),  # 시카고
    "ATL": (33.6407, -84.4277),  # 애틀랜타

    # 대양주
    "SYD": (-33.9399, 151.1753),  # 시드니
}

# ---------------------------------------------------------
# 3. 주요 항공 경로 정의 (출발 -> 도착)
# 실제 트래픽이 많은 구간들 (대서양 횡단, 태평양 횡단 등)
# ---------------------------------------------------------
routes = [
    # [태평양 횡단] 북미 <-> 아시아 (북극 항로 유사 효과)
    ("ICN", "LAX"), ("ICN", "JFK"), ("NRT", "LAX"), ("NRT", "SFO"),
    ("HKG", "LAX"), ("HKG", "JFK"), ("SIN", "SFO"),

    # [대서양 횡단] 북미 <-> 유럽
    ("JFK", "LHR"), ("JFK", "CDG"), ("JFK", "FRA"),
    ("ORD", "LHR"), ("ATL", "CDG"), ("LAX", "LHR"),

    # [유라시아] 유럽 <-> 아시아
    ("LHR", "DXB"), ("FRA", "SIN"), ("CDG", "HKG"),
    ("LHR", "ICN"), ("DXB", "ICN"),

    # [대양주]
    ("SYD", "LAX"), ("SYD", "SIN"), ("SYD", "HKG"),

    # [미국 내륙]
    ("JFK", "LAX"), ("ORD", "MIA"), ("SFO", "JFK")
]


# ---------------------------------------------------------
# 4. 유틸리티 함수: 대권 경로 계산 (Great Circle)
# ---------------------------------------------------------
def get_point_on_great_circle(lat1, lon1, lat2, lon2, fraction):
    """
    두 지점 사이의 대권 경로 상에서 fraction(0.0~1.0) 위치에 있는 점의 좌표를 반환
    """
    # 라디안 변환
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # 구면 거리 계산 (Haversine 등)
    delta_lon = lon2 - lon1

    # 구면 삼각법을 이용한 중간 지점 계산
    # A, B 계산
    x = math.cos(lat1) * math.cos(lat2) * math.cos(delta_lon)
    y = math.sin(lat1) * math.sin(lat2)
    d = math.acos(x + y)  # 각거리 (angular distance)

    if d == 0: return math.degrees(lat1), math.degrees(lon1)

    a = math.sin((1 - fraction) * d) / math.sin(d)
    b = math.sin(fraction * d) / math.sin(d)

    x = a * math.cos(lat1) * math.cos(lon1) + b * math.cos(lat2) * math.cos(lon2)
    y = a * math.cos(lat1) * math.sin(lon1) + b * math.cos(lat2) * math.sin(lon2)
    z = a * math.sin(lat1) + b * math.sin(lat2)

    new_lat = math.atan2(z, math.sqrt(x ** 2 + y ** 2))
    new_lon = math.atan2(y, x)

    return math.degrees(new_lat), math.degrees(new_lon)


def add_noise(lat, lon, width_km):
    """
    경로를 너무 일직선이 아니게 만들기 위해 랜덤 노이즈 추가
    """
    # 대략 위도 1도 = 111km
    sigma = width_km / 111.0 / 2  # 표준편차
    new_lat = random.gauss(lat, sigma)
    # 경도는 위도에 따라 거리 비율이 달라지지만, 단순화하여 처리
    new_lon = random.gauss(lon, sigma)
    return new_lat, new_lon


# ---------------------------------------------------------
# 5. 데이터 생성 메인 로직
# ---------------------------------------------------------
def generate_data():
    flight_data = []

    print(f"[INFO] Generating flight paths based on {len(routes)} major routes...")

    for start_code, end_code in routes:
        if start_code not in airports or end_code not in airports:
            continue

        start_pos = airports[start_code]
        end_pos = airports[end_code]

        # 한 경로당 NUM_POINTS_PER_ROUTE 만큼의 가상 항공기 위치 생성
        for _ in range(NUM_POINTS_PER_ROUTE):
            # 0.0 ~ 1.0 사이 랜덤 위치 (경로상의 어딘가)
            f = random.random()

            # 대권 경로 상의 좌표 계산
            lat, lon = get_point_on_great_circle(start_pos[0], start_pos[1], end_pos[0], end_pos[1], f)

            # 약간의 위치 노이즈 추가 (항공로 너비 효과)
            lat, lon = add_noise(lat, lon, PATH_WIDTH_KM)

            flight_data.append({"lat": lat, "lon": lon})

    # DataFrame 변환
    df = pd.DataFrame(flight_data)

    # 폴더가 없으면 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # CSV 저장
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(save_path, index=False)
    print(f"[SUCCESS] Saved {len(df)} flight points to {save_path}")
    print(f"          (Columns: lat, lon)")


if __name__ == "__main__":
    generate_data()