from Params import *
from Utility import *

import numpy as np
import math

d2r = np.deg2rad
r2d = np.rad2deg

class Satellite:
    def __init__(self, id, orb, spo, alt, phasing_inter_plane, sat_list, mode, sat_log_path):
        self.id = id
        self.sat_list = sat_list
        self.mode = mode
        self.sat_log_path = sat_log_path

        # 위치
        self.orb = orb  # P
        self.spo = spo  # S
        self.sat = orb * spo  # t = P * S
        self.x = self.id // self.spo  # plane index
        self.y = self.id % self.spo  # slot index (궤도 내 몇 번째 위성인지)
        self.alt = alt
        self.radius = R_EARTH_RADIUS + alt # 지구 반지름 + 고도

        # RAAN (Right Ascension of the Ascending Node): 궤도면의 회전 각도
        # Walker Star는 180도 범위 내에 궤도면을 배치합니다. (입력받은 phasing_inter_plane 사용)
        self.raan = d2r(self.x * phasing_inter_plane)

        # 초기 위상 (Mean Anomaly): 궤도 내 위성의 시작 위치
        # 360도를 위성 수(spo)로 나눈 간격
        self.initial_phase = d2r(self.y * (360.0 / self.spo))

        # 경사각 (Inclination): Walker Star(Polar)는 90도에 가까움
        # 정확히 90도면 atan2 계산 시 특이점이 생길 수 있어 86.4~89도 추천 (Iridium: 86.4)
        self.inclination = d2r(86.4)

        # 궤도 속도 (Mean Motion) 계산
        mu = 3.986004418e5 # 지구 중력 상수
        self.mean_motion = math.sqrt(mu / self.radius ** 3)

        # 위치 (Lat: -90~90, Lon: -180~180)
        self.lat = 0
        self.lon = 0

        # 시간 [ms]
        self.time = 0

        self.vnf_list = []  # list [int]
        self.current_vsg_id = -1  # int
        self.vsg_enter_time = -1  # int

        # Inter Satellite Link (ISL)
        # adjacent satellite : [intra1, intra2, inter1, inter2]
        self.adj_sat_index_list = [-1, -1, -1, -1]
        self.adj_sat_p_d_list = [-1, -1, -1, -1]
        self.intra_ISL_list = []
        self.inter_ISL_list = []
        self.intra_ISL_p_d = []
        self.inter_ISL_p_d = []

        # processing queue
        # self.process_queue = deque()
        self.process_queue = []

        # ISL queue # transmitting queue
        self.queue_ISL_intra_1 = []
        self.queue_ISL_intra_2 = []
        self.queue_ISL_inter_1 = []
        self.queue_ISL_inter_2 = []
        self.queue_ISL = [self.queue_ISL_intra_1, self.queue_ISL_intra_2,
                          self.queue_ISL_inter_1, self.queue_ISL_inter_2]
        self.queue_TSL = []

        self.update_coordinates(0)

    def update_coordinates(self, time_ms):
        """
        3차원 궤도 회전을 통해 위경도를 계산합니다.
        이 함수를 사용하면 경도가 -180~180 범위로 자동 계산됩니다.
        """
        t_s = time_ms / 1000.0  # ms -> sec

        # 1. 현재 궤도 내 각도 (Mean Anomaly)
        current_angle = self.initial_phase + (self.mean_motion * t_s)

        # 2. 궤도 평면상 좌표 (2D)
        x_orb = self.radius * math.cos(current_angle)
        y_orb = self.radius * math.sin(current_angle)

        # 3. 3차원 회전 (ECI 좌표계 변환)
        # Z축(RAAN) -> X축(Inclination) 회전 적용
        cos_om = math.cos(self.raan)
        sin_om = math.sin(self.raan)
        cos_i = math.cos(self.inclination)
        sin_i = math.sin(self.inclination)

        x_eci = x_orb * cos_om - y_orb * cos_i * sin_om
        y_eci = x_orb * sin_om + y_orb * cos_i * cos_om
        z_eci = y_orb * sin_i

        # 4. 지구 자전 고려 (ECEF 좌표계 변환)
        # 지구가 도는 속도(WE)만큼 보정
        we = 7.2921151467e-5  # rad/s
        theta_g = we * t_s

        x_ecef = x_eci * math.cos(theta_g) + y_eci * math.sin(theta_g)
        y_ecef = -x_eci * math.sin(theta_g) + y_eci * math.cos(theta_g)
        z_ecef = z_eci

        # 5. 위경도 변환 (핵심: atan2 사용으로 -180~180 범위 확보)
        self.lat = r2d(math.asin(z_ecef / self.radius))  # -90 ~ 90
        self.lon = r2d(math.atan2(y_ecef, x_ecef))  # -180 ~ 180

    def set_adjacent_node(self):
        # horizontal adjacent node
        # 오른쪽 이웃 (East)
        if self.x == self.orb - 1:
            h_adj_1 = -1
        else:
            h_adj_1 = self.id + self.spo

        # 왼쪽 이웃 (West)
        if self.x == 0:
            h_adj_2 = -1
        else:
            h_adj_2 = self.id - self.spo

        self.adj_sat_index_list[2] = h_adj_1
        self.adj_sat_index_list[3] = h_adj_2

        # ISL 리스트에도 유효한 위성만 추가
        if h_adj_1 != -1:
            self.inter_ISL_list.append(h_adj_1)
        if h_adj_2 != -1:
            self.inter_ISL_list.append(h_adj_2)

        # vertical adjacent node
        v_adj_1 = self.id + 1
        v_adj_2 = self.id - 1
        # 궤도면(x)이 바뀌지 않도록 모듈러 연산 처리
        # (현재 궤도면의 시작 ID와 끝 ID 범위를 벗어나면 순환)
        plane_start_id = self.x * self.spo
        plane_end_id = plane_start_id + self.spo - 1

        if v_adj_1 > plane_end_id:
            v_adj_1 = plane_start_id

        if v_adj_2 < plane_start_id:
            v_adj_2 = plane_end_id

        self.adj_sat_index_list[0] = v_adj_1
        self.adj_sat_index_list[1] = v_adj_2
        self.intra_ISL_list.append(v_adj_1)
        self.intra_ISL_list.append(v_adj_2)

    def get_ecef_coords(self, lat, lon):
        """LLA (Lat, Lon, Alt)를 ECEF (x, y, z)로 변환합니다."""

        # Convert to radians
        lat_rad = d2r(lat)
        lon_rad = d2r(lon)

        R_obj = R_EARTH_RADIUS + ORBIT_ALTITUDE

        # ECEF conversion
        x = R_obj * math.cos(lat_rad) * math.cos(lon_rad)
        y = R_obj * math.cos(lat_rad) * math.sin(lon_rad)
        z = R_obj * math.sin(lat_rad)

        return x, y, z

    def calculate_delay_to_sat(self, other_sat):
        """현재 위성과 다른 위성 간의 전파 지연 시간(ms)을 ECEF 기반으로 계산합니다."""

        # 1. Get ECEF coords for self
        x1, y1, z1 = self.get_ecef_coords(self.lat, self.lon)

        # 2. Get ECEF coords for other_sat
        x2, y2, z2 = self.get_ecef_coords(other_sat.lat, other_sat.lon)

        # 3. Calculate 3D Euclidean distance (m)
        distance_m = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

        # 4. Calculate propagation delay (ms)
        delay_s = distance_m / PARAM_C
        delay_ms = delay_s * 1000

        # Original code used 'round(length / PARAM_C)', so we round the result to the nearest integer (ms).
        return delay_ms

    def get_propagation_delay(self):
        """
        인접 위성 간의 전파 지연 시간(Propagation Delay, ms)을 ECEF 기반으로 계산하여 업데이트
        """
        # 리스트 초기화
        self.adj_sat_p_d_list = []
        self.intra_ISL_p_d = []
        self.inter_ISL_p_d = []

        # adj_sat_index_list: [intra1, intra2, inter1, inter2] 순서
        for i, adj_sat_id in enumerate(self.adj_sat_index_list):
            if adj_sat_id == -1:
                delay = -1
            else:
                try:
                    adj_sat = self.sat_list[adj_sat_id]
                    # ECEF 기반 지연 시간 계산
                    delay = self.calculate_delay_to_sat(adj_sat)
                except IndexError:
                    delay = -1

            # Update the main adjacent delay list
            self.adj_sat_p_d_list.append(delay)

            # Update the intra/inter lists for compatibility
            if i < 2:  # Indices 0 and 1 are intra-plane (vertical)
                self.intra_ISL_p_d.append(delay)
            else:  # Indices 2 and 3 are inter-plane (horizontal)
                self.inter_ISL_p_d.append(delay)

    def get_next_hop_link_idx(self, gsfc, node_id=None):
        if node_id is None:
            remained_path = get_remain_path(gsfc)
            next_node_id = remained_path[0][0]
        else:
            next_node_id = node_id

        for i in range(len(self.adj_sat_index_list)):
            idx = self.adj_sat_index_list[i]
            if idx == next_node_id:
                return i

    def append_process_queue(self, gsfc):
        is_duplicate = any(
            item[0] == gsfc.id and item[1] == gsfc.vnf_id
            for item in self.process_queue
        )

        if not is_duplicate:
            self.process_queue.append([gsfc.id, gsfc.vnf_id, gsfc.vnf_sizes[gsfc.vnf_id]])

    def remove_process_queue(self, gsfc):
        target_gsfc_id = gsfc.id

        self.process_queue = [
            item for item in self.process_queue if item[0] != target_gsfc_id
        ]

        gsfc.vnf_id += 1

    def time_tic(self, delta_time=1):  # 1ms 마다
        self.time = delta_time

        self.update_coordinates(self.time)

        # 위성 간 propagation delay 재계산
        self.get_propagation_delay()

        write_sat_csv_log(self)