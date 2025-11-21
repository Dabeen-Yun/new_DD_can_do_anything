from Params import *
import numpy as np
import math

from Utility import get_remain_path

d2r = np.deg2rad

class Satellite:
    def __init__(self, id, orb, spo, alt, phasing_inter_plane, inc_deg, sat_list):
        self.id = id
        self.sat_list = sat_list

        # 위치
        self.orb = orb  # P
        self.spo = spo  # S
        self.sat = orb * spo  # t = P * S
        self.x = self.id // self.spo  # plane index
        self.y = self.id % self.spo  # slot index
        self.lat = 0  # grid latitude [0~360]
        self.lon = 0
        self.alt = alt

        self.phasing_offset_deg = lambda y: (360 * CONSTELLATION_PARAM_F * self.y) / self.sat
        self.orbit_spacing_deg = 360 / self.orb
        self.phasing_inter_plane = phasing_inter_plane

        self.phasing_intra_plane = 360 / self.spo
        self.phasing_adjacent_plane = CONSTELLATION_PARAM_F * 360 / self.sat

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

        self.set_lla()

    # walker-star constellation
    def set_lla(self):
        self.lon = (self.x * self.orbit_spacing_deg + self.phasing_offset_deg(self.y)) % 360

        phase = (2 * np.pi * self.y) / self.spo
        self.lat = 90 * np.sin(phase)  # POLAR_LATITUDE * np.sin(phase)

    def set_adjacent_node(self):
        # horizontal adjacent node
        h_adj_1 = self.id + self.spo
        h_adj_2 = self.id - self.spo
        if h_adj_1 >= self.sat:
            h_adj_1 = h_adj_1 % self.sat
        if h_adj_2 < 0:
            h_adj_2 = self.sat + h_adj_2
        self.adj_sat_index_list[2] = h_adj_1
        self.adj_sat_index_list[3] = h_adj_2
        self.inter_ISL_list.append(h_adj_1)
        self.inter_ISL_list.append(h_adj_2)

        # vertical adjacent node
        v_adj_1 = self.id + 1
        v_adj_2 = self.id - 1
        if self.id // self.spo != v_adj_1 // self.spo:
            v_adj_1 -= self.spo
        if self.id // self.spo != v_adj_2 // self.spo:
            v_adj_2 += self.spo
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

    def time_tic(self, delta_time=1):  # 1ms 마다
        self.time += delta_time

        orbital_period_ms = SATELLITE_ORBITAL_PERIOD #* 1000  # sec->ms 변환

        # 궤도 주기 계산 (초당 360도 회전)
        mean_motion_deg_per_ms = 360 / orbital_period_ms

        # 위성의 초기 위상 차이 (경도 기준)
        init_lon = (self.x * self.orbit_spacing_deg + self.phasing_offset_deg(self.y)) % 360

        # 시간에 따라 경도 업데이트
        self.lon = (init_lon + mean_motion_deg_per_ms * self.time) % 360

        # 위도는 경사 궤도를 따라 sin 형태로 주기적 움직임
        # 전체 궤도 주기 90분 기준으로 위도 변화 (위상은 y에 따라 달라짐)
        phase = 2 * np.pi * self.y / self.spo
        self.lat = 90 * np.sin(2 * np.pi * self.time / orbital_period_ms + phase)

        # 위성 간 propagation delay 재계산
        self.get_propagation_delay()