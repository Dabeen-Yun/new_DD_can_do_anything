from Params import *
from Satellite import *
from Gserver import *
from VSG import *
from GSFC import *
from Utility import *

import numpy as np
import random
import networkx as nx
import math
from time import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from global_land_mask import globe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
d2r = np.deg2rad

class Simulation:
    def __init__(self, seed: int = 921):
        np.random.seed(seed)
        random.seed(seed)

        # topology
        self.sat_list = []
        self.vsg_list = []
        self.gserver_list = []
        self.gsfc_list = []

        self.G = None
        self.vsg_G = None

        self.gsfc_log_path = ""
        self.sat_log_path = ""
        self.vsg_log_path = ""
        self.test_gsfc_id = -1 # 경로 추적을 위한 GSFC ID 변수
        self.generated_gsfc_id = 0 # gsfc 등록을 위한 gsfc id

        # video 관련 변수
        self.video_writer = None
        self.video_path = ""
        self.video_fps = 10

        # eMBB gateway 관련 파라미터
        gw_city_df = pd.read_csv("./data/gateways_starlink_sx.csv")
        gw_continent_df = pd.read_csv("./data/gateway_continets.csv")
        flight_path_df = pd.read_csv("./data/flight_paths.csv")

        self.gw_city_df = gw_city_df[["lat", "lng"]].reset_index(drop=True)
        self.gw_city_lats = self.gw_city_df["lat"].to_numpy(dtype=float)
        self.gw_city_lons = self.gw_city_df["lng"].to_numpy(dtype=float)

        self.gw_continent_df = gw_continent_df[["lat", "lng"]].reset_index(drop=True)

        self.flight_path_df = flight_path_df[["lat", "lng"]].reset_index(drop=True)

        self.gw_city_df = self.gw_city_df[
            (self.gw_city_df["lat"] >= LAT_RANGE[0]) & (self.gw_city_df["lat"] <= LAT_RANGE[1]) &
            (self.gw_city_df["lng"] >= LON_RANGE[0]) & (self.gw_city_df["lng"] <= LON_RANGE[1])
            ]

        self.gw_continent_df = self.gw_continent_df[
            (self.gw_continent_df["lat"] >= LAT_RANGE[0]) & (self.gw_continent_df["lat"] <= LAT_RANGE[1]) &
            (self.gw_continent_df["lng"] >= LON_RANGE[0]) & (self.gw_continent_df["lng"] <= LON_RANGE[1])
            ]

    def set_constellation(self, mode):
        phasing_inter_plane = 180 / NUM_ORBITS

        for sat_id in range(NUM_SATELLITES):
            sat = Satellite(sat_id, NUM_ORBITS, NUM_SATELLITES_PER_ORBIT, ORBIT_ALTITUDE, phasing_inter_plane,
                            self.sat_list, mode, self.sat_log_path)
            self.sat_list.append(sat)

        for sat_id in range(NUM_SATELLITES):
            sat = self.sat_list[sat_id]
            sat.set_adjacent_node()
            sat.get_propagation_delay()

            write_sat_csv_log(sat)

    def get_distance_between_VSGs(self, vid1, vid2):
        vsg1 = next((x for x in self.vsg_list if x.id == vid1), None)
        vsg2 = next((x for x in self.vsg_list if x.id == vid2), None)

        vsg1_lon = vsg1.center_coords[0]
        vsg1_lat = vsg1.center_coords[1]
        vsg2_lon = vsg2.center_coords[0]
        vsg2_lat = vsg2.center_coords[1]

        vsg1_lon_rad = d2r(vsg1_lon)
        vsg1_lat_rad = d2r(vsg1_lat)
        vsg1_alt_m = ORBIT_ALTITUDE
        vsg1_R_obj = R_EARTH_RADIUS + vsg1_alt_m

        vsg2_lon_rad = d2r(vsg2_lon)
        vsg2_lat_rad = d2r(vsg2_lat)
        vsg2_alt_m = ORBIT_ALTITUDE
        vsg2_R_obj = R_EARTH_RADIUS + vsg2_alt_m

        vsg1_x = vsg1_R_obj * math.cos(vsg1_lat_rad) * math.cos(vsg1_lon_rad)
        vsg1_y = vsg1_R_obj * math.cos(vsg1_lat_rad) * math.sin(vsg1_lon_rad)
        vsg1_z = vsg1_R_obj * math.sin(vsg1_lat_rad)

        vsg2_x = vsg2_R_obj * math.cos(vsg2_lat_rad) * math.cos(vsg2_lon_rad)
        vsg2_y = vsg2_R_obj * math.cos(vsg2_lat_rad) * math.sin(vsg2_lon_rad)
        vsg2_z = vsg2_R_obj * math.sin(vsg2_lat_rad)

        # 3D 유클리드 거리 계산 (미터)
        distance_m = math.sqrt((vsg1_x - vsg2_x) ** 2 + (vsg1_y - vsg2_y) ** 2 + (vsg1_z - vsg2_z) ** 2)

        return distance_m

    def initial_vsg_regions(self, mode, lon_step = None):
        if lon_step is None:
            lon_step = LON_STEP
        self.vsg_list = []
        self.gserver_list = []
        self.vsg_G = nx.Graph()

        vid = 0
        gid = 0

        lat_bins = np.arange(LAT_RANGE[0], LAT_RANGE[1]+1, LAT_STEP)
        lon_bins = np.arange(LON_RANGE[0], LON_RANGE[1]+1, lon_step)

        num_row = math.ceil((LAT_RANGE[1] - LAT_RANGE[0]) / LAT_STEP)
        num_col = math.ceil((LON_RANGE[1] - LON_RANGE[0]) / lon_step)

        for lat_min in lat_bins:
            lat_max = lat_min + LAT_STEP
            for lon_min in lon_bins:
                lon_max = lon_min + lon_step

                # 현재 그리드 셀 안에 속하는 위성 추출
                cell_sats = [
                    sat for sat in self.sat_list
                    if lat_min <= sat.lat < lat_max and lon_min <= sat.lon < lon_max
                ]

                if not cell_sats:
                    print(f"NO SAT in VSG {vid+1}")
                    continue

                center_lat = (lat_min + LAT_STEP) / 2
                center_lon = (lon_min + lon_step) / 2

                ground_server = Gserver(gid, center_lon, center_lat, vid)

                vsg = VSG(vid, (center_lon, center_lat), lon_min, lon_max, lat_min, lat_max, cell_sats, ground_server, mode, self.vsg_log_path)
                for sat in cell_sats:
                    sat.current_vsg_id = vid
                    sat.vsg_enter_time = time()

                self.vsg_list.append(vsg)
                self.gserver_list.append(ground_server)

                vid += 1
                gid += 1

        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        existing = {v.id for v in self.vsg_list}

        for vid in range(num_row * num_col):
            if vid not in existing:
                continue

            row, col = divmod(vid, num_col)
            for dr, dc in DIRS:
                nrow = (row + dr) % num_row
                ncol = (col + dc) % num_col
                nvid = nrow * num_col + ncol

                if nvid not in existing:
                    continue  # 이웃이 비어 있으면 스킵

                if self.vsg_G.has_edge(vid, nvid):
                    continue

                vsg_distance = self.get_distance_between_VSGs(vid, nvid)
                self.vsg_G.add_edge(vid, nvid, weight=vsg_distance)

        for vsg in self.vsg_list:
            print("\n--- VSG 엣지 및 VSG 포함 satellite idx 확인 (샘플) ---")
            print(f"  -> VSG 내 위성 idxs {[s.id for s in vsg.satellites]}")
            sample_vsg_id = vsg.id
            if sample_vsg_id in self.vsg_G:
                print(f"VSG {sample_vsg_id}의 인접 엣지:")
                for neighbor, data in self.vsg_G[sample_vsg_id].items():
                    print(f"  -> VSG {neighbor}")
            else:
                print(f"위성 {sample_vsg_id}가 그래프에 존재하지 않습니다. (혼잡 상태 등으로 제외되었을 수 있음)")
            print("------------------------------------------")

    def get_urllc_vsg_ids_from_gateways(self):
        """
        URLLC 트래픽이 주로 발생하는 VSG id 집합을 샘플링 기반으로 추정.
        - generate_traffic()에서 URLLC를 만들 때 쓰는 로직과 동일한
          (hub_lat, hub_lon + Gaussian) 를 그대로 사용.
        """
        urllc_vsg_ids = set()

        # 각 city gateway를 URLLC 핫스팟 후보로 보고
        for _, row in self.gw_city_df.iterrows():
            hub_lat = float(row["lat"])
            hub_lon = float(row["lng"])

            vsg_id = get_vsg_id_from_coords(self.vsg_list, hub_lat, hub_lon)
            if vsg_id is not None:
                urllc_vsg_ids.add(vsg_id)

        return urllc_vsg_ids

    def initial_vnfs_to_vsg(self):
        # vnf_types = list(range(VNF_TYPES_PER_VSG[0], VNF_TYPES_PER_VSG[1]))  # [1,2,3,4,5,6]
        all_vnfs = ['1','2','3','4','5','6']

        # URLLC SFC에 포함된 VNF들만 따로 추출
        urllc_vnfs = [str(v) for v in SFC_URLLC_SEQ]
        urllc_vnfs = list(dict.fromkeys(urllc_vnfs))
        non_urllc_vnfs = [v for v in all_vnfs if v not in urllc_vnfs]

        # 1) gw_city만 보고 URLLC 가능 VSG id 집합 구하기
        urllc_vsg_ids = self.get_urllc_vsg_ids_from_gateways()
        print("[INFO] URLLC VSG candidates from gw_city:", urllc_vsg_ids)

        for vsg in self.vsg_list:
            if vsg.id in urllc_vsg_ids:
                for v in urllc_vnfs:
                    vsg.assigned_vnfs.append(v)

        for v in non_urllc_vnfs:
            candidate_vsgs = [
                vsg for vsg in self.vsg_list
                if len(set(vsg.assigned_vnfs)) < NUM_VNFS_PER_VSG
            ]
            if candidate_vsgs:
                random.choice(candidate_vsgs).assigned_vnfs.append(v)

        for vsg in self.vsg_list:
            current = set(vsg.assigned_vnfs)
            remain_slots = NUM_VNFS_PER_VSG - len(current)

            if remain_slots <= 0:
                vsg.assigned_vnfs = sorted(current, key=int)
                continue

            if vsg.id in urllc_vsg_ids:
                candidates = [v for v in all_vnfs if v not in current]
            else:
                candidates = [v for v in non_urllc_vnfs if v not in current]

            if len(candidates) <= remain_slots:
                extra = candidates
            else:
                extra = random.sample(candidates, remain_slots)

            assigned_vnfs_set = current.union(extra)
            vsg.assigned_vnfs = sorted(list(assigned_vnfs_set), key=int)

        for vsg in self.vsg_list:
            assigned_vnfs_set = set(vsg.assigned_vnfs)

            current_hosted_vnfs = set()
            for sat in vsg.satellites:
                current_hosted_vnfs.update(sat.vnf_list)

            # 3. 부족한(채워 넣어야 할) VNF만 추출
            missing_vnfs = list(assigned_vnfs_set - current_hosted_vnfs)

            # 4. 부족한 VNF 배치 시작
            for vnf_type in missing_vnfs:
                is_placed = False

                candidates = vsg.satellites[:]  # 복사본 생성
                random.shuffle(candidates)

                for sat in candidates:
                    if len(sat.vnf_list) < NUM_VNFS_PER_SAT:
                        sat.vnf_list.append(vnf_type)
                        is_placed = True
                        break

                if is_placed: continue

                # 전략 2: 빈 공간이 없다면, 교체 가능한(Non-assigned) VNF를 가진 위성 탐색 (Replacement)
                for sat in candidates:
                    victim_candidates = [v for v in sat.vnf_list if v not in assigned_vnfs_set]

                    if victim_candidates:
                        victim_vnf = random.choice(victim_candidates)
                        sat.vnf_list.remove(victim_vnf)
                        sat.vnf_list.append(vnf_type)
                        is_placed = True
                        break

                if not is_placed:
                    input(f"NOT PLACED")
                    pass

            print(f"  -> VSG {vsg.id} 내 할당 vnfs {vsg.assigned_vnfs}")

    def compute_all_pair_distance(self, csv_dir_path, mode):
        self.G = nx.Graph()
        # self.TG = nx.Graph()

        # congestion 아닌 위성들만으로 그래프 구성
        for sat in self.sat_list:
            for neighbor_id in sat.adj_sat_index_list:
                if neighbor_id != -1:

                    neighbor_sat = self.sat_list[neighbor_id]
                    prop_delay_ms = sat.calculate_delay_to_sat(neighbor_sat)

                    self.G.add_edge(sat.id, neighbor_id, weight=prop_delay_ms, link_type='isl')
                    # self.TG.add_edge(sat.id, neighbor_id, weight=prop_delay_ms, link_type='tsl')

                    # self.G.add_edge(sat.id, neighbor_id, weight=1, link_type='isl')
                    # self.TG.add_edge(sat.id, neighbor_id, weight=1, link_type='tsl')

            print("\n--- ISL 엣지 및 Propagation Delay 확인 (샘플) ---")
            sample_sat_id = sat.id
            if sample_sat_id in self.G:
                print(f"위성 {sample_sat_id}의 installed vnfs:")
                print(f"  -> VNF List: {sat.vnf_list}")
                print(f"위성 {sample_sat_id}의 인접 엣지:")
                for neighbor, data in self.G[sample_sat_id].items():
                    # 'weight'는 전파 지연 (ms)
                    print(f"  -> 위성 {neighbor}: weight (Delay) = {data.get('weight', 'N/A'):.2f} ms")
            else:
                print(f"위성 {sample_sat_id}가 그래프에 존재하지 않습니다. (혼잡 상태 등으로 제외되었을 수 있음)")
            print("------------------------------------------")

        graph_file_path = os.path.join(csv_dir_path, f"{mode}_network_G.pkl")
        save_networkx_graph(self.G, graph_file_path)

    # TODO. packet 생성 코드 삽입
    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        두 위경도 좌표 간의 거리를 m 단위로 반환 (Haversine Formula)
        """
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (np.sin(dlat / 2.0) ** 2 +
             np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)

        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

        # dlat = math.radians(lat2 - lat1)
        # dlon = math.radians(lon2 - lon1)
        #
        # a = (math.sin(dlat / 2) ** 2 +
        #      math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
        #      math.sin(dlon / 2) ** 2)
        #
        # c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R_EARTH_RADIUS * c
        return distance

    # ---------------------------------------------------------
    # 2. 트래픽 생성 로직
    # ---------------------------------------------------------
    def pick_random_flight_path_point(self):
        """
        항공로 데이터에서 랜덤한 지점(lat, lon) 하나를 반환
        """
        if self.flight_path_df.empty:
            return None, None

        idx = random.randrange(len(self.flight_path_df))
        row = self.flight_path_df.iloc[idx]
        return float(row["lat"]), float(row["lng"])

    def pick_random_vsg_gw(self, range):
        """
        CSV에서 랜덤 도시 하나 뽑아서 (lat, lon, vsg_id) 반환
        """
        if range == "city":
            idx = random.randrange(len(self.gw_city_df))
            row = self.gw_city_df.iloc[idx]
        elif range == "continent":
            idx = random.randrange(len(self.gw_continent_df))
            row = self.gw_continent_df.iloc[idx]
        else:
            input("------------ RANGE IS 'city' OR 'continent'. ------------")
            return 0
        gw_lat = float(row["lat"])
        gw_lon = float(row["lng"])

        gw_vsg_id = get_vsg_id_from_coords(self.vsg_list, gw_lat, gw_lon)
        return gw_vsg_id, gw_lat, gw_lon

    def nearest_gateway_coords(self, lat, lon):
        """
        (lat_deg, lon_deg)에서 거리 기준으로 가장 가까운 gateway의 (lat, lon)을 반환
        """
        c = self.calculate_haversine_distance(lat, lon, self.gw_city_lats, self.gw_city_lons)

        # 지구 반지름 곱하면 실제 거리지만, argmin만 필요하니까 c만 써도 됨
        idx = int(np.argmin(c))
        return float(self.gw_city_lats[idx]), float(self.gw_city_lons[idx])

    def pick_nearest_dst_vsg(self, lat, lon):
        gw_lat, gw_lon = self.nearest_gateway_coords(lat, lon)
        return get_vsg_id_from_coords(self.vsg_list, gw_lat, gw_lon)

    def generate_embb_cluster(self, hub_lat, hub_lon, radius_km, mode):
        """
        특정 eMBB 도시에서의 트래픽 생성 (Poisson + Pareto)
        """
        # 1초(1000ms)당 EMBB_ARRIVAL_RATE 만큼 발생한다고 가정
        # 1ms 당 평균 발생 횟수 (Lambda)
        lam = EMBB_ARRIVAL_RATE / 1000.0
        # Poisson 분포를 사용하여 이번 1ms에 발생한 패킷 수 결정
        # lam이 아주 작아도(예: 0.05), 확률적으로 0 또는 1이 나옵니다.
        num_packets = np.random.poisson(lam)

        if num_packets > 0:
            # Pareto Packet Sizes
            sizes = (np.random.pareto(EMBB_PARETO_SHAPE, num_packets) + 1) * EMBB_PACKET_MAX_SIZE

            for i in range(num_packets):
                # 위치: 도시 중심에서 Gaussian 분포로 퍼짐
                sigma = radius_km / 111.0 / 2
                p_lat = random.gauss(hub_lat, sigma)
                p_lon = random.gauss(hub_lon, sigma)

                pkt_size = min(int(sizes[i]), EMBB_PACKET_MAX_SIZE)

                src_vsg_id = get_vsg_id_from_coords(self.vsg_list, p_lat, p_lon)
                dst_vsg_id, dst_lat, dst_lon = self.pick_random_vsg_gw(range='continent')

                gsfc_type = 'eMBB'
                # TODO. gsfc가 변동될 것을 고려해서 dst_lat, dst_lon도 넘겨야하나?
                gsfc = GSFC(self.generated_gsfc_id, src_vsg_id, dst_vsg_id, pkt_size, SFC_EMBB_SEQ, EMBB_LATENCY_LIMIT, mode, gsfc_type, self.gsfc_log_path, p_lon, p_lat)
                self.gsfc_list.append(gsfc)
                self.generated_gsfc_id += 1

    def generate_urllc_cluster(self, hub_lat, hub_lon, radius_km, current_time_ms, mode):
        """
        특정 URLLC 핫스팟에서의 트래픽 생성 (Periodic Deterministic)
        """
        # 주기적 발생 체크 (2ms 마다)
        if current_time_ms % URLLC_PERIOD == 0:
            # 핫스팟 내 기기들이 동시다발적으로 전송한다고 가정 (Burst)
            num_devices = 2  # 핫스팟 당 기기 수

            for _ in range(num_devices):
                sigma = radius_km / 111.0 / 3  # URLLC는 더 좁게 밀집
                p_lat = random.gauss(hub_lat, sigma)
                p_lon = random.gauss(hub_lon, sigma)

                src_vsg_id = get_vsg_id_from_coords(self.vsg_list, p_lat, p_lon)
                dst_vsg_id = self.pick_nearest_dst_vsg(p_lat, p_lon) # Todo. src, dst vsg 동일하지 않게

                gsfc_type = 'URLLC'
                gsfc = GSFC(self.generated_gsfc_id, src_vsg_id, dst_vsg_id, URLLC_PACKET_SIZE, SFC_URLLC_SEQ, URLLC_LATENCY_LIMIT, mode,
                            gsfc_type, self.gsfc_log_path, p_lon, p_lat)
                self.gsfc_list.append(gsfc)
                self.generated_gsfc_id += 1

    def generate_global_mmtc(self, mode):
        """
        전 지구 배경 트래픽 (Uniform Random)
        """
        packets = []
        # 넓은 지역에서 드문드문 발생 (Time scaling)
        # 1ms 동안 100 * FACTOR 개 발생 (평균)
        # int()로 자르면 소수점이 사라지므로 Poisson 분포 사용 필수
        expected_num_per_ms = 100.0 * MMTC_DENSITY_FACTOR

        num_packets = np.random.poisson(expected_num_per_ms)

        for _ in range(num_packets):
            lat, lon = 0, 0
            while True:
                # 1. 랜덤 좌표 생성
                lat = random.uniform(LAT_RANGE[0], LAT_RANGE[1])
                lon = random.uniform(LON_RANGE[0], LON_RANGE[1])

                # 2. 조건 확인
                # 조건 A: 극지방인가? (위도 절댓값 60 이상)
                is_polar = abs(lat) >= 60

                # 조건 B: 바다인가? (육지가 아닌가?)
                # globe.is_land(lat, lon)은 육지면 True, 바다면 False를 반환
                is_ocean = not globe.is_land(lat, lon)

                # 둘 중 하나라도 만족하면 루프 탈출 (유효 좌표)
                if is_polar or is_ocean:
                    break

            src_vsg_id = get_vsg_id_from_coords(self.vsg_list, lat, lon)
            dst_vsg_id, dst_lat, dst_lon = self.pick_random_vsg_gw(range='continent')

            gsfc_type = 'mMTC'
            gsfc = GSFC(self.generated_gsfc_id, src_vsg_id, dst_vsg_id, MMTC_PACKET_SIZE, SFC_MMTC_SEQ, MMTC_LATENCY_LIMIT, mode,
                        gsfc_type, self.gsfc_log_path, lon, lat)
            self.gsfc_list.append(gsfc)
            self.generated_gsfc_id += 1

    def generate_traffic(self, current_sim_time_ms, mode):
        """
        [Main Generator] 수정됨
        MAJOR_HUBS 루프를 제거하고, 랜덤 게이트웨이 기반으로 트래픽 생성
        """

        # # ---------------------------------------------------------
        # # 1. eMBB 트래픽 생성 (Global Poisson Process)
        # # ---------------------------------------------------------
        # # 1ms 동안 전체 네트워크에서 발생할 기대 패킷 수 (Lambda)
        # # 만약 GLOBAL_EMBB_ARRIVAL_RATE가 20000이면, 1ms당 평균 20개
        # lam = EMBB_ARRIVAL_RATE / 1000.0
        #
        # # 이번 1ms에 실제로 발생할 이벤트(패킷 생성 요청) 수
        # num_embb_events = np.random.poisson(lam)
        #
        # if num_embb_events > 0:
        #     # 패킷 사이즈 미리 생성 (Pareto)
        #     sizes = (np.random.pareto(EMBB_PARETO_SHAPE, num_embb_events) + 1) * EMBB_PACKET_MAX_SIZE
        #
        #     for i in range(num_embb_events):
        #         # 1) 랜덤 출발지 선정
        #         src_lat, src_lon = self.pick_random_flight_path_point()
        #         src_vsg_id = get_vsg_id_from_coords(self.vsg_list, src_lat, src_lon)
        #
        #         # 2) 랜덤 도착지 선정 (Continent 레벨)
        #         dst_vsg_id, dst_lat, dst_lon = self.pick_random_vsg_gw(range='continent')
        #
        #         # src와 dst가 같다면 todo. 다시 뽑거나
        #         if src_vsg_id == dst_vsg_id:
        #             continue
        #
        #         pkt_size = min(int(sizes[i]), EMBB_PACKET_MAX_SIZE)
        #
        #         gsfc_type = 'eMBB'
        #         gsfc = GSFC(self.generated_gsfc_id, src_vsg_id, dst_vsg_id, pkt_size,
        #                     SFC_EMBB_SEQ, EMBB_LATENCY_LIMIT, mode, gsfc_type,
        #                     self.gsfc_log_path, src_lon, src_lat)
        #         self.gsfc_list.append(gsfc)
        #         self.generated_gsfc_id += 1

        # ---------------------------------------------------------
        # 2. URLLC 트래픽 생성 (Periodic Random Burst)
        # ---------------------------------------------------------
        # 특정 주기마다 '랜덤한 N개의 도시'에서 긴급 트래픽 발생
        NUM_ACTIVE_URLLC_SOURCES = 5 #10  # 예: 한 주기에 10개 도시 활성화

        if current_sim_time_ms % URLLC_PERIOD == 0:

            # 이번 틱에 활성화될 도시들을 랜덤으로 선정
            for _ in range(NUM_ACTIVE_URLLC_SOURCES):
                # 랜덤 도시 선정
                src_vsg_id, hub_lat, hub_lon = self.pick_random_vsg_gw(range='city')

                # 해당 도시 내 기기 수 (Burst) - 예: 2개 기기 동시 전송
                num_devices = 2

                for _ in range(num_devices):
                    radius_km = 30  # URLLC 커버리지
                    sigma = radius_km / 111.0 / 3
                    p_lat = random.gauss(hub_lat, sigma)
                    p_lon = random.gauss(hub_lon, sigma)

                    real_src_vsg_id = get_vsg_id_from_coords(self.vsg_list, p_lat, p_lon)
                    dst_vsg_id = self.pick_nearest_dst_vsg(p_lat, p_lon)

                    gsfc_type = 'URLLC'
                    gsfc = GSFC(self.generated_gsfc_id, real_src_vsg_id, dst_vsg_id,
                                URLLC_PACKET_SIZE, SFC_URLLC_SEQ, URLLC_LATENCY_LIMIT, mode,
                                gsfc_type, self.gsfc_log_path, p_lon, p_lat)
                    self.gsfc_list.append(gsfc)
                    self.generated_gsfc_id += 1

        # ---------------------------------------------------------
        # 3. mMTC 배경 트래픽 (기존 유지)
        # ---------------------------------------------------------
        self.generate_global_mmtc(mode)

    def interval_overlap(self, a_min, a_max, b_min, b_max, eps=1e-6):
        """
        열린 구간 겹침 여부 (길이가 0보다 크게 겹치는지)
        """
        return (a_min < b_max - eps) and (b_min < a_max - eps)

    def is_neighbor_vsg(self, v1, v2, eps=1e-6):
        """
        두 VSG 사각형이 변을 공유하면 True.
        - lat 방향으로 구간이 겹치고(long edge가 맞닿는 경우)
        - 혹은 lon 방향으로 구간이 겹치고(lat edge가 맞닿는 경우)
        """
        # 세로 방향 인접 (동/서 방향 이웃)
        lat_overlap = Simulation.interval_overlap(v1.lat_min, v1.lat_max,
                                                   v2.lat_min, v2.lat_max, eps)
        vertical_touch = lat_overlap and (
                abs(v1.lon_max - v2.lon_min) < eps or
                abs(v2.lon_max - v1.lon_min) < eps
        )

        # 가로 방향 인접 (남/북 방향 이웃)
        lon_overlap = Simulation.interval_overlap(v1.lon_min, v1.lon_max,
                                                   v2.lon_min, v2.lon_max, eps)
        horizontal_touch = lon_overlap and (
                abs(v1.lat_max - v2.lat_min) < eps or
                abs(v2.lat_max - v1.lat_min) < eps
        )

        return vertical_touch or horizontal_touch

    def rebuild_vsg_graph(self):
        """
        현재 self.vsg_list의 lon/lat 범위를 기반으로 vsg_G를 다시 만든다.
        """
        # 기존 그래프 비우기 (노드/엣지 모두 삭제)
        self.vsg_G = nx.Graph()

        # 1) 노드 추가
        for vsg in self.vsg_list:
            self.vsg_G.add_node(vsg.id)

        # 2) 모든 VSG 쌍에 대해 인접 여부 체크 후 edge 추가
        n = len(self.vsg_list)
        for i in range(n):
            v1 = self.vsg_list[i]
            for j in range(i + 1, n):
                v2 = self.vsg_list[j]

                if not self.is_neighbor_vsg(v1, v2):
                    continue

                # 가중치는 기존처럼 중심점 간 거리 사용
                dist = self.get_distance_between_VSGs(v1.id, v2.id)
                self.vsg_G.add_edge(v1.id, v2.id, weight=dist)

    def visualized_network_constellation(self, t, lon_step=None, filename="./results/network_constellation.png"):
        if lon_step is None:
            lon_step = LON_STEP

        # 경로 추적
        if (self.test_gsfc_id == -1) or (self.gsfc_list[self.test_gsfc_id].is_succeed) or (self.gsfc_list[self.test_gsfc_id].is_dropped):
            if len(self.gsfc_list) < 1:
                self.test_gsfc_id = -1
            else:
                # 1. 아직 완료되지 않은(is_succeed가 False인) GSFC들만 리스트로 추려냅니다.
                active_gsfcs = [g for g in self.gsfc_list if not g.is_succeed and not g.is_dropped]

                # 2. 후보가 하나라도 있다면 그 중에서 랜덤으로 선택하여 변경합니다.
                if active_gsfcs:
                    self.test_gsfc_id = random.choice(active_gsfcs).id
                else:
                    # 3. 모든 패킷이 succeed 상태라면, 아무것도 하지 않습니다 (기존 ID 유지).
                    pass

        # 컬러맵 생성 (VSG별 색상)
        cmap = cm.get_cmap('tab20', len(self.vsg_list))
        vsg_colors = {vsg.id: cmap(vsg.id) for vsg in self.vsg_list}

        fig = plt.figure(figsize=(15, 10))
        # 1. Cartopy의 PlateCarree 투영법(일반적인 위경도 평면) 사용
        ax = plt.axes(projection=ccrs.PlateCarree())
        # 2. 배경 설정
        ax.stock_img()  # 실제 지구 위성 사진 배경 (저화질 내장)
        ax.coastlines()  # 해안선 그리기
        # ax.add_feature(cfeature.BORDERS, linestyle=':') # 국경선 (선택)
        ax.add_feature(cfeature.LAND, alpha=0.3)  # 육지 색칠 (선택)

        ax = plt.gca()

        # 데이터 분리
        data = {
            "URLLC": [],
            "eMBB": [],
            "mMTC": [],
        }

        for gsfc in self.gsfc_list:
            if gsfc.gsfc_type in data:
                data[gsfc.gsfc_type].append((gsfc.lon, gsfc.lat))

        # 시나리오별 색상 및 마커 설정
        styles = {
            "URLLC": {"color": "red", "marker": "^", "s": 50, "label": "URLLC"},
            "eMBB": {"color": "green", "marker": "$\u2708$", "s": 150, "label": "eMBB"},
            "mMTC": {"color": "cyan", "marker": "*", "s": 50, "label": "mMTC"},
        }

        # 플롯 그리기 (레이어 순서: 배경(바다/극지) -> 육지 -> 도시 -> 핫스팟)
        draw_order = ["mMTC", "eMBB", "URLLC"]

        for key in draw_order:
            points = data[key]
            if points:
                lons, lats = zip(*points)
                style = styles[key]
                ax.scatter(lons, lats, c=style["color"], marker=style["marker"],
                            s=style["s"], label=style["label"], alpha=1, edgecolors='none')

        ax.legend(loc='lower left', markerscale=1.5, frameon=True, facecolor='white', framealpha=0.9)

        # 0. VSG 영역 표현
        for vsg in self.vsg_list:
            rect = Rectangle((vsg.lon_min, vsg.lat_min), lon_step, LAT_STEP,
                             linewidth=0.8, edgecolor=vsg_colors[vsg.id], facecolor=vsg_colors[vsg.id],
                             alpha=0.4, zorder=0)
            ax.add_patch(rect)

            if len(vsg.assigned_vnfs) > 1:
                plt.annotate(f"VNF {vsg.assigned_vnfs}", (vsg.lon_min+1, vsg.lat_min+1), fontsize=13, color='black',
                             alpha=0.8, zorder=12)

        # 1. ISL (adjacency edge) 그리기
        for sat in self.sat_list:
            for nbr_id in sat.adj_sat_index_list:
                if nbr_id == -1 or nbr_id >= len(self.sat_list):
                    continue
                nbr_sat = self.sat_list[nbr_id]
                ax.plot([sat.lon, nbr_sat.lon], [sat.lat, nbr_sat.lat],
                         color='gray', linewidth=1.0, alpha=0.3, zorder=1)

        # 2. VSG 영역 위성 산점도
        for vsg in self.vsg_list:
            for sat in vsg.satellites:
                ax.scatter(sat.lon, sat.lat, s=100, color=vsg_colors[vsg.id], edgecolors='black', linewidths=0.8,
                            alpha=0.6, zorder=2)

        # 3. VNF 수행 위성 강조
        for sat in self.sat_list:
            if sat.vnf_list:
                ax.scatter(sat.lon, sat.lat, marker='*', s=80, color='red', edgecolors='black', linewidths=0.8,
                            zorder=4)
                ax.annotate(f"VNF {sat.vnf_list}", (sat.lon + 3.0, sat.lat + 2.0), fontsize=13, color='darkred',
                             alpha=0.8, zorder=12)

        # 4. 위성 인덱스 모두 표시
        for sat in self.sat_list:
            ax.annotate(str(sat.id), (sat.lon, sat.lat), fontsize=13, alpha=0.7, zorder=5)

        # 경로 추적
        if self.test_gsfc_id != -1:
            try:
                # 1. 추적할 GSFC 객체 찾기
                gsfc_to_track = next(gsfc for gsfc in self.gsfc_list if gsfc.id == self.test_gsfc_id)

                # 2. 현재 경로 데이터 가져오기
                cur_path_idx = gsfc_to_track.cur_path_id
                processed_path = gsfc_to_track.satellite_path[:cur_path_idx] # 이미 지나간 경로 (초록색)
                remain_path = gsfc_to_track.satellite_path[cur_path_idx:] # 앞으로 갈 경로 (빨간색)

                # ----------------------------------------------------------------------
                # 3. 엣지 및 노드 추출
                # ----------------------------------------------------------------------
                processed_edges = []
                remaining_edges = []

                # 전체 경로를 추적하여 노드 목록을 통합합니다.
                processed_sat_ids = sat_ids(processed_path)
                remaining_sat_ids = sat_ids(remain_path)

                # --------------------------------------------------
                # A. Processed Path (초록색) 엣지 추출
                # --------------------------------------------------
                if processed_path:
                    # 첫 번째 홉의 시작 위성은 G/S 또는 이전 위성 목록에서 시작되므로,
                    # processed_path의 시작점에서부터 엣지를 추출합니다.
                    prev_sat_id = processed_path[0][0]
                    for sat_id, _ in processed_path[1:]:  # 첫 위성 다음부터 순회
                        if sat_id != prev_sat_id:
                            processed_edges.append((prev_sat_id, sat_id))
                        prev_sat_id = sat_id

                # --------------------------------------------------
                # B. Remaining Path (빨간색) 엣지 추출
                # --------------------------------------------------
                if remain_path:
                    # 1. 시작 위성 결정 (연결을 위한 이전 위성)
                    if processed_path:
                        # Processed Path가 있다면, 마지막 위성이 다음 홉의 출발점
                        prev_sat_id = processed_path[-1][0]
                        # 2. [핵심 수정] Processed Path와 Remain Path 연결 엣지 (빨간색)
                        # processed_path의 마지막 위성 -> remain_path의 첫 위성
                        current_sat_id = remain_path[0][0]
                        if current_sat_id != prev_sat_id:
                            # 이 엣지는 이제부터 '남은 경로'의 첫 엣지입니다.
                            remaining_edges.append((prev_sat_id, current_sat_id))
                        # 다음 루프를 위해 prev_sat_id를 remain_path의 첫 위성으로 업데이트
                        prev_sat_id = current_sat_id
                        # 루프 시작: remain_path[1:]부터 시작
                        start_index = 1
                    else:
                        # Processed Path가 없다면 (경로 시작점), remain_path[0]부터 시작
                        prev_sat_id = remain_path[0][0]
                        start_index = 1 if len(remain_path) > 1 else 0

                    # 3. 나머지 Remaining Path 엣지 생성
                    for sat_id, _ in remain_path[start_index:]:
                        if sat_id != prev_sat_id:
                            remaining_edges.append((prev_sat_id, sat_id))
                        prev_sat_id = sat_id

                # ----------------------------------------------------------------------
                # 4. 하이라이팅 시각화
                # ----------------------------------------------------------------------
                sat_positions = {sat.id: (sat.lon, sat.lat) for sat in self.sat_list}

                # 4-1. 노드 하이라이팅 (전체 경로 노드)
                all_tracked_sat_ids = list(set(processed_sat_ids + remaining_sat_ids))

                # 지나간 경로는 초록색, 남은 경로는 빨간색으로 노드를 구분하여 그릴 수 있습니다.
                # 여기서는 중복 없이 한번에 그립니다.
                nx.draw_networkx_nodes(self.G,
                                       pos=sat_positions,
                                       nodelist=all_tracked_sat_ids,
                                       node_color='gray',  # 기본 색상 (아래에서 덧그립니다)
                                       node_size=150,
                                       ax=ax,)

                # 4-2. Processed Path 엣지 (초록색)
                nx.draw_networkx_edges(self.G,
                                       pos=sat_positions,
                                       edgelist=processed_edges,
                                       edge_color='green',
                                       width=2.5,
                                       ax=ax,)

                # 4-3. Remaining Path 엣지 (빨간색)
                nx.draw_networkx_edges(self.G,
                                       pos=sat_positions,
                                       edgelist=remaining_edges,
                                       edge_color='red',
                                       width=2.5,
                                       ax=ax,)

                # 4-4. 현재 위치 위성 강조 (Processed Path의 마지막 위성)
                if processed_path:
                    current_sat_id = processed_path[-1][0]
                    nx.draw_networkx_nodes(self.G,
                                           pos=sat_positions,
                                           nodelist=[current_sat_id],
                                           node_color='yellow',  # 현재 위치는 노란색으로 강조
                                           node_size=200,
                                           ax=ax,)
            except StopIteration:
                # GSFC가 아직 생성되지 않았거나 ID가 잘못된 경우
                pass

        plt.xlim([-180, 180])
        plt.ylim([-90, 90])
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Network Constellation at {t}ms")
        plt.grid(True)
        plt.tight_layout()

        fig.canvas.draw()

        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape((height, width, 3))

        # VideoWriter 초기화 (첫 프레임에서)
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc,
                self.video_fps,
                (width, height)
            )

        # OpenCV는 BGR이라 변환 후 write
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame_bgr)

        plt.close(fig)

    def get_progress(self):
        total = len(self.gsfc_list)
        success = 0
        dropped = 0
        working_list = []

        for gsfc in self.gsfc_list:
            if gsfc.is_succeed:
                success += 1
            elif gsfc.is_dropped:
                dropped += 1
            else:
                working_list.append(gsfc.id)

        return total, success, dropped, working_list

    def simulation_proceeding(self, mode, lon_step, data_rate_pair, csv_dir_path):
        self.gsfc_log_path = init_gsfc_csv_log(csv_dir_path, mode)
        self.sat_log_path = init_sat_csv_log(csv_dir_path, mode)
        self.vsg_log_path = init_vsg_csv_log(csv_dir_path, mode)
        self.video_path = os.path.join(csv_dir_path, f"{mode}_network_constellation.mp4")

        # 1. 토폴로지 초기화
        self.set_constellation(mode)
        self.initial_vsg_regions(mode, lon_step)
        self.initial_vnfs_to_vsg()
        self.compute_all_pair_distance(csv_dir_path, mode)

        self.visualized_network_constellation(t=0)

        new_gsfc_id_start = 0
        t = 0 #ms

        while True:
            # gsfc 생성
            if t < NUM_ITERATIONS:
                # self.generate_gsfc(new_gsfc_id_start, mode)
                self.generate_traffic(t, mode)
                self.visualized_network_constellation(t)
            # gsfc 초기 경로 생성
            for gsfc in self.gsfc_list[new_gsfc_id_start:]:
                if mode == "dd":
                    gsfc.set_dd_satellite_path(self.vsg_list, self.sat_list, self.G)
                else:
                    gsfc.set_gsfc_flow_rule(self.vsg_list, self.vsg_G)
                    gsfc.set_vsg_path(self.vsg_list, self.vsg_G)
                    if mode == "basic":
                        gsfc.set_basic_satellite_path(self.vsg_list, self.G)
                    elif "sd" in mode:
                        gsfc.set_sd_satellite_path(self.vsg_list, self.gserver_list, self.sat_list, self.G, self.vsg_G)
                    else:
                        print("\n")
                        print(f"[ERROR] Unknown mode {mode}")
                        print("\n")
                        return []
                write_gsfc_csv_log(self.gsfc_log_path, t, gsfc, "INIT_PATH")
                new_gsfc_id_start += 1

            # 종료 여부 파악
            all_completed = True
            for gsfc in self.gsfc_list:
                if not gsfc.is_succeed and not gsfc.is_dropped:
                    all_completed = False
                    break

            if all_completed:
                print(f"\n*** 모든 GSFC가 succeed 또는 dropped 상태로 완료되었습니다. 시뮬레이션을 종료합니다. ***")
                break

            # gsfc 처리
            for gsfc in self.gsfc_list:
                gsfc.time_tic(t, data_rate_pair, self.vsg_list, self.gserver_list, self.sat_list, self.G, self.vsg_G)

            # 위성 이동
            for sat in self.sat_list:
                sat.time_tic(t)

            # vsg 구역 재설정 # TODO. vsg_G 갱신
            lost_vnf_type_list = {}  # key: vsg_id, value: lost_vnf_types
            is_topology_changed = False

            # while i < len(self.vsg_list): # 중간에 vsg 추가되어도 추가된 vsg까지 검사하기 위함
            for vsg in self.vsg_list:
                # vsg = self.vsg_list[i]
                lost_vnf_type_list[vsg.id] = []
                lost_vnf_types = vsg.time_tic(self.vsg_list, self.sat_list, self.gsfc_list, t)
                lost_vnf_type_list[vsg.id].extend(lost_vnf_types)

                if lost_vnf_types:
                    is_topology_changed = True
                    print(f"====== CHANGE TOPOLOGY ======")
                    print(f"  -> vsg id {vsg.id} lost vnf types {lost_vnf_types}")

            if is_topology_changed:
                self.rebuild_vsg_graph()
                # gsfc 경로 재설정
                for gsfc in self.gsfc_list:
                    gsfc.detour_satellite_path(lost_vnf_type_list, self.vsg_list, self.gserver_list, self.sat_list, self.G, mode)

            total, success, dropped, working_list = self.get_progress()
            print(f"TIME TICK {t}MS --- Working....... --- TOTAL: {total} SUCCESS: {success} DROPPED: {dropped}")

            self.visualized_network_constellation(t)
            t += 1

        # 루프가 끝나고 나서
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"[INFO] Saved video to {self.video_path}")