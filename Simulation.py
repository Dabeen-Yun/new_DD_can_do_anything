from Params import *
from Satellite import *
from Gserver import *
from VSG import *
from GSFC import *

import numpy as np
import random
import networkx as nx
import math
from time import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle


d2r = np.deg2rad

class Simulation:
    def __init__(self):
        np.random.seed(921)
        random.seed(921)

        # topology
        self.sat_list = []
        self.vsg_list = []
        self.gserver_list = []
        self.gsfc_list = []

        self.G = None
        self.vsg_G = None

        self.gsfc_log_path = ""

        self.test_gsfc_id = -1 # 경로 추적을 위한 GSFC ID 변수

    def set_constellation(self):
        phasing_inter_plane = 180 / NUM_ORBITS

        for sat_id in range(NUM_SATELLITES):
            sat = Satellite(sat_id, NUM_ORBITS, NUM_SATELLITES_PER_ORBIT, ORBIT_ALTITUDE, phasing_inter_plane,
                            POLAR_LATITUDE, self.sat_list)
            self.sat_list.append(sat)

        for sat_id in range(NUM_SATELLITES):
            sat = self.sat_list[sat_id]
            sat.set_adjacent_node()
            sat.get_propagation_delay()

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

    def initial_vsg_regions(self):
        self.vsg_list = []
        self.gserver_list = []
        self.vsg_G = nx.Graph()

        vid = 0
        gid = 0

        lat_bins = np.arange(LAT_RANGE[0], LAT_RANGE[1]+1, LAT_STEP)
        lon_bins = np.arange(LON_RANGE[0], LON_RANGE[1]+1, LON_STEP)

        num_row = math.ceil((LAT_RANGE[1] - LAT_RANGE[0]) / LAT_STEP)
        num_col = math.ceil((LON_RANGE[1] - LON_RANGE[0]) / LON_STEP)

        for lat_min in lat_bins:
            lat_max = lat_min + LAT_STEP
            for lon_min in lon_bins:
                lon_max = lon_min + LON_STEP

                # 현재 그리드 셀 안에 속하는 위성 추출
                cell_sats = [
                    sat for sat in self.sat_list
                    if lat_min <= sat.lat < lat_max and lon_min <= sat.lon < lon_max
                ]

                if not cell_sats:
                    continue

                center_lat = (lat_min + LAT_STEP) / 2
                center_lon = (lon_min + LON_STEP) / 2

                ground_server = Gserver(gid, center_lon, center_lat, vid)

                vsg = VSG(vid, (center_lon, center_lat), lon_min, lat_min, cell_sats, ground_server)
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

    def initial_vnfs_to_vsg(self):
        # NUM_SATELLITES는 self.sat_list의 길이 또는 전역 상수를 사용합니다.
        vnf_set_sat_ids = sorted(random.sample(range(0,NUM_SATELLITES), int(NUM_SATELLITES*0.8)))

        for sat in self.sat_list:
            if sat.id in vnf_set_sat_ids:
                # 3개 이상 탑재 (최대 개수는 넘기지 않도록)
                vnf_per_sat = random.randint(3,NUM_VNFS_PER_SAT) # 흠...
                vnfs = sorted(random.sample(range(*VNF_TYPES_PER_VSG), vnf_per_sat))
                assigned_vnfs = [str(v) for v in vnfs]
                sat.vnf_list = assigned_vnfs

        for vsg in self.vsg_list:
            sampled_vnf_types = random.sample(range(VNF_TYPES_PER_VSG[0], VNF_TYPES_PER_VSG[1] + 1), k=NUM_VNFS_PER_VSG)
            assigned_vnfs = [str(v) for v in sampled_vnf_types]
            vsg.assigned_vnfs = assigned_vnfs

            for vnf_type in assigned_vnfs:
                # 2-1. 현재 VSG 내에 해당 VNF를 호스팅하는 위성이 있는지 확인
                is_covered = any(vnf_type in sat.vnf_list for sat in vsg.satellites)
                if is_covered:
                    continue  # 이미 커버됨

                target_sat = random.choice(vsg.satellites)

                if len(target_sat.vnf_list) < NUM_VNFS_PER_SAT:
                    # 공간이 남은 경우: 추가 (Addition)
                    target_sat.vnf_list.append(vnf_type)
                else:
                    # 공간이 없는 경우: 교체 (Replacement)

                    # ✨ [수정된 핵심 로직]: VSG에 할당된 VNF가 아닌 것을 제거 대상으로 선택
                    non_assigned_vnfs = [v for v in target_sat.vnf_list if v not in assigned_vnfs]
                    victim_vnf = random.choice(non_assigned_vnfs)
                    target_sat.vnf_list.remove(victim_vnf)
                    target_sat.vnf_list.append(vnf_type)

            print(f"  -> VSG {vsg.id} 내 할당 vnfs {vsg.assigned_vnfs}")

    def compute_all_pair_distance(self):
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

    def generate_gsfc(self, new_gsfc_id_start, mode, num_gsfcs=None):
        if num_gsfcs is None:
            num_gsfcs = NUM_GSFC

        gsfc_id = new_gsfc_id_start

        for i in range(num_gsfcs):
            sfc_type_idx = random.randint(0, 2)
            vnf_sequence = SFC_TYPE_LIST.get(sfc_type_idx)
            tolerance_time_ms = SFC_TOLERANCE_TIME.get(sfc_type_idx)

            # SFC 종류 별 경로 설정
            # if sfc_type_idx == 5: # uRLLC:
            #     # SRC VSG와 DST VSG가 동일하도록
            #     src_vsg = random.choice(self.vsg_list)
            #     src_vsg_id = src_vsg.id
            #     dst_vsg_id = src_vsg_id
            # else:
            src_vsg = random.choice(self.vsg_list)
            src_vsg_id = src_vsg.id
            src_lon = src_vsg.center_coords[0]  # (lon, lat)이므로 [0]이 lon

            # 2. dst_vsg 후보: src_vsg_id를 제외하고, 경도가 src_vsg의 경도보다 큰 VSG들
            # 이렇게 하면 'src가 왼쪽, dst가 오른쪽' 조건이 만족됩니다.
            dst_candidates = [
                v for v in self.vsg_list
                if v.id != src_vsg_id and v.center_coords[0] > src_lon
            ]

            if dst_candidates:
                # 조건(src_lon < dst_lon)을 만족하는 후보가 있으면 그 중에서 무작위 선택
                dst_vsg_id = random.choice(dst_candidates).id
            else:
                # 조건(src_lon < dst_lon)을 만족하는 후보가 없거나, src_vsg 밖에 없는 경우
                # (예: src가 가장 동쪽에 있는 VSG인 경우)
                # 3. 차선책: src_vsg_id를 제외한 나머지 모든 VSG 중에서 무작위 선택
                other_vsgs = [v for v in self.vsg_list if v.id != src_vsg_id]
                if other_vsgs:
                    dst_vsg_id = random.choice(other_vsgs).id
                else:
                    # VSG가 하나뿐인 경우. 생성을 건너뜁니다.
                    print(f"[WARNING] Only one VSG found (ID: {src_vsg_id}). Skipping GSFC creation.")
                    continue  # 다음 루프로 이동

            gsfc = GSFC(gsfc_id, src_vsg_id, dst_vsg_id, vnf_sequence, tolerance_time_ms, mode, self.gsfc_log_path)
            print("gsfc list : ", gsfc_id, src_vsg_id, vnf_sequence, dst_vsg_id)
            self.gsfc_list.append(gsfc)
            gsfc_id += 1

    def visualized_network_constellation(self, t, filename="./results/network_constellation.png"):
        # 경로 추적
        if (self.test_gsfc_id == -1) or (self.gsfc_list[self.test_gsfc_id].is_succeed):
            if len(self.gsfc_list) < 1:
                self.test_gsfc_id = -1
            else:
                while True:
                    self.test_gsfc_id = random.choice(self.gsfc_list).id
                    if not self.gsfc_list[self.test_gsfc_id].is_succeed:
                        break

        # 컬러맵 생성 (VSG별 색상)
        cmap = cm.get_cmap('tab20', len(self.vsg_list))
        vsg_colors = {vsg.id: cmap(vsg.id) for vsg in self.vsg_list}

        plt.figure(figsize=(24, 12))

        # 0. VSG 영역 표현
        for vsg in self.vsg_list:
            rect = Rectangle((vsg.lon_min, vsg.lat_min), LON_STEP, LAT_STEP,
                             linewidth=0.8, edgecolor=vsg_colors[vsg.id], facecolor=vsg_colors[vsg.id],
                             alpha=0.4, zorder=0)
            plt.gca().add_patch(rect)

            if len(vsg.assigned_vnfs) > 1:
                plt.annotate(f"VNF {vsg.assigned_vnfs}", (vsg.lon_min+1, vsg.lat_min+1), fontsize=13, color='black',
                             alpha=0.8, zorder=12)

        # 1. ISL (adjacency edge) 그리기
        for sat in self.sat_list:
            for nbr_id in sat.adj_sat_index_list:
                if nbr_id == -1 or nbr_id >= len(self.sat_list):
                    continue
                nbr_sat = self.sat_list[nbr_id]
                plt.plot([sat.lon, nbr_sat.lon], [sat.lat, nbr_sat.lat],
                         color='gray', linewidth=1.0, alpha=0.3, zorder=1)

        # 2. VSG 영역 위성 산점도
        for vsg in self.vsg_list:
            for sat in vsg.satellites:
                edge = 'black'
                lw = 0.8
                plt.scatter(sat.lon, sat.lat, s=100, color=vsg_colors[vsg.id], edgecolors=edge, linewidths=lw,
                            alpha=0.6, zorder=2)

        # 3. VNF 수행 위성 강조
        for sat in self.sat_list:
            if sat.vnf_list:
                plt.scatter(sat.lon, sat.lat, marker='*', s=80, color='red', edgecolors='black', linewidths=0.8,
                            zorder=4)
                plt.annotate(f"VNF {sat.vnf_list}", (sat.lon + 3.0, sat.lat + 2.0), fontsize=13, color='darkred',
                             alpha=0.8, zorder=12)

        # 4. 위성 인덱스 모두 표시
        for sat in self.sat_list:
            plt.annotate(str(sat.id), (sat.lon, sat.lat), fontsize=13, alpha=0.7, zorder=5)

        # 경로 추적
        if self.test_gsfc_id != -1:
            try:
                # 1. 추적할 GSFC 객체 찾기
                gsfc_to_track = next(gsfc for gsfc in self.gsfc_list if gsfc.id == self.test_gsfc_id)

                # 2. 현재 경로 데이터 가져오기
                processed_path = gsfc_to_track.processed_satellite_path # 이미 지나간 경로 (초록색)
                remain_path = get_remain_path(gsfc_to_track) # 앞으로 갈 경로 (빨간색)

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
                                       ax=plt.gca(),)

                # 4-2. Processed Path 엣지 (초록색)
                nx.draw_networkx_edges(self.G,
                                       pos=sat_positions,
                                       edgelist=processed_edges,
                                       edge_color='green',
                                       width=2.5,
                                       ax=plt.gca(),)

                # 4-3. Remaining Path 엣지 (빨간색)
                nx.draw_networkx_edges(self.G,
                                       pos=sat_positions,
                                       edgelist=remaining_edges,
                                       edge_color='red',
                                       width=2.5,
                                       ax=plt.gca(),)

                # 4-4. 현재 위치 위성 강조 (Processed Path의 마지막 위성)
                if processed_path:
                    current_sat_id = processed_path[-1][0]
                    nx.draw_networkx_nodes(self.G,
                                           pos=sat_positions,
                                           nodelist=[current_sat_id],
                                           node_color='yellow',  # 현재 위치는 노란색으로 강조
                                           node_size=200,
                                           ax=plt.gca(),)
            except StopIteration:
                # GSFC가 아직 생성되지 않았거나 ID가 잘못된 경우
                pass

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Network Constellation at {t}ms")
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.tight_layout()
        # plt.savefig(filename, dpi=300)
        plt.show()

    def simulation_proceeding(self, mode, data_rate_pair, csv_dir_path):
        # 1. 토폴로지 초기화
        self.set_constellation()
        self.initial_vsg_regions()
        self.initial_vnfs_to_vsg()
        self.compute_all_pair_distance()

        self.gsfc_log_path = init_gsfc_csv_log(csv_dir_path, mode)
        # self.visualized_network_constellation(t=0)

        new_gsfc_id_start = 0
        t = 0 #ms

        for i in range(500):
            print(f"\n==================== TIME TICK {t} MS ====================")

            # gsfc 생성
            if t < NUM_ITERATIONS:
                self.generate_gsfc(new_gsfc_id_start, mode)
            # gsfc 초기 경로 생성
            for gsfc in self.gsfc_list[new_gsfc_id_start:]:
                if mode == "dd":
                    gsfc.set_dd_satellite_path(self.vsg_list, self.sat_list, self.G)
                else:
                    gsfc.set_gsfc_flow_rule(self.vsg_list, self.vsg_G)
                    gsfc.set_vsg_path(self.vsg_G)
                    if mode == "basic":
                        gsfc.set_basic_satellite_path(self.vsg_list, self.G)
                    elif mode == "noname":
                        gsfc.set_noname_satellite_path(self.vsg_list, self.gserver_list, self.sat_list, self.G, self.vsg_G)
                    else:
                        print("\n")
                        print(f"[ERROR] Unknown mode {mode}")
                        print("\n")
                        return []
                write_gsfc_csv_log(self.gsfc_log_path, t, gsfc, "INIT_PATH")
                new_gsfc_id_start += 1

            # gsfc 처리
            for gsfc in self.gsfc_list:
                gsfc.time_tic(t, data_rate_pair, self.vsg_list, self.gserver_list, self.sat_list, self.G, self.vsg_G)

            # 위성 이동
            for sat in self.sat_list:
                sat.time_tic(t)

            # vsg 구역 재설정
            lost_vnf_type_list = {}  # key: vsg_id, value: lost_vnf_types
            is_topology_changed = False

            for vsg in self.vsg_list:
                lost_vnf_type_list[vsg.id] = []
                lost_vnf_types = vsg.time_tic(self.sat_list, self.gsfc_list)
                lost_vnf_type_list[vsg.id].extend(lost_vnf_types)

                if lost_vnf_types:
                    is_topology_changed = True
                    print(f"====== CHANGE TOPOLOGY ======")
                    print(f"  -> vsg id {vsg.id} lost vnf types {lost_vnf_types}")

            if is_topology_changed:
                # gsfc 경로 재설정
                for gsfc in self.gsfc_list:
                    gsfc.detour_satellite_path(lost_vnf_type_list, self.vsg_list, self.gserver_list, self.sat_list, self.G, mode)

            # self.visualized_network_constellation(t)

            t += 1