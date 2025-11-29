from Params import *
import numpy as np
from Utility import *

import random
import math

class VSG:
    def __init__(self, id, center_coords, lon_min, lat_min, satellites, gserver, mode, vsg_log_path):
        self.id = id
        self.mode = mode
        self.vsg_log_path = vsg_log_path
        self.assigned_vnfs = []
        self.satellites = satellites
        self.center_coords = center_coords  # lon, lat
        self.lon_min = lon_min
        self.lon_max = self.lon_min + LON_STEP
        self.lat_min = lat_min
        self.lat_max = self.lat_min + LAT_STEP
        self.gserver = gserver

        self.time = 0

    def update_satellite_in_vsg(self, all_sat_list):
        is_changed = False

        # 현재 그리드 셀 안에 속하는 위성 추출
        cell_sats = [
            sat for sat in all_sat_list
            if self.lat_min <= sat.lat < self.lat_max and self.lon_min <= sat.lon < self.lon_max
        ]

        for sat in cell_sats:
            sat.current_vsg_id = self.id

        if not is_changed:
            is_changed = self.satellites != cell_sats
            if is_changed:
                self.satellites = cell_sats

        return is_changed

    def reassign_vnfs_to_satellite(self, all_gsfc_list):
        lost_vnf_types = []

        for vnf in self.assigned_vnfs:
            has_vnf = any(vnf in sat.vnf_list for sat in self.satellites)
            if has_vnf:
                continue
            lost_vnf_types.append(vnf)

            # ----------------------------------------------------------------------
            # 🎯 [우선 순위 1]: VNF 최대 개수 미만인 위성에게 할당
            # ----------------------------------------------------------------------
            capacity_candidate = []
            for sat in self.satellites:
                # 1. VNF 최대 개수(1) 미만인 위성
                if len(sat.vnf_list) < NUM_VNFS_PER_SAT:
                    capacity_candidate.append(sat)

            if capacity_candidate:
                # VNF 슬롯이 비어있는 위성에게 할당하고 다음 VNF로 넘어갑니다.
                selected_sat = random.choice(capacity_candidate)
                selected_sat.vnf_list.append(vnf)
                # print(f"[REASSIGN] VNF **{vnf}** assigned to Sat **{selected_sat.id}** (Simple Capacity Check: {len(selected_sat.vnf_list)}/{NUM_VNFS_PER_SAT}).")
                continue  # 다음 vnf로 넘어감

            # ----------------------------------------------------------------------
            # 🎯 [우선 순위 2]: VNF 슬롯이 가득 찼거나 없으므로, 로드 밸런싱을 통해 할당
            # ----------------------------------------------------------------------
            # # 1. random 방식
            # best_sat = None
            # best_vnf_kind_in_sat = None
            #
            # candidates = []  # (sat, vnf_kind) 튜플 저장
            #
            # for sat in self.satellites:
            #     vnf_loads_dict = get_satellite_load(sat, all_gsfc_list)
            #
            #     for vnf_kind in vnf_loads_dict.keys():
            #         # 2. 현재 VSG에 이미 할당된 VNF는 제외
            #         if vnf_kind in self.assigned_vnfs:
            #             continue
            #
            #         candidates.append((sat, vnf_kind))
            #
            # if candidates:
            #     best_sat, best_vnf_kind_in_sat = random.choice(candidates)
            # else:
            #     pass

            # 2. 잔류시간만 고려
            max_time_entered = 0
            best_sat = None
            best_vnf_kind_in_sat = None

            for sat in self.satellites:
                vnf_loads_dict = get_satellite_load(sat, all_gsfc_list)

                for vnf_kind, load in vnf_loads_dict.items():
                    if vnf_kind not in self.assigned_vnfs:
                        time_entered = sat.vsg_enter_time

                        if time_entered > max_time_entered:
                            best_sat = sat
                            best_vnf_kind_in_sat = vnf_kind
                            max_time_entered = time_entered


            # # 3. queue 상태 + 잔류 시간
            # best_sat = None
            # best_vnf_kind_in_sat = None
            # best_efficiency = -1
            # alpha = 0.5
            #
            # for sat in self.satellites:
            #     max_time_entered = sat.vsg_enter_time
            #
            #     # 2-1. 위성(sat)의 VNF 종류별 로드 딕셔너리를 가져옵니다.
            #     vnf_loads_dict = get_satellite_load(sat, all_gsfc_list)
            #
            #     for vnf_kind, load in vnf_loads_dict.items():
            #         # VSG에 할당된 VNF는 무시하고 (nothing), 할당되지 않은 VNF만 검사
            #         if vnf_kind not in self.assigned_vnfs:
            #             efficiency = alpha * max_time_entered - (1 - alpha) * load # 클 수록 좋음 (늦게 들어옴), load는 작을 수록 좋음
            #             if efficiency > best_efficiency:
            #                 best_sat = sat
            #                 best_efficiency = efficiency

            # 3. 할당 가능한 위성이 있는지 확인
            if best_sat is None:
                # Capacity Check도 실패했고, Load Balancing으로도 후보를 찾지 못한 경우
                print(f"[ERROR] Cannot assign VNF {vnf} in VSG {self.id}. All satellites are full or unavailable.")
                continue

            # 4. 가장 로드가 적은 위성(best_sat)에 VNF 할당 (재할당 로직 실행)
            selected_sat = best_sat
            selected_sat.vnf_list.remove(best_vnf_kind_in_sat)
            selected_sat.vnf_list.append(vnf)

            # 5. 재할당 정보 출력 # TODO. 남의꺼 뺏을 때만 로그 찍히게 --> 왜 detour 안되는지 확인
            print(f"[REASSIGN] VNF **{vnf}** assigned to Sat **{selected_sat.id}** in VSG **{self.id}**.")
            # print(
            #     f"           Selection Criterion: Found minimum queue process (Load: **{min_overall_load:.2f}** bytes) across the VSG.")
            print(
                f"           The least loaded VNF Queue was **{best_vnf_kind_in_sat}** on Sat **{selected_sat.id}** (Filtering out VSG {self.id}'s assigned VNFs).")

        return lost_vnf_types

    def time_tic(self, all_sat_list, all_gsfc_list, cur_time):
        self.time = cur_time

        is_inconsistent = False
        lost_vnfs = []
        is_changed = self.update_satellite_in_vsg(all_sat_list)

        if is_changed:
            found_vnfs = set()
            for sat in self.satellites:
                if sat.vnf_list:
                    for vnf in sat.vnf_list:
                        found_vnfs.add(vnf)

            # 한 개라도 없는 VNF가 있다면 재할당 필요
            for vnf in self.assigned_vnfs:
                if vnf not in found_vnfs:
                    is_inconsistent = True

        if is_inconsistent:
            lost_vnfs = self.reassign_vnfs_to_satellite(all_gsfc_list)

        write_vsg_csv_log(self)

        return lost_vnfs