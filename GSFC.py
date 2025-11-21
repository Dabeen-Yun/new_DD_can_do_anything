from Params import *
from Utility import *

import networkx as nx
import random
from statistics import mean
from itertools import product
import os
import csv
import math

class GSFC:
    def __init__(self, gsfc_id, src_vsg_id, dst_vsg_id, vnf_sequence, tolerance_time_ms, mode, gsfc_log_path):
        self.id = gsfc_id
        self.mode = mode # basic, dd, sd
        self.vnf_sequence = vnf_sequence
        self.src_vsg_id = src_vsg_id
        self.dst_vsg_id = dst_vsg_id
        self.vnf_sizes = [] #value: total vnf sizes [bit]
        self.vnf_id = 0 #processing vnf id
        self.sfc_size = 0
        self.gserver = None
        self.is_keeping = False # mMTC 시나리오에서 vnf 처리 가능할 때까지 기다리기 위한 플래그
        self.tolerance_time_ms = tolerance_time_ms # mMTC 시나리오에서 기다릴 수 있는 최대 시간 [ms]
        self.gsfc_log_path = gsfc_log_path

        self.cur_vsg_path_id = 0 # sd 변수 (현재 satellite_path를 만드는데 사용한 vsg_idx)
        self.processed_vnfs_size = []
        self.gsfc_flow_rule = [] # 필수 VSG 경로 구성
        self.vsg_path = [] # 실제 VSG 경로 구성
        self.satellite_path = []
        self.cur_path_id = 0 # 현재 실행해야하는 path 인덱스
        self.node_id_to_move = -1 # 이동해야할 sat id

        self.proc_delay_ms = 0
        self.prop_delay_ms = 0
        self.trans_delay_ms = 0
        self.queue_delay_ms = 0
        self.is_dropped = False
        self.is_succeed = False

        self.estimated_remain_proc_delay = 0
        self.state = 1 # 1: process, 2: queue, 3: transmit, 4: propagate
        self.is_appended = False # sd 방식에서 경로 연장에 대한 변수
        self.DH_remaining_ongoing_time_slot = 0
        self.current_remaining_vnf_process = 0

        self.set_vnf_sizes()

    def set_vnf_sizes(self):
        for vnf in self.vnf_sequence:
            self.vnf_sizes.append(VNF_SIZE)
            self.processed_vnfs_size.append(0)
        self.sfc_size = sum(self.vnf_sizes)

    # 생성된 gsfc를 기반으로 vsg path 구성
    def set_gsfc_flow_rule(self, all_vsg_list, vsg_G):
        self.gsfc_flow_rule.append(("src", self.src_vsg_id))

        prev_vsg_id = self.src_vsg_id

        for idx, vnf in enumerate(self.vnf_sequence):
            candidate_vsgs = []
            for vsg in all_vsg_list:
                if vnf in vsg.assigned_vnfs:
                    available_sats = [
                        sat for sat in vsg.satellites
                        if vnf in sat.vnf_list
                    ]
                    if available_sats:
                        candidate_vsgs.append(vsg)

            if not candidate_vsgs: # 해당 VNF를 수행할 VSG가 없음
                print(f"[ERROR] 1-1 No non-congested VSG found for VNF {vnf}")
                self.gsfc_flow_rule = []
                self.is_dropped = True
                return []

            try:
                hop_sorted = sorted(
                    candidate_vsgs,
                    key=lambda vsg: nx.shortest_path_length(vsg_G, source=prev_vsg_id, target=vsg.id)
                )
            except nx.NetworkXNoPath:
                print(f"[ERROR] 1-2 No path from src VSG to any candidate VSG for VNF {vnf}")
                self.gsfc_flow_rule = []
                self.is_dropped = True
                return []

            for vsg in hop_sorted:
                self.gsfc_flow_rule.append(('vnf'+vnf, vsg.id))
                prev_vsg_id = vsg.id
                break

        self.gsfc_flow_rule.append(("dst", self.dst_vsg_id))

    def set_vsg_path(self, vsg_G):
        essential_vsgs = self.gsfc_flow_rule

        items = list(essential_vsgs)
        n = len(items)
        if n < 1:
            return []

        for i in range(n - 1):
            (src_vnf, src_vsg) = items[i]
            (dst_vnf, dst_vsg) = items[i + 1]

            try:
                sub_path = nx.shortest_path(vsg_G, source=src_vsg, target=dst_vsg)
            except nx.NetworkXNoPath:
                print(f"[ERROR] 2-1 No path between VSG {src_vsg} and {dst_vsg}")
                self.is_dropped = True
                return []

            # 1) 첫 구간이면 시작 VSG + vnf 태그 저장
            if i == 0:
                self.vsg_path.append((src_vsg, src_vnf))

            # 2) 중간 VSG들 (None 태그)
            for vid in sub_path[1:-1]:
                self.vsg_path.append((vid, None))

            # 3) 도착 VSG + vnf 태그 저장
            self.vsg_path.append((dst_vsg, dst_vnf))

    def set_basic_satellite_path(self, all_vsg_list, G):
        prev_sat_id = -1
        len_vsg_path = len(self.vsg_path)
        if len_vsg_path < 1:
            print(f"[ERROR] No VSG path in GSFC {self.id}")
            return []

        for i in range(len_vsg_path - 1):
            src_vsg, src_vnf = self.vsg_path[i]
            dst_vsg, dst_vnf = self.vsg_path[i + 1]

            if prev_sat_id == -1:
                is_vnf = has_vnf_tag(src_vnf)
                if is_vnf:
                    current_vnf_id = get_vnf_id_for_list(src_vnf)
                    candidate_src_sat_ids = [
                        sat.id for sat in all_vsg_list[src_vsg].satellites
                        if current_vnf_id in sat.vnf_list
                    ]
                else:
                    candidate_src_sat_ids = [
                        sat.id for sat in all_vsg_list[src_vsg].satellites
                    ]
                if not candidate_src_sat_ids:
                    print(f"[ERROR] 3-1 No SATELLITE TO SRC")
                    self.is_dropped = True
                    return []
                src_sat_id = random.choice(candidate_src_sat_ids)
                prev_sat_id = src_sat_id

            is_vnf = has_vnf_tag(dst_vnf)
            if is_vnf:
                current_vnf_id = get_vnf_id_for_list(dst_vnf)
                candidate_dst_sat_ids = [
                    sat.id for sat in all_vsg_list[dst_vsg].satellites
                    if current_vnf_id in sat.vnf_list
                ]
            else:
                candidate_dst_sat_ids = [
                    sat.id for sat in all_vsg_list[dst_vsg].satellites
                ]

            if not candidate_dst_sat_ids:
                print(f"[ERROR] 3-1 No SATELLITE TO DST")
                self.is_dropped = True
                return []
            dst_sat_id = random.choice(candidate_dst_sat_ids)

            if prev_sat_id == dst_sat_id: # 이동 X
                if i == 0:
                    self.satellite_path.append([prev_sat_id, src_vnf])
                self.satellite_path.append([dst_sat_id, dst_vnf])
                prev_sat_id = dst_sat_id
            else:
                try:
                    sub_path = nx.shortest_path(G, source=prev_sat_id, target=dst_sat_id)

                    if i == 0:
                        self.satellite_path.append([prev_sat_id, src_vnf])
                    if len(sub_path) > 2:
                        for sid in sub_path[1:-1]:
                            self.satellite_path.append([sid, None])
                    self.satellite_path.append([dst_sat_id, dst_vnf])
                    prev_sat_id = dst_sat_id
                except nx.NetworkXNoPath:
                    print(f"[ERROR] 3-2 No TRAJECTORY SAT TO SAT")
                    self.is_dropped = True
                    return []

        print(f"\n========== INITIAL BASIC GSFC {self.id} ==========")
        print(f"  -> src {self.src_vsg_id}")
        print(f"  -> dst {self.dst_vsg_id}")
        print(f"  -> vnf sequence {self.vnf_sequence}")
        print(f"  -> flow rule {self.gsfc_flow_rule}")
        print(f"  -> vsg path {self.vsg_path}")
        print(f"  -> satellite path {self.satellite_path}")

    def update_basic_satellite_path(self, all_vsg_list, all_gserver_list, all_sat_list, G):
        if len(self.satellite_path) < 1:
            # self.basic_satellite_path(all_vsg_list, G)
            print(f"[WARN] INITIAL PATH IN UPDATE FUNCTION")
            input()
        else:
            prev_path_id = max(self.cur_path_id - 1, 0) # 진행 전 -> idx 0, 진행 후 -> idx (self.cur_path_id - 1)
            prev_sat_id = self.satellite_path[prev_path_id][0]

        remain_vsg_path = self.get_remained_vsg_path(all_gserver_list, all_sat_list)
        if not remain_vsg_path:
            print(f"[ERROR] NO VSG LEFT. processed path: {self.satellite_path[:self.cur_path_id]}")
            return []

        updated_satellite_path = self.satellite_path[:self.cur_path_id]

        print(f"{self.id} BEFORE UPDATE VSG PATH: {self.vsg_path}")
        print(f"{self.id} BEFORE UPDATE SATELLITE PATH: {self.satellite_path}")
        print(f"{self.id} BEFORE current satellite idx: {self.cur_path_id}")

        for i in range(len(remain_vsg_path)):
            dst_vsg, dst_vnf = remain_vsg_path[i]

            is_vnf = has_vnf_tag(dst_vnf)
            if is_vnf:
                current_vnf_id = get_vnf_id_for_list(dst_vnf)
                candidate_dst_sat_ids = [
                    sat.id for sat in all_vsg_list[dst_vsg].satellites
                    if current_vnf_id in sat.vnf_list
                ]
            else:
                candidate_dst_sat_ids = [
                    sat.id for sat in all_vsg_list[dst_vsg].satellites
                ]

            if not candidate_dst_sat_ids:
                print(f"[ERROR] 3-1 No SATELLITE TO DST")
                self.is_dropped = True
                return []
            dst_sat_id = random.choice(candidate_dst_sat_ids)

            if prev_sat_id == dst_sat_id:
                updated_satellite_path.append([dst_sat_id, dst_vnf])
                prev_sat_id = dst_sat_id
            else:
                try:
                    sub_path = nx.shortest_path(G, source=prev_sat_id, target=dst_sat_id)

                    if len(sub_path) > 2:
                        for sid in sub_path[1:-1]:
                            updated_satellite_path.append([sid, None])
                    updated_satellite_path.append([dst_sat_id, dst_vnf])
                    prev_sat_id = dst_sat_id
                except nx.NetworkXNoPath:
                    print(f"[ERROR] 3-2 No TRAJECTORY SAT TO SAT")
                    self.is_dropped = True
                    return []

        self.satellite_path = updated_satellite_path


        print(f"{self.id} UPDATE UPDATE VSG PATH: {self.vsg_path}")
        print(f"{self.id} UPDATE UPDATE SATELLITE PATH: {self.satellite_path}")
        # print(f"\n========== UPDATE BASIC GSFC {self.id} ==========")
        # print(f"  -> src {self.src_vsg_id}")
        # print(f"  -> dst {self.dst_vsg_id}")
        # print(f"  -> vnf sequence {self.vnf_sequence}")
        # print(f"  -> flow rule {self.gsfc_flow_rule}")
        # print(f"  -> vsg path {self.vsg_path}")
        # print(f"  -> processed satellite path {self.processed_satellite_path}")
        # print(f"  -> NEW satellite path {self.satellite_path}")

    def set_sd_satellite_path(self, all_vsg_list, all_gserver_list, all_sat_list, G, vsg_G):
        # src vsg에 prev sat, src sat 존재
        # dst vsg에 dst sat, vnf sat 존재
        # src sat과 dst sat은 vsg 간 최단 egress ingress node

        len_vsg_path = len(self.vsg_path)
        if len_vsg_path < 1:
            print(f"[ERROR] No VSG path in GSFC {self.id}")
            return []

        if not self.satellite_path: # 경로 첫 설정
            cur_vsg_path_id = self.cur_vsg_path_id
            src_vsg, src_vnf = self.vsg_path[cur_vsg_path_id]

            is_vnf_src = has_vnf_tag(src_vnf)
            if is_vnf_src:
                current_vnf_id = get_vnf_id_for_list(src_vnf)
                candidate_prev_sat_ids = [
                    sat.id for sat in all_vsg_list[src_vsg].satellites
                    if current_vnf_id in sat.vnf_list
                ]
            else:
                candidate_prev_sat_ids = [
                    sat.id for sat in all_vsg_list[src_vsg].satellites
                ]

            if not candidate_prev_sat_ids:
                print(f"[ERROR] 3-1 No SATELLITE TO SRC")
                self.is_dropped = True
                return []

            # TODO. random choice?
            prev_sat_id = random.choice(candidate_prev_sat_ids)
            self.satellite_path.append([prev_sat_id, src_vnf])
            self.cur_vsg_path_id += 1
        else: # 경로 추가
            prev_sat_id = self.satellite_path[-1][0]

            # 위성 이동성으로 prev sat이 src vsg를 벗어났을 경우 커버 불가
            # prev_vsg_path_id = self.cur_vsg_path_id - 1
            # src_vsg, src_vnf = self.vsg_path[prev_vsg_path_id]

            prev_sat = all_sat_list[prev_sat_id]
            src_vsg = prev_sat.current_vsg_id

            dst_vsg_path_id = self.cur_vsg_path_id
            dst_vsg, dst_vnf = self.vsg_path[dst_vsg_path_id]

            is_vnf_dst = has_vnf_tag(dst_vnf)
            if is_vnf_dst:
                current_vnf_id = get_vnf_id_for_list(dst_vnf)
                candidate_vnf_sat_ids = [
                    sat.id for sat in all_vsg_list[dst_vsg].satellites
                    if current_vnf_id in sat.vnf_list
                ]
            else:
                candidate_vnf_sat_ids = [
                    sat.id for sat in all_vsg_list[dst_vsg].satellites
                ]

            if not candidate_vnf_sat_ids:
                print(f"[ERROR] 3-1 No SATELLITE TO DST")
                self.is_dropped = True
                return []

            if src_vsg != dst_vsg: # vsg 이동이 필요할 때만 진행
                src_vsg_sat_ids = [sat.id for sat in all_vsg_list[src_vsg].satellites]
                dst_vsg_sat_ids = [sat.id for sat in all_vsg_list[dst_vsg].satellites]
                src_sat_id, dst_sat_id, src_dst_distance_m = get_src_dst_sat(src_vsg, dst_vsg, src_vsg_sat_ids, dst_vsg_sat_ids, all_vsg_list)

                # prev_sat -> src_sat
                if prev_sat_id != src_sat_id:
                    try:
                        sub_path = nx.shortest_path(G, source=prev_sat_id, target=src_sat_id)

                        for sid in sub_path[1:]:
                            self.satellite_path.append([sid, None])
                    except nx.NetworkXNoPath:
                        print(f"[ERROR] 3-2 No TRAJECTORY SAT TO SAT")
                        self.is_dropped = True
                        return []

                # src_sat_id -> dst_sat
                if src_sat_id == dst_sat_id:  # 이동 X
                    # SRC랑 DST는 다른 VSG 내에 있음.
                    print(f"[WARN] src_sat_id is not in dst_vsg")
                    input("ㅈ댔다!! src sat과 dst sat이 같은 vsg에 있을 리 없어~~~~~")
                else:
                    try:
                        sub_path = nx.shortest_path(G, source=src_sat_id, target=dst_sat_id)

                        for sid in sub_path[1:]:
                            self.satellite_path.append([sid, None])
                    except nx.NetworkXNoPath:
                        print(f"[ERROR] 3-2 No TRAJECTORY SAT TO SAT")
                        self.is_dropped = True
                        return []
            else:
                dst_sat_id = prev_sat_id

            # dst_sat -> vnf_sat
            _, vnf_sat_id, dst_vnf_distance_m = get_src_dst_sat(dst_vsg, dst_vsg, [dst_sat_id], candidate_vnf_sat_ids, all_vsg_list)

            if dst_sat_id != vnf_sat_id:
                vnf_sat_avg_queue = mean([len(isl_k) for isl_k in all_sat_list[vnf_sat_id].queue_ISL])

                if vnf_sat_avg_queue > 200:
                    # gserver까지 graph에 추가
                    current_G = G
                    selected_gserver_id = dst_vsg.gserver
                    if selected_gserver_id is not None:
                        current_G = create_temp_gserver_graph(G, all_gserver_list, all_vsg_list, selected_gserver_id)
                        # gsfc에 처리 gserver 추가
                        selected_gserver = all_gserver_list[selected_gserver_id]
                        self.gserver = selected_gserver

                    # dst_sat -> vnf_g
                    try:
                        sub_path = nx.shortest_path(current_G, source=dst_sat_id, target=selected_gserver_id+NUM_SATELLITES)

                        if len(sub_path) > 2:
                            for sid in sub_path[1:-1]:
                                self.satellite_path.append([sid, None])
                        self.satellite_path.append([selected_gserver_id+NUM_SATELLITES, dst_vnf])
                    except nx.NetworkXNoPath:
                        print(f"[ERROR] 3-2 No TRAJECTORY SAT TO SAT")
                        self.is_dropped = True
                        return []

                    # vnf_g -> vnf_sat
                    try:
                        sub_path = nx.shortest_path(current_G, source=selected_gserver_id+NUM_SATELLITES, target=vnf_sat_id)

                        if len(sub_path) > 2:
                            for sid in sub_path[1:-1]:
                                self.satellite_path.append([sid, None])
                        if has_dst_tag(dst_vnf): # TODO. 이렇게 욱여막았지만.. 처리 필요
                            self.satellite_path.append([vnf_sat_id, dst_vnf])
                        else:
                            self.satellite_path.append([vnf_sat_id, None])
                    except nx.NetworkXNoPath:
                        print(f"[ERROR] 3-2 No TRAJECTORY SAT TO SAT")
                        self.is_dropped = True
                        return []
                else:
                    try:
                        sub_path = nx.shortest_path(G, source=dst_sat_id, target=vnf_sat_id)

                        if len(sub_path) > 2:
                            for sid in sub_path[1:-1]:
                                self.satellite_path.append([sid, None])
                        self.satellite_path.append([vnf_sat_id, dst_vnf])
                    except nx.NetworkXNoPath:
                        print(f"[ERROR] 3-2 No TRAJECTORY SAT TO SAT")
                        self.is_dropped = True
                        return []
            else:
                self.satellite_path.append([vnf_sat_id, dst_vnf])

            self.cur_vsg_path_id += 1

        print(f"\n========== SD GSFC {self.id} ==========")
        print(f"  -> src {self.src_vsg_id}")
        print(f"  -> dst {self.dst_vsg_id}")
        print(f"  -> vnf sequence {self.vnf_sequence}")
        print(f"  -> flow rule {self.gsfc_flow_rule}")
        print(f"  -> vsg path {self.vsg_path}")
        print(f"  -> satellite path {self.satellite_path}")

    def set_dd_satellite_path(self, all_vsg_list, all_sat_list, G):
        best_full_vsg_path = None
        best_full_path = None
        min_total_hops = float('inf')

        src_vsg_id = self.src_vsg_id
        src_vsg_sat_ids = [sat.id for sat in all_vsg_list[src_vsg_id].satellites]

        dst_vsg_id = self.dst_vsg_id
        dst_vsg_sat_ids = [sat.id for sat in all_vsg_list[dst_vsg_id].satellites]

        # vnf 별 가능한 위성 후보군 추출
        unique_vnf_types = sorted(list(set(self.vnf_sequence)))
        vnf_to_sat_ids = {}
        for vnf in unique_vnf_types:
            candidate_sat_ids = [
                sat.id for sat in all_sat_list
                if vnf in sat.vnf_list
            ]
            if not candidate_sat_ids:
                print(f"[ERROR] No satellites available for VNF {vnf}")
            vnf_to_sat_ids[vnf] = candidate_sat_ids

        # 모든 VNF 조합
        list_of_candidate_lists = [vnf_to_sat_ids[vnf_type] for vnf_type in self.vnf_sequence]
        vnf_combinations = list(product(*list_of_candidate_lists))

        for vnf_combo in vnf_combinations:
            path = list(vnf_combo)
            total_hops = 0
            valid = True
            full_vsg_path = [] # vsg path 저장
            full_path = []  # 전체 경로 누적

            # 시작점 처리
            if not src_vsg_sat_ids or not dst_vsg_sat_ids:
                print("[ERROR] There is no available satellites in SRC or DST VSG")
                break

            src_sat_id = random.choice(src_vsg_sat_ids)
            path.insert(0, src_sat_id)
            dst_sat_id = random.choice(dst_vsg_sat_ids)
            path.append(dst_sat_id)

            # 전체 경로 유효성 및 홉 수 계산
            current_vnf_id = 0

            for i in range(len(path)):
                if (i == 0):
                    full_path.append([path[i], ("src")])
                    full_vsg_path.append((self.src_vsg_id, "src"))
                elif (i == (len(path)-1)):
                    if path[-2] != path[-1]:
                        segment = get_shortest_path(G, path[-2], path[-1])
                        if not segment:
                            valid = False
                            break

                        total_hops += len(segment) - 1
                        if full_path:
                            if len(segment) > 2:
                                for seg in segment[1:-1]:
                                    full_path.append([seg, (None)])
                        else:
                            vnf_sat_id = segment[0]
                            full_path.append([vnf_sat_id, (f"vnf{self.vnf_sequence[current_vnf_id]}")])

                            vnf_sat = all_sat_list[vnf_sat_id]
                            full_vsg_path.append((vnf_sat.current_vsg_id, f"vnf{self.vnf_sequence[current_vnf_id]}"))

                            current_vnf_id += 1
                            if len(segment) > 2:
                                for seg in segment[1:-1]:
                                    full_path.append([seg, (None)])
                    full_path.append([path[-1], ("dst")])
                    full_vsg_path.append((self.dst_vsg_id, "src"))
                else:
                    if path[i - 1] == path[i]:
                        vnf_sat_id = path[i]
                        full_path.append([path[i], (f"vnf{self.vnf_sequence[current_vnf_id]}")])

                        vnf_sat = all_sat_list[vnf_sat_id]
                        full_vsg_path.append((vnf_sat.current_vsg_id, f"vnf{self.vnf_sequence[current_vnf_id]}"))

                        current_vnf_id += 1
                    else:
                        segment = get_shortest_path(G, path[i - 1], path[i])
                        if not segment:
                            valid = False
                            break

                        total_hops += len(segment) - 1
                        if full_path:
                            if len(segment) > 2:
                                for seg in segment[1:-1]:
                                    full_path.append([seg, (None)])
                            vnf_sat_id = segment[-1]
                            full_path.append([segment[-1], (f"vnf{self.vnf_sequence[current_vnf_id]}")])

                            vnf_sat = all_sat_list[vnf_sat_id]
                            full_vsg_path.append((vnf_sat.current_vsg_id, f"vnf{self.vnf_sequence[current_vnf_id]}"))

                            current_vnf_id += 1
                        else:
                            vnf_sat_id = segment[0]
                            full_path.append([segment[0], (f"vnf{self.vnf_sequence[current_vnf_id]}")])

                            vnf_sat = all_sat_list[vnf_sat_id]
                            full_vsg_path.append((vnf_sat.current_vsg_id, f"vnf{self.vnf_sequence[current_vnf_id]}"))

                            current_vnf_id += 1
                            if len(segment) > 2:
                                for seg in segment[1:-1]:
                                    full_path.append([seg, (None)])

                            vnf_sat_id = segment[-1]
                            full_path.append([segment[-1], (f"vnf{self.vnf_sequence[current_vnf_id]}")])

                            vnf_sat = all_sat_list[vnf_sat_id]
                            full_vsg_path.append((vnf_sat.current_vsg_id, f"vnf{self.vnf_sequence[current_vnf_id]}"))

                            current_vnf_id += 1

            if valid and total_hops < min_total_hops:
                # print(f"NEWNEW {total_hops}, vnf_combo: {vnf_combo}")
                min_total_hops = total_hops
                best_full_path = full_path
                best_full_vsg_path = full_vsg_path

        self.vsg_path = best_full_vsg_path
        self.satellite_path = best_full_path

        print(f"\n========== INITIAL DD GSFC {self.id} ==========")
        print(f"  -> src {self.src_vsg_id}")
        print(f"  -> dst {self.dst_vsg_id}")
        print(f"  -> vnf sequence {self.vnf_sequence}")
        print(f"  -> vsg path {self.vsg_path}")
        print(f"  -> satellite path {self.satellite_path}")

    def update_dd_satellite_path(self, all_vsg_list, all_sat_list, all_gserver_list, G):
        best_full_vsg_path = None
        best_full_path = None
        min_total_hops = float('inf')

        remain_vnf_sequence = self.get_remain_vnf_sequence(all_sat_list, all_gserver_list)
        if not remain_vnf_sequence:
            print(f"[ERROR] NO VSG LEFT. processed path: {self.satellite_path[:self.cur_path_id]}")
            return []

        if len(self.satellite_path) < 1:
            # self.basic_satellite_path(all_vsg_list, G)
            print(f"[WARN] INITIAL PATH IN UPDATE FUNCTION")
            input()
        else:
            prev_path_id = max(self.cur_path_id - 1, 0) # 진행 전 -> idx 0, 진행 후 -> idx (self.cur_path_id - 1)
            src_vsg_sat_ids = [self.satellite_path[prev_path_id][0]]

        dst_vsg_id = self.dst_vsg_id
        dst_vsg_sat_ids = [sat.id for sat in all_vsg_list[dst_vsg_id].satellites]

        # vnf 별 가능한 위성 후보군 추출
        unique_vnf_types = sorted(list(set(remain_vnf_sequence)))
        vnf_to_sat_ids = {}
        for vnf in unique_vnf_types:
            candidate_sats = [
                sat.id for sat in all_sat_list
                if vnf in sat.vnf_list]
            if not candidate_sats:
                print(f"[ERROR] No satellites available for VNF {vnf}")
            vnf_to_sat_ids[vnf] = candidate_sats

        # 모든 VNF 조합
        list_of_candidate_lists = [vnf_to_sat_ids[vnf_type] for vnf_type in remain_vnf_sequence]
        vnf_combinations = list(product(*list_of_candidate_lists))
        # print(f"combinations in dd: {vnf_combinations}")

        for vnf_combo in vnf_combinations:
            path = list(vnf_combo)
            total_hops = 0
            valid = True
            full_vsg_path = self.get_processed_vsg_path(all_gserver_list, all_sat_list)
            full_path = self.satellite_path[:self.cur_path_id]

            # 시작점 처리
            if not dst_vsg_sat_ids:
                print("[ERROR] There is no available satellites in DST VSG")
                break

            src_sat_id = self.satellite_path[prev_path_id][0]
            path.insert(0, src_sat_id)
            dst_sat_id = random.choice(dst_vsg_sat_ids)
            path.append(dst_sat_id)

            # 전체 경로 유효성 및 홉 수 계산
            current_vnf_id = 0

            for i in range(len(path)):
                if (i == (len(path) - 1)):
                    if path[-2] != path[-1]:
                        segment = get_shortest_path(G, path[-2], path[-1])
                        if not segment:
                            valid = False
                            break

                        total_hops += len(segment) - 1
                        if full_path:
                            if len(segment) > 2:
                                for seg in segment[1:-1]:
                                    full_path.append([seg, (None)])
                        else:
                            vnf_sat_id = segment[0]
                            full_path.append([vnf_sat_id, (f"vnf{self.vnf_sequence[current_vnf_id]}")])

                            vnf_sat = all_sat_list[vnf_sat_id]
                            full_vsg_path.append((vnf_sat.current_vsg_id, f"vnf{self.vnf_sequence[current_vnf_id]}"))

                            current_vnf_id += 1
                            if len(segment) > 2:
                                for seg in segment[1:-1]:
                                    full_path.append([seg, (None)])
                    full_path.append([path[-1], ("dst")])
                    full_vsg_path.append((self.dst_vsg_id, "src"))
                else:
                    if path[i - 1] == path[i]:
                        vnf_sat_id = path[i]
                        full_path.append([path[i], (f"vnf{self.vnf_sequence[current_vnf_id]}")])

                        vnf_sat = all_sat_list[vnf_sat_id]
                        full_vsg_path.append((vnf_sat.current_vsg_id, f"vnf{self.vnf_sequence[current_vnf_id]}"))

                        current_vnf_id += 1
                    else:
                        segment = get_shortest_path(G, path[i - 1], path[i])
                        if not segment:
                            valid = False
                            break

                        total_hops += len(segment) - 1
                        if full_path:
                            if len(segment) > 2:
                                for seg in segment[1:-1]:
                                    full_path.append([seg, (None)])
                            vnf_sat_id = segment[-1]
                            full_path.append([segment[-1], (f"vnf{self.vnf_sequence[current_vnf_id]}")])

                            vnf_sat = all_sat_list[vnf_sat_id]
                            full_vsg_path.append((vnf_sat.current_vsg_id, f"vnf{self.vnf_sequence[current_vnf_id]}"))

                            current_vnf_id += 1
                        else:
                            vnf_sat_id = segment[0]
                            full_path.append([segment[0], (f"vnf{self.vnf_sequence[current_vnf_id]}")])

                            vnf_sat = all_sat_list[vnf_sat_id]
                            full_vsg_path.append((vnf_sat.current_vsg_id, f"vnf{self.vnf_sequence[current_vnf_id]}"))

                            current_vnf_id += 1
                            if len(segment) > 2:
                                for seg in segment[1:-1]:
                                    full_path.append([seg, (None)])

                            vnf_sat_id = segment[-1]
                            full_path.append([segment[-1], (f"vnf{self.vnf_sequence[current_vnf_id]}")])

                            vnf_sat = all_sat_list[vnf_sat_id]
                            full_vsg_path.append((vnf_sat.current_vsg_id, f"vnf{self.vnf_sequence[current_vnf_id]}"))

                            current_vnf_id += 1
            if valid and total_hops < min_total_hops:
                # print(f"NEWNEW {total_hops}, vnf_combo: {vnf_combo}")
                min_total_hops = total_hops
                best_full_path = full_path
                best_full_vsg_path = full_vsg_path

        self.vsg_path = best_full_vsg_path
        self.satellite_path = best_full_path

    def get_remain_vnf_sequence(self, all_sat_list, all_gserver_list):
        full_vnf_sequence = self.vnf_sequence

        processed_vnf_sequence = []

        for node_id, vnf_info in self.satellite_path[:self.cur_path_id]:
            if node_id < NUM_SATELLITES:
                node = all_sat_list[node_id]
                vsg_id = node.current_vsg_id
            else:
                node_id -= NUM_SATELLITES
                node = all_gserver_list[node_id]
                vsg_id = node.vsg_id

            if vsg_id is None:
                print("[Warn] Satellite does not belong to any VSG.")
                continue

            vnf_tag = get_vnf_id_for_list(vnf_info)

            if vnf_tag:
                processed_vnf_sequence.append(vnf_tag)

        remain_start_id = len(processed_vnf_sequence)
        remain_vnf_sequence = full_vnf_sequence[remain_start_id:]
        return remain_vnf_sequence

    def get_processed_vsg_path(self, all_gserver_list, all_sat_list):
        full_vsg_path = self.vsg_path
        processed_satellite_path = self.satellite_path[:self.cur_path_id]

        processed_vsg_path = []
        processed_vsg_ids = 0

        for node_id, vnf_info in processed_satellite_path:
            if not (0 <= node_id < NUM_SATELLITES):
                node_id -= NUM_SATELLITES
                node = all_gserver_list[node_id]
                vsg_id = node.vsg_id
            else:
                node = all_sat_list[node_id]
                vsg_id = node.current_vsg_id

            if vsg_id is None:
                print("[Warn] Satellite does not belong to any VSG.")
                continue

            expected_vsg_id, expected_vnf_tag = full_vsg_path[processed_vsg_ids]

            if vnf_info == expected_vnf_tag:
                processed_vsg_path.append([expected_vsg_id, expected_vnf_tag])
                processed_vsg_ids += 1

        return processed_vsg_path

    def get_remained_vsg_path(self, all_gserver_list, all_sat_list):
        full_vsg_path = self.vsg_path
        processed_vsg_path = self.get_processed_vsg_path(all_gserver_list, all_sat_list)
        processed_vsg_ids = len(processed_vsg_path)

        if not processed_vsg_path:
            return full_vsg_path

        remain_vsg_path = full_vsg_path[processed_vsg_ids:]
        return remain_vsg_path

    def check_need_detour(self, lost_vnf_type_list, all_gserver_list, all_sat_list):
        is_affected = False
        if not self.vsg_path:
            print(f"[WANR] GSFC {self.id} has no vsg path")
            return []

        # uses_changed_vnf = False
        # for vsg_id, vnf_tag in self.vsg_path:
        #     if vsg_id not in lost_vnf_type_list:
        #         continue
        #     if not has_vnf_tag(vnf_tag):
        #         continue
        #     vnf_type = get_vnf_id_for_list(vnf_tag)
        #     if vnf_type in lost_vnf_type_list[vsg_id]:
        #         uses_changed_vnf = True
        #         break

        # if uses_changed_vnf:
        remain_vsg_path = self.get_remained_vsg_path(all_gserver_list, all_sat_list)

        for vsg_id, vnf_tag in remain_vsg_path:
            if not lost_vnf_type_list[vsg_id]:
                continue
            if not has_vnf_tag(vnf_tag):
                continue
            vnf_type = get_vnf_id_for_list(vnf_tag)

            if vnf_type in lost_vnf_type_list[vsg_id][0]:
                is_affected = True
                break

        return is_affected

    def detour_satellite_path(self, lost_vnf_type_list, all_vsg_list, all_gserver_list, all_sat_list, G, mode):
        if self.is_succeed or self.is_dropped:
            return []
        is_affected = self.check_need_detour(lost_vnf_type_list, all_gserver_list, all_sat_list)

        if is_affected:
            print(f"====== DETOUR GSFC {self.id} ======")

            if mode == 'dd': # dynamic 버전이랑 아닌 거 구분해서 저장
                self.update_dd_satellite_path(all_vsg_list, all_sat_list, all_gserver_list, G)
            elif mode == 'basic':
                print(f"  -> BEFORE satellite path {self.satellite_path}, cur_path_id {self.cur_path_id} processed satellite path {self.satellite_path[:self.cur_path_id]}")
                self.update_basic_satellite_path(all_vsg_list, all_gserver_list, all_sat_list, G)
                print(f"  -> AFTER satellite path {self.satellite_path}, cur_path_id {self.cur_path_id} processed satellite path {self.satellite_path[:self.cur_path_id]}")
            elif mode == 'sd':
                pass
            else:
                print("\n")
                print(f"[ERROR] Unknown mode {mode}")
                print("\n")
                return []

        # print(f"gsfc {self.id} is_affected: {is_affected} satellite path: {self.satellite_path}")
        # print(f"current_vsg:{self.vsg_path} remain_vsg_path: {remain_vsg_path}")

    def DH_set_state_proc(self):
        self.state = 1

    def DH_eval_proc_delay(self, node=None, node_type=None, data_rate_pair=None):
        if data_rate_pair is None:
            sat_processing_rate = SAT_PROCESSING_RATE
            gserver_processing_rate = GSERVER_PROCESSING_RATE
        else:
            sat_processing_rate = data_rate_pair[0]
            gserver_processing_rate = data_rate_pair[1]

        if node_type == 'satellite':
            num_computing_gsfc = max(len(node.process_queue), 1)
            current_timeslot_capa = sat_processing_rate / num_computing_gsfc
            self.current_remaining_vnf_process -= current_timeslot_capa
        elif node_type == 'gserver':
            num_computing_gsfc = max(len(node.process_queue), 1)
            current_timeslot_capa = gserver_processing_rate / num_computing_gsfc
            self.current_remaining_vnf_process -= current_timeslot_capa
        estimated_remain_proc_delay = math.ceil(self.current_remaining_vnf_process / current_timeslot_capa)
        # 이 gsfc가 이번 time slot에서 끝나면 0, 아니면 estimated_remain_proc_delay 반환
        if self.current_remaining_vnf_process > 0:
            self.estimated_remain_proc_delay = estimated_remain_proc_delay
            return estimated_remain_proc_delay
            # return 1
        else:
            self.estimated_remain_proc_delay = 0
            return 0
            # return 1

    def DH_accumulate_proc_delay(self, node=None, node_type=None, data_rate_pair=None):
        estimated_remain_proc_delay = self.DH_eval_proc_delay(node, node_type, data_rate_pair)
        self.proc_delay_ms += 1
        self.DH_set_state_proc()
        return estimated_remain_proc_delay

    def DH_set_state_queue(self, queue_delay=0):
        self.state = 2
        self.DH_remaining_ongoing_time_slot = queue_delay  # 여기에서는 FIFO 구조를 따르므로, 다른 패킷들이 다 나갈 때까지 그냥 기다리는 상황이므로, DH_remaining_ongoing_time_slot으로 조정할 필요가 없음

    def DH_eval_queue_delay(self, node=None, node_type=None, ISL_next_hop=None):
        queue_delay = 1
        if node_type == 'satellite':
            # TSL, 정확히는 STL
            if ISL_next_hop == 4:
                queue = node.queue_TSL
            else:
                queue = node.queue_ISL[ISL_next_hop]
            tmp_sum = 0  # 누적합
            for i in range(len(queue)):
                size = queue[i][1]
                tmp_sum += size
                while tmp_sum >= SAT_LINK_CAPACITY:
                    queue_delay += 1
                    tmp_sum -= SAT_LINK_CAPACITY
        elif node_type == 'gserver':
            queue = node.queue_TSL
            tmp_sum = 0  # 누적합
            for i in range(len(queue)):
                size = queue[i][1]
                tmp_sum += size
                while tmp_sum >= GSERVER_LINK_CAPACITY:
                    queue_delay += 1
                    tmp_sum -= GSERVER_LINK_CAPACITY
        return queue_delay

    def DH_set_queue_delay(self, node=None, node_type=None, ISL_next_hop=None):
        queue_delay = self.DH_eval_queue_delay(node, node_type, ISL_next_hop)
        self.queue_delay_ms += queue_delay
        self.DH_set_state_queue(queue_delay=queue_delay)

    def DH_set_state_trans(self, trans_delay=0):
        self.state = 3
        self.trans_delay_ms += trans_delay
        self.DH_remaining_ongoing_time_slot = trans_delay

    def DH_eval_trans_delay(self, node_type=None):
        if node_type == 'satellite':
            trans_delay = math.ceil(self.sfc_size / (SAT_LINK_CAPACITY))
        elif node_type == 'gserver':
            trans_delay = math.ceil(self.sfc_size / GSERVER_LINK_CAPACITY)
        return trans_delay

    def DH_set_trans_delay(self, node_type=None):
        trans_delay = self.DH_eval_trans_delay(node_type)
        self.DH_set_state_trans(trans_delay=trans_delay)

    def DH_set_state_prop(self, prop_delay=0):
        self.state = 4
        self.DH_remaining_ongoing_time_slot = prop_delay

    def DH_eval_prop_delay(self, propagation_delay=0):
        return propagation_delay

    def DH_set_prop_delay(self, propagation_delay=0):
        prop_delay = self.DH_eval_prop_delay(propagation_delay=propagation_delay)
        self.prop_delay_ms += prop_delay
        self.DH_set_state_prop(prop_delay=prop_delay)

    def check_gsfc_done(self, t, all_vsg_list, all_gserver_list, all_sat_list, G, vsg_G):
        if len(self.satellite_path) == self.cur_path_id:  # 생성된 경로 처리 완
            last_processed_vnf = self.satellite_path[self.cur_path_id - 1][1]
            was_dst = has_dst_tag(last_processed_vnf)

            if was_dst:  # dst까지 처리 완. 진짜진짜 끝
                self.is_succeed = True
                write_gsfc_csv_log(self.gsfc_log_path, t, self, "DONE")
            else:  # 처리 다 했는데? dst 안 거침????? 이새키 뭐야
                if self.mode == 'sd':  # sd 원래 다음 vsg까지의 경로만을 만듦 => 다음 vsg까지의 경로 생성
                    self.set_sd_satellite_path(all_vsg_list, all_gserver_list, all_sat_list, G, vsg_G)
                    self.cur_path_id -= 1
                    self.state = 1
                    self.is_appended = True
                else:  # 얘네는 안됨!!!!!! 다 처리했어야함!!!!
                    self.is_dropped = True
                    write_gsfc_csv_log(self.gsfc_log_path, t, self, "DROP")
                    input("ㅈ댔다!! 경로가 끝났을 리 없어~~~~~")

        is_done = True if (self.is_succeed or self.is_dropped) else False
        return is_done

    def time_tic(self, t, data_rate_pair, all_vsg_list, all_gserver_list, all_sat_list, G, vsg_G):
        # GSFC 처리 로직
        if self.is_succeed:
            return 0
        else:
            is_done = self.check_gsfc_done(t, all_vsg_list, all_gserver_list, all_sat_list, G, vsg_G)
            if is_done:
                return 0

            if self.state == 1:  # processing
                cur_node_id, cur_vnf_tag = self.satellite_path[self.cur_path_id]
                cur_node_type, cur_node = get_node_type(cur_node_id, all_gserver_list, all_sat_list)

                if (not has_vnf_tag(cur_vnf_tag)) or self.is_appended: # vnf 처리 필요 여부 파악
                    estimated_remain_proc_delay = 0
                    self.is_appended = False
                else:
                    estimated_remain_proc_delay = self.DH_accumulate_proc_delay(cur_node, cur_node_type, data_rate_pair)
                write_gsfc_csv_log(self.gsfc_log_path, t, self, "PROCESS")

                if estimated_remain_proc_delay <= 1: # process 종료
                    self.cur_path_id += 1
                    if len(self.satellite_path) == self.cur_path_id:
                        return 0

                    self.node_id_to_move = self.satellite_path[self.cur_path_id][0]
                    if cur_node_id != self.node_id_to_move:
                        ISL_next_hop = cur_node.get_next_hop_link_idx(self)
                        if ISL_next_hop == None:
                            input(f"{cur_node_id}, {self.node_id_to_move}")
                        self.DH_set_queue_delay(cur_node, cur_node_type, ISL_next_hop)
                        cur_node.queue_ISL[ISL_next_hop].append((self.id, self.sfc_size))

            elif self.state == 2: # queue
                prev_node_id, prev_vnf_tag = self.satellite_path[self.cur_path_id -1]
                prev_node_type, prev_node = get_node_type(prev_node_id, all_gserver_list, all_sat_list)
                next_node_id, next_vnf_tag = self.satellite_path[self.cur_path_id]
                next_node_type, next_node = get_node_type(next_node_id, all_gserver_list, all_sat_list)

                if next_node_id != self.node_id_to_move: # 경로 재설정됨
                    self.queue_delay_ms -= self.DH_remaining_ongoing_time_slot
                    self.DH_remaining_ongoing_time_slot = 0
                    # queue 제거
                    self.node_id_to_move = next_node_id

                    if prev_node_id == self.node_id_to_move: # 이동 없음
                        self.DH_accumulate_proc_delay(next_node, next_node_type, data_rate_pair)
                        write_gsfc_csv_log(self.gsfc_log_path, t, self, "DETOUR")
                        return 0
                    else:
                        ISL_next_hop = prev_node.get_next_hop_link_idx(self)
                        if ISL_next_hop == None:
                            input(f"{prev_node_id}, {self.node_id_to_move}")
                        self.DH_set_queue_delay(prev_node, prev_node_type, ISL_next_hop)
                        prev_node.queue_ISL[ISL_next_hop].append((self.id, self.sfc_size))
                        write_gsfc_csv_log(self.gsfc_log_path, t, self, "DETOUR")
                        return 0

                write_gsfc_csv_log(self.gsfc_log_path, t, self, "QUEUE")

                if self.DH_remaining_ongoing_time_slot <= 1: # 이번 time slot에 queue 종료
                    # queue 제거?
                    self.DH_set_trans_delay(prev_node_type)
                self.DH_remaining_ongoing_time_slot -= 1

            elif self.state == 3: # transmitting
                prev_node_id, prev_vnf_tag = self.satellite_path[self.cur_path_id - 1]
                prev_node_type, prev_node = get_node_type(prev_node_id, all_gserver_list, all_sat_list)
                next_node_id, next_vnf_tag = self.satellite_path[self.cur_path_id]
                next_node_type, next_node = get_node_type(next_node_id, all_gserver_list, all_sat_list)

                if next_node_id != self.node_id_to_move: # 경로 재설정됨
                    self.trans_delay_ms -= self.DH_remaining_ongoing_time_slot
                    self.DH_remaining_ongoing_time_slot = 0
                    self.node_id_to_move = next_node_id

                    if prev_node_id == self.node_id_to_move: # 이동 없음
                        self.DH_accumulate_proc_delay(next_node, next_node_type, data_rate_pair)
                        write_gsfc_csv_log(self.gsfc_log_path, t, self, "DETOUR")
                        return 0
                    else:
                        ISL_next_hop = prev_node.get_next_hop_link_idx(self)
                        if ISL_next_hop == None:
                            input(f"{prev_node_id}, {self.node_id_to_move}")
                        self.DH_set_queue_delay(prev_node, prev_node_type, ISL_next_hop)
                        prev_node.queue_ISL[ISL_next_hop].append((self.id, self.sfc_size))
                        write_gsfc_csv_log(self.gsfc_log_path, t, self, "DETOUR")
                        return 0

                write_gsfc_csv_log(self.gsfc_log_path, t, self, "TRANSMIT")

                if self.DH_remaining_ongoing_time_slot <= 1:  # 이번 time slot에 transmit 종료
                    ISL_next_hop = prev_node.get_next_hop_link_idx(self)
                    if ISL_next_hop == None:
                        print(f"{self.id}")
                    propagation_delay = math.ceil(prev_node.adj_sat_p_d_list[ISL_next_hop])
                    self.DH_set_prop_delay(propagation_delay)
                self.DH_remaining_ongoing_time_slot -= 1

            elif self.state == 4: # propagation
                prev_node_id, prev_vnf_tag = self.satellite_path[self.cur_path_id - 1]
                prev_node_type, prev_node = get_node_type(prev_node_id, all_gserver_list, all_sat_list)
                next_node_id, next_vnf_tag = self.satellite_path[self.cur_path_id]
                next_node_type, next_node = get_node_type(next_node_id, all_gserver_list, all_sat_list)

                if next_node_id != self.node_id_to_move:  # 경로 재설정됨
                    self.prop_delay_ms -= self.DH_remaining_ongoing_time_slot
                    self.DH_remaining_ongoing_time_slot = 0
                    self.node_id_to_move = next_node_id

                    if prev_node_id == self.node_id_to_move:  # 이동 없음
                        self.DH_accumulate_proc_delay(next_node, next_node_type, data_rate_pair)
                        write_gsfc_csv_log(self.gsfc_log_path, t, self, "DETOUR")
                        return 0
                    else:
                        ISL_next_hop = prev_node.get_next_hop_link_idx(self)
                        if ISL_next_hop == None:
                            input(f"{prev_node_id}, {self.node_id_to_move}")
                        self.DH_set_queue_delay(prev_node, prev_node_type, ISL_next_hop)
                        prev_node.queue_ISL[ISL_next_hop].append((self.id, self.sfc_size))
                        write_gsfc_csv_log(self.gsfc_log_path, t, self, "DETOUR")
                        return 0

                write_gsfc_csv_log(self.gsfc_log_path, t, self, "PROPAGATE")

                if self.DH_remaining_ongoing_time_slot <= 1: # 이번 time slot에 propagation 종료
                    self.DH_accumulate_proc_delay(next_node, next_node_type, data_rate_pair)
                self.DH_remaining_ongoing_time_slot -= 1
            else:
                input("ㅈ댔다!! state가 4보다 클 순 없어~~~~~")