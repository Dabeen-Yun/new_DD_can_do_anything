from Params import *
import numpy as np
import networkx as nx

import csv
import os
import json

# 해당 vnf tag가 'vnf#'인지 확인
def has_vnf_tag(x):
    if x is None:
        return False
    if isinstance(x, (list, tuple)):
        return any(isinstance(e, str) and 'vnf' in e.lower() for e in x)
    if isinstance(x, str):
        return 'vnf' in x.lower()
    return False

# 해당 vnf tag가 'dst'인지 확인
def has_dst_tag(x):
    if x is None:
        return False
    if isinstance(x, (list, tuple)):
        return any(isinstance(e, str) and 'dst' in e.lower() for e in x)
    if isinstance(x, str):
        return 'dst' in x.lower()
    return False

# vnf tag가 'vnf#'라면, #만 추출
def get_vnf_id_for_list(vnf_tag):
    """
    경로 태그(예: 'vnf1', ('src', 'vnf1'))에서 VNF 번호(예: '1')를 추출합니다.

    :param vnf_tag: VNF 정보가 담긴 문자열 또는 튜플.
    :return: VNF 번호('1', '2' 등)를 담은 문자열, 또는 False (VNF가 없는 경우).
    """
    # 1. vnf_tag가 튜플일 경우 (예: ('src', 'vnf1'))
    if isinstance(vnf_tag, tuple):
        for item in vnf_tag:
            if isinstance(item, str) and item.startswith('vnf'):
                # 'vnf1'에서 '1'을 추출하여 반환
                return item[3:]
        return False

    # 2. vnf_tag가 단일 문자열일 경우 (예: 'vnf1' 또는 'src')
    elif isinstance(vnf_tag, str) and vnf_tag.startswith('vnf'):
        return vnf_tag[3:]
    return False

def to_ecef_m(lat_deg, lon_deg, alt_m=ORBIT_ALTITUDE):
    """(deg, deg, m) -> ECEF (x,y,z) in meters (구형 지구 근사)"""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = R_EARTH_RADIUS + (alt_m if alt_m is not None else 0.0)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z

def filter_sats_with_xyz_m(vsg_sats, candidate_ids):
    """
    VSG 위성 리스트에서 후보 ID만 추출하고 ECEF(m) 좌표까지 준비.
    sat.alt가 km라면 alt_m = s.alt * 1000.0 로 바꿔주세요.
    """
    cid = set(candidate_ids or [])
    rows = []

    for s in vsg_sats:
        if s.id in cid:
            alt_m = getattr(s, "alt", 0.0)  # meters 가정
            x, y, z = to_ecef_m(float(s.lat), float(s.lon), float(alt_m))
            rows.append((int(s.id), x, y, z))
    return rows

def best_pair_euclid_broadcast_m(src_arr, dst_arr):
    """
    src_arr: (n,4)[id,x,y,z] in meters, dst_arr: (m,4)
    브로드캐스팅으로 제곱거리 행렬 계산 후 최소 쌍.
    """
    sx = src_arr[:, 1][:, None]
    sy = src_arr[:, 2][:, None]
    sz = src_arr[:, 3][:, None]
    dx = dst_arr[:, 1][None, :]
    dy = dst_arr[:, 2][None, :]
    dz = dst_arr[:, 3][None, :]

    D2 = (sx - dx) ** 2 + (sy - dy) ** 2 + (sz - dz) ** 2
    k = int(np.argmin(D2))
    i, j = divmod(k, D2.shape[1])
    return int(src_arr[i, 0]), int(dst_arr[j, 0]), float(np.sqrt(D2[i, j]))  # 거리(m)

def best_pair_euclid_ckdtree_m(src_arr, dst_arr):
    """
    큰 스케일에서는 KD-트리로 최근접 탐색 (meters).
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(dst_arr[:, 1:4])  # xyz (meters)
    dists, idxs = tree.query(src_arr[:, 1:4], k=1)
    k = int(np.argmin(dists))
    return int(src_arr[k, 0]), int(dst_arr[int(idxs[k]), 0]), float(dists[k])  # 거리(m)

# src sat 후보군과 dst sat 후보군 중 서로의 거리가 가장 가까운 조합 도출
def get_src_dst_sat(src_vsg, dst_vsg, candidate_src_sats, candidate_dst_sats, all_vsg_list,
                    brute_threshold_pairs=200_000, prefer_ckdtree=True):
    """
    src_vsg/dst_vsg: VSG 인덱스
    candidate_*_sats: 고려할 위성 id 모음
    반환: (best_src_id, best_dst_id) 또는 return_distance=True면 (best_src_id, best_dst_id, best_dist_m)
    전부 미터(m) 기준.
    """
    src_rows = filter_sats_with_xyz_m(all_vsg_list[src_vsg].satellites, candidate_src_sats)
    dst_rows = filter_sats_with_xyz_m(all_vsg_list[dst_vsg].satellites, candidate_dst_sats)


    src_arr = np.array(src_rows, dtype=float)
    if src_arr.size == 0:
        print(src_rows)
    dst_arr = np.array(dst_rows, dtype=float)

    n, m = len(src_arr), len(dst_arr)
    pairs = n * m

    if pairs <= brute_threshold_pairs:
        sid, did, dist_m = best_pair_euclid_broadcast_m(src_arr, dst_arr)
    else:
        if prefer_ckdtree:
            try:
                sid, did, dist_m = best_pair_euclid_ckdtree_m(src_arr, dst_arr)
            except Exception:
                sid, did, dist_m = best_pair_euclid_broadcast_m(src_arr, dst_arr)
        else:
            sid, did, dist_m = best_pair_euclid_broadcast_m(src_arr, dst_arr)

    return (sid, did, dist_m)

# satellite 만으로 구성된 graph에 원하는 gserver 추가 (gserver 간 경로 생성을 막기 위해 해당 gserver만을 추가_
def create_temp_gserver_graph(G, all_gserver_list, all_vsg_list, gserver_id):
    TG_temp = G.copy()

    gserver = all_gserver_list[gserver_id]
    gserver_node_id = NUM_SATELLITES + gserver_id
    TG_temp.add_node(gserver_node_id, type="gserver", vsg_id=gserver.vsg_id)

    target_vsg = next((vsg for vsg in all_vsg_list if vsg.id == gserver.vsg_id), None)
    if target_vsg:
        for sat in target_vsg.satellites:
            tsl_delay_ms = sat.calculate_TSL_propagation_delay(gserver)
            # TSL 엣지 추가 (weight=delay)
            TG_temp.add_edge(sat.id, gserver_node_id, weight=tsl_delay_ms, link_type='tsl')

    return TG_temp

# src와 dst 간 최단 경로 생성
def get_shortest_path(G, src_id, dst_id, graph=None):
    if graph is None:
        graph = G

    try:
        return nx.shortest_path(graph, src_id, dst_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

# 해당 위성의 process queue에 쌓여있는 VNF 종류별 패킷 사이즈(로드)를 계산
def get_satellite_load(sat, all_gsfc_list):
    """
    :param sat: Satellite 객체
    :return: VNF 종류별 로드 딕셔너리(dict)
    """
    # VNF 종류별 로드를 저장할 딕셔너리: {'vnf1': load, 'vnf2': load, ...}
    vnf_load = {vnf_kind: 0 for vnf_kind in sat.vnf_list}

    # 큐의 각 항목은 [gsfc_id, vnf_idx, vnf_size] 형식
    for item in sat.process_queue:
        if len(item) < 3: continue

        gsfc_id = item[0]
        vnf_idx = item[1]
        vnf_size = item[2]

        try:
            # 1. GSFC 객체와 VNF Sequence를 사용하여 VNF 종류 확인
            gsfc = all_gsfc_list[gsfc_id]
            # vnf_sequence가 SFC의 VNF 종류를 저장하는 리스트라고 가정
            vnf_kind = gsfc.vnf_sequence[vnf_idx]

            # 위성이 지원하는 VNF 종류인 경우에만 로드 누적
            if vnf_kind in vnf_load and isinstance(vnf_size, (int, float)):
                vnf_load[vnf_kind] += vnf_size

        except IndexError:
            # gsfc_id나 vnf_idx가 유효하지 않은 경우 (데이터 불일치)
            continue

    # VNF 종류별 전체 로드 딕셔너리 반환
    return vnf_load

def get_node_type(node_id, all_gserver_list, all_sat_list):
    if node_id >= NUM_SATELLITES:
        node_type = 'gserver'
        node = all_gserver_list[node_id - NUM_SATELLITES]
    else:
        node_type = 'satellite'
        node = all_sat_list[node_id]

    return node_type, node

# 경로 내 위성들의 id 추출
def sat_ids(path):
    """[sat_id, meta] 또는 sat_id 를 sat_id 리스트로 정규화"""
    ids = []
    for step in path:
        if isinstance(step, (list, tuple)):
            if len(step) == 0:
                continue
            ids.append(step[0])
        else:
            ids.append(step)
    return ids

# 해당 gsfc에서 남은 경로 추출
def get_remain_path(gsfc):
    return gsfc.satellite_path[gsfc.cur_path_id:]

GSFC_LOG_HEADER = [
    "time_ms",
    "mode",
    "gsfc_id",
    "is_succeed",
    "is_dropped",
    "event",
    "vnf_sequence",
    "src_vsg",
    "dst_vsg",
    "gsfc_flow_rule",
    "vsg_path",
    "satellite_path",
    "processed_satellite_path",
    "state",
    "remaining ongoing time slot",
    "process delay",
    "queueing delay",
    "transmitting delay",
    "propagation delay",
    "total delay"
]

def init_gsfc_csv_log(csv_dir_path, mode):
    os.makedirs(csv_dir_path, exist_ok=True)
    file_path = os.path.join(csv_dir_path, f"gsfc_log.csv")

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(GSFC_LOG_HEADER)

    return file_path

def write_gsfc_csv_log(file_path, time_ms, gsfc, event):
    # event: "INIT_PATH", "PROCESS", "QUEUE", "TRANSMIT","PROPAGATE", 'DONE", "DROP"
    if not file_path:
        return

    def to_str(x):
        # 숫자/문자열은 그대로, 나머지는 json으로
        if isinstance(x, (int, float, str)) or x is None:
            return x
        try:
            return json.dumps(x, ensure_ascii=False)
        except TypeError:
            return str(x)

    processed_satellite_path = gsfc.satellite_path[:gsfc.cur_path_id]

    row = [
        time_ms,
        getattr(gsfc, "mode", ""),
        getattr(gsfc, "id", None),
        getattr(gsfc, "is_succeed", False),
        getattr(gsfc, "is_dropped", False),
        event,
        to_str(getattr(gsfc, "vnf_sequence", [])),
        getattr(gsfc, "src_vsg_id", ""),
        getattr(gsfc, "dst_vsg_id", ""),
        to_str(getattr(gsfc, "gsfc_flow_rule", [])),
        to_str(getattr(gsfc, "vsg_path", [])),
        to_str(getattr(gsfc, "satellite_path", [])),
        # to_str(getattr(gsfc, f"satellite_path{gsfc.cur_path_id}", [])),
        processed_satellite_path,
        to_str(getattr(gsfc, "state", -1)),
        to_str(getattr(gsfc, "DH_remaining_ongoing_time_slot", -1)),
        to_str(getattr(gsfc, "proc_delay_ms", 0)),
        to_str(getattr(gsfc, "queue_delay_ms", 0)),
        to_str(getattr(gsfc, "trans_delay_ms", 0)),
        to_str(getattr(gsfc, "prop_delay_ms", 0)),
        to_str(getattr(gsfc, "proc_delay_ms", 0) +
               getattr(gsfc, "queue_delay_ms", 0) +
               getattr(gsfc, "transmit_delay_ms", 0) +
               getattr(gsfc, "prop_delay_ms", 0))
    ]

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)



