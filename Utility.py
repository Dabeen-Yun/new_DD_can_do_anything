from Params import *

import numpy as np
import networkx as nx
import csv
import os
import json
import pandas as pd
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import cv2
import matplotlib.pyplot as plt
import pickle
import ast
import math

def get_vsg_id_from_coords(vsg_list, lat, lon, eps=1e-9):
    """
    위경도 좌표를 입력받아 해당 위치의 VSG ID를 반환 (O(1))
    """
    for vsg in vsg_list:
        in_lat = (vsg.lat_min - eps) <= lat <= (vsg.lat_max + eps)
        in_lon = (vsg.lon_min - eps) <= lon <= (vsg.lon_max + eps)
        if in_lat and in_lon:
            return vsg.id

    input(f"No VSG found for coords lat={lat}, lon={lon}")
    return -1

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
    "gsfc_type",
    "tolerance_delay_ms",
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
    "total delay",
    "additional_path"
]

def init_gsfc_csv_log(csv_dir_path, mode):
    os.makedirs(csv_dir_path, exist_ok=True)
    file_path = os.path.join(csv_dir_path, f"{mode}_gsfc_log.csv")

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
        getattr(gsfc, "gsfc_type", None),
        getattr(gsfc, "tolerance_time_ms", None),
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
        # to_str(getattr(gsfc, "queue_delay_ms", 0)),
        to_str(getattr(gsfc, "proc_queue_delay_ms", 0)),
        to_str(getattr(gsfc, "trans_delay_ms", 0)),
        to_str(getattr(gsfc, "prop_delay_ms", 0)),
        to_str(getattr(gsfc, "proc_delay_ms", 0) +
               getattr(gsfc, "proc_queue_delay_ms", 0) +
               getattr(gsfc, "queue_delay_ms", 0) +
               getattr(gsfc, "trans_delay_ms", 0) +
               getattr(gsfc, "prop_delay_ms", 0)),
        getattr(gsfc, "additional_path", 0)
    ]

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    setattr(gsfc, "additional_path", 0)

def calculate_additional_path_stats(file_path):
    if not os.path.exists(file_path):
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        df['additional_path'] = pd.to_numeric(df['additional_path'], errors='coerce').fillna(0)

        # 총량 계산 (합계)
        total_additional_path = df['additional_path'].sum()

        # 평균 계산 (결측값 제외 후 평균)
        # 전체 행에 대한 평균
        average_additional_path = df['additional_path'].mean()

        print(f"--- CSV 파일 분석 결과: {file_path} ---")
        print(f"전체 레코드 수: {len(df)}")
        print(f"'additional_path' 총량 (합계): {total_additional_path}")
        print(f"'additional_path' 평균: {average_additional_path:.4f}")

    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")

def calculate_success_hop_stats(file_path):
    if not os.path.exists(file_path):
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
        return

    try:
        df = pd.read_csv(file_path)

        # is_succeed 컬럼을 bool로 정규화
        succeed_col = df['is_succeed']

        if succeed_col.dtype == bool:
            success_df = df[df['is_succeed']]
        else:
            # 문자열/숫자 형태인 경우 ('True', 'False', 1, 0 등)
            success_df = df[succeed_col.astype(str).str.lower().isin(['true', '1', 'yes'])]

        total_hop_count = 0

        if success_df.empty:
            print(f"--- CSV 파일 분석 결과: {file_path} ---")
            print("성공한(success=True) GSFC가 없습니다.")
            return

        def count_hops(path_val):
            # 결측 처리
            if pd.isna(path_val):
                return 0

            # CSV 읽어오면 보통 문자열이므로 literal_eval로 파싱
            if isinstance(path_val, str):
                try:
                    parsed = ast.literal_eval(path_val)
                except Exception:
                    # 파싱 실패하면 0으로 처리
                    return 0
            else:
                parsed = path_val

            # 우리가 기대하는 형태: [[38, "src"], [27, "vnf1"], ...]
            if isinstance(parsed, list):
                return len(parsed)  # 요소 개수 = hop count (노드 개수 기준)
                # 만약 링크 개수 기준 hop 을 원하면 len(parsed) - 1 로 바꾸면 됨

            return 0

        # hop_count 컬럼 추가
        success_df = success_df.copy()
        success_df['hop_count'] = success_df['satellite_path'].apply(count_hops)
        total_hop_count += success_df['hop_count']

        total_hops = success_df['hop_count'].sum()
        avg_hops = success_df['hop_count'].mean()
        max_hops = success_df['hop_count'].max()
        min_hops = success_df['hop_count'].min()

        print(f"--- CSV 파일 분석 결과 (success == True): {file_path} ---")
        print(f"성공한 GSFC 레코드 수: {len(success_df)}")
        print(f"총 hop 수 합계: {total_hops}")
        print(f"평균 hop 수: {avg_hops:.4f}")
        print(f"최소 hop 수: {min_hops}")
        print(f"최대 hop 수: {max_hops}")

        # 필요하면 개별 hop_count 분포를 보고 싶을 때:
        # print(success_df[['gsfc_id', 'satellite_path', 'hop_count']].head())

    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")

SATELLITE_LOG_HEADER = [
    "time_ms",
    "mode",
    "sat_id",
    "lon",
    "lat",
    "vnf list",
    "adjacent sat index",
    "process queue size",
    "process queue",
]

def init_sat_csv_log(csv_dir_path, mode):
    os.makedirs(csv_dir_path, exist_ok=True)
    file_path = os.path.join(csv_dir_path, f"{mode}_sat_log.csv")

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SATELLITE_LOG_HEADER)

    return file_path

def write_sat_csv_log(sat):
    if not sat.sat_log_path:
        return

    process_queue = getattr(sat, "process_queue", [])

    row = [
        getattr(sat, "time", -1),
        getattr(sat, "mode", ""),
        getattr(sat, "id", -1),
        getattr(sat, "lon", 0),
        getattr(sat, "lat", 0),
        getattr(sat, "vnf_list", []),
        getattr(sat, "adj_sat_index_list", []),
        len(process_queue),
        process_queue,
    ]

    with open(sat.sat_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

VSG_LOG_HEADER = [
    "time_ms",
    "mode",
    "vsg_id",
    "lon min",
    "lat min",
    "assigned vnfs",
    "satellites"
]

def init_vsg_csv_log(csv_dir_path, mode):
    os.makedirs(csv_dir_path, exist_ok=True)
    file_path = os.path.join(csv_dir_path, f"{mode}_vsg_log.csv")

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(VSG_LOG_HEADER)

    return file_path

def write_vsg_csv_log(vsg):
    if not vsg.vsg_log_path:
        return

    satellite_ids = [sat.id for sat in getattr(vsg, "satellites", [])]

    row = [
        getattr(vsg, "time", -1),
        getattr(vsg, "mode", ""),
        getattr(vsg, "id", -1),
        getattr(vsg, "lon_min", 0),
        getattr(vsg, "lat_min", 0),
        getattr(vsg, "assigned_vnfs", []),
        satellite_ids # satellite idx
    ]

    with open(vsg.vsg_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def save_networkx_graph(G, file_path):
    """NetworkX 그래프 객체를 Pickle 파일로 저장합니다."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(G, f)
        print(f"[INFO] NetworkX graph saved to {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save graph: {e}")

def load_networkx_graph(file_path):
    """Pickle 파일에서 NetworkX 그래프 객체를 로드합니다."""
    try:
        with open(file_path, 'rb') as f:
            G = pickle.load(f)
        print(f"[INFO] NetworkX graph loaded from {file_path}")
        return G
    except FileNotFoundError:
        print(f"[ERROR] Graph file not found: {file_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load graph: {e}")
        return None

def load_all_gsfc_logs(modes, lon_steps, data_rate_pairs, base_results_dir="./results"):
    """
    results/{NUM_GSFC*NUM_ITERATIONS}/{mode}/{sat_Mbps}sat_{gs_Mbps}gs/gsfc_log_{mode}.csv
    구조로 저장된 로그를 모두 읽어서 하나의 DataFrame으로 합친다.
    """
    dfs = []
    for sat_rate, gs_rate in data_rate_pairs:
        sat_mbps = sat_rate / 1e6
        gs_mbps = gs_rate / 1e6
        data_rate_label = f"{sat_mbps}sat_{gs_mbps}gs"
        for lon_step in lon_steps:
            for mode in modes:
                csv_dir = f"{base_results_dir}/{lon_step}_{NUM_GSFC*NUM_ITERATIONS}/{mode}/{data_rate_label}/"
                csv_path = f"{csv_dir}{mode}_gsfc_log.csv"

                if not os.path.exists(csv_path):
                    print(f"[WARN] CSV not found: {csv_path}")
                    continue

                df = pd.read_csv(csv_path)

                # 혹시 mode 컬럼 없으면 붙여주기
                if "mode" not in df.columns:
                    df["mode"] = mode

                # 혹시 lon_step 컬럼 없으면 붙여주기
                if "lon_step" not in df.columns:
                    df["lon_step"] = lon_step

                df["sat_rate_bps"] = sat_rate
                df["gs_rate_bps"] = gs_rate
                df["data_rate_label"] = data_rate_label

                dfs.append(df)

    if not dfs:
        print("[ERROR] No CSV loaded")
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all

def preprocess_gsfc_logs(df_all):
    """
    - is_succeed == True인 행만 사용
    - 동일 GSFC에 대해 마지막 time_ms (완료 시점)만 남기기
    - delay 숫자형 변환, e2e_delay_ms 계산
    """
    if df_all.empty:
        return df_all

    # is_succeed / is_dropped 정리
    for col in ["is_succeed", "is_dropped"]:
        if col in df_all.columns and df_all[col].dtype == object:
            df_all[col] = df_all[col].astype(str).str.lower().isin(["true", "1", "yes"])

    # 숫자형 컬럼 변환
    numeric_cols = [
        "time_ms",
        "process delay",
        "queueing delay",
        "transmitting delay",
        "propagation delay",
    ]
    for col in numeric_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    # e2e_delay_ms (total delay) 불러오기
    df_all.get("total delay", 0)

    # 성공한 GSFC만 사용
    df_succ = df_all[df_all["is_succeed"] == True].copy()

    if df_succ.empty:
        print("[WARN] No successful GSFCs found")
        return df_succ

    # 같은 GSFC가 여러 time_ms에 찍혀 있을 수 있으니, 마지막 시점만 사용
    # key: mode, data_rate_label, gsfc_id
    sort_cols = []
    for c in ["mode", "data_rate_label", "gsfc_id", "time_ms"]:
        if c in df_succ.columns:
            sort_cols.append(c)

    if "time_ms" in sort_cols:
        df_succ = df_succ.sort_values(sort_cols)
        group_cols = [c for c in ["mode", "data_rate_label", "gsfc_id"] if c in df_succ.columns]
        if group_cols:
            df_succ = df_succ.groupby(group_cols, as_index=False).tail(1)

    return df_succ

def plot_e2e_vs_data_rate(df_succ, lon_steps, out_path="e2e_vs_data_rate_per_mode.png"):
    """
    각 mode에 대해 data_rate_pair에 따른 E2E delay 변화를 라인 그래프로.
    """
    if df_succ.empty:
        print("[WARN] df_succ is empty, skip plot_e2e_vs_data_rate")
        return

    modes = sorted(df_succ["mode"].dropna().unique())
    # data_rate_label 순서를 sat_rate 기준으로 정렬
    # (sat_rate_bps가 들어있다는 가정)
    df_succ["sat_rate_mbps"] = df_succ["sat_rate_bps"] / 1e6

    # label 순서
    label_order = (
        df_succ[["data_rate_label", "sat_rate_mbps"]]
        .drop_duplicates()
        .sort_values("sat_rate_mbps")
    )
    labels = label_order["data_rate_label"].tolist()

    plt.figure()
    for lon_step in lon_steps:
        for mode in modes:
            df_mode = df_succ[
                (df_succ["mode"] == mode) & (df_succ["lon_step"] == lon_step)
            ]
            # mode + data_rate_label 그룹으로 평균 E2E
            grp = (
                df_mode.groupby("data_rate_label")["total delay"]
                .mean()
                .reindex(labels)
            )
            plt.plot(labels, grp.values, marker="o", label=f"{mode}, lon_step={lon_step}")

    plt.xlabel("Data rate pair (sat_Mbps / gs_Mbps)")
    plt.ylabel("Average E2E delay [ms]")
    plt.title("E2E delay vs data_rate_pair (per mode, per lon_step)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    # plt.show()
    print(f"[INFO] Saved {out_path}")

def plot_e2e_vs_mode(df_succ, out_dir="./", prefix="e2e_vs_mode_"):
    """
    각 data_rate_pair에 대해, mode별 E2E delay 평균을 bar 그래프로.
    data_rate_pair가 여러 개면, pair별로 파일을 여러 개 만든다.
    """
    if df_succ.empty:
        print("[WARN] df_succ is empty, skip plot_e2e_vs_mode")
        return

    os.makedirs(out_dir, exist_ok=True)

    delay_components = [
        "process delay",
        "queueing delay",
        "transmitting delay",
        "propagation delay",
    ]

    # data_rate_label 단위로 loop
    for data_rate_label, df_pair in df_succ.groupby("data_rate_label"):
        df_agg = df_pair.groupby(["mode", "lon_step"])[delay_components].mean()

        # 데이터가 비어있으면 건너뜁니다.
        if df_agg.empty:
            print(f"[WARN] No aggregated data for data_rate_label {data_rate_label}")
            continue

        # 인덱스 정렬 (mode, lat_step 순)
        df_agg = df_agg.sort_index()  # MultiIndex: (mode, lat_step)

        # x축에 쓸 라벨: "mode\nlat{lat_step}"
        idx_tuples = df_agg.index.tolist()
        labels = [f"{mode}\nlat{lat_step}" for (mode, lat_step) in idx_tuples]

        x = np.arange(len(labels))

        plt.figure(figsize=(max(8, len(labels) * 0.6), 6))
        current_bottom = np.zeros(len(labels))
        bar_containers = {}

        for i, component in enumerate(delay_components):
            # i번째 컴포넌트의 평균 지연 값 (Numpy Array)
            delay_values = df_agg[component].values

            # bar를 그리고, 그 결과를 저장하여 나중에 주석(Annotation)에 사용
            bars = plt.bar(
                x,
                delay_values,
                label=component.replace(' delay', '').title(),  # 범례를 위해 텍스트 수정
                bottom=current_bottom
            )
            bar_containers[component] = bars

            # 다음 막대를 위해 현재 막대 높이를 current_bottom에 더합니다 (누적)
            current_bottom += delay_values

        def autolabel_stack(bars, segment_vals, current_bottom_vals):
            for bar, val, bottom_start in zip(bars, segment_vals, current_bottom_vals):
                if val > 0.1:  # 값이 너무 작으면 생략
                    y_center = bottom_start + val / 2.0
                    plt.text(bar.get_x() + bar.get_width() / 2.0, y_center,
                             f"{val:.1f}", ha="center", va="center", fontsize=8)

        current_bottom_annotation = np.zeros(len(labels))

        for i, component in enumerate(delay_components):
            delay_values = df_agg[component].values
            bars = bar_containers[component]

            # 주석 헬퍼 함수 호출
            autolabel_stack(bars, delay_values, current_bottom_annotation)

            # 다음 주석을 위해 누적 합 업데이트
            current_bottom_annotation += delay_values

        plt.xticks(x, labels, rotation=30)
        plt.ylabel("Average E2E Delay [ms]")
        plt.title(f"Average E2E Delay Breakdown by Mode (data_rate={data_rate_label})")
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.grid(True, axis="y", linestyle='--', alpha=0.6)
        plt.tight_layout()

        filename = f"{prefix}{data_rate_label}.png"
        out_path = os.path.join(out_dir, filename)

        plt.savefig(out_path, dpi=300)
        plt.close()  # 메모리 관리를 위해 플롯 닫기
        print(f"[INFO] Saved {out_path}")

def plot_e2e_summary(modes, lon_steps, data_rate_pairs, base_results_dir="./results"):
    """
    main.py에서 시뮬 다 돈 뒤에 한 번 호출:
    - 모든 mode × data_rate_pair CSV를 읽고
    - is_succeed == True + 마지막 시점 row만 사용해서
    - 1) mode 고정, data_rate_pair 비교 (라인 그래프)
    - 2) data_rate_pair 고정, mode 비교 (bar 그래프)
    """
    df_all = load_all_gsfc_logs(modes, lon_steps, data_rate_pairs, base_results_dir)
    df_succ = preprocess_gsfc_logs(df_all)

    if df_succ.empty:
        print("[WARN] No successful GSFCs to plot")
        return

    # out_dir은 결과 위치 통일해서 관리
    out_dir = os.path.join(base_results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # 1) [모드별] data_rate_pair 비교
    plot_e2e_vs_data_rate(
        df_succ,
        lon_steps,
        out_path=os.path.join(out_dir, "e2e_vs_data_rate_per_mode.png"),
    )

    # 2) [data_rate_pair별] mode 비교
    plot_e2e_vs_mode(
        df_succ,
        out_dir=out_dir,
        prefix="e2e_vs_mode_",
    )

# 파싱 헬퍼 함수 (CSV의 문자열을 리스트/딕셔너리로 변환)
def safe_eval(s):
    """CSV에 저장된 문자열 형태의 리스트/딕셔너리를 파이썬 객체로 안전하게 변환"""
    if isinstance(s, (list, dict)):
        return s
    if pd.isna(s) or s == "":
        return None
    try:
        # json.loads 대신 eval을 사용하여 튜플 리스트 형태도 파싱 (주의: 보안에 취약할 수 있으나 시뮬레이션 환경에서는 허용)
        return eval(str(s))
    except:
        return s

def load_data_for_time_tick(t, mode, csv_dir_path):
    """
    특정 시간(t)과 모드(mode)에 대한 GSFC, SAT, VSG 데이터를 CSV에서 로드합니다.
    """

    # 파일 경로 정의
    gsfc_log_path = os.path.join(csv_dir_path, f"{mode}_gsfc_log.csv")
    sat_log_path = os.path.join(csv_dir_path, f"{mode}_sat_log.csv")
    vsg_log_path = os.path.join(csv_dir_path, f"{mode}_vsg_log.csv")

    # 1. GSFC 데이터 로드
    try:
        df_gsfc = pd.read_csv(gsfc_log_path)
        # t 시점의 가장 최근 경로 (INIT_PATH 또는 최신 업데이트) 기록을 찾기 위해 필터링
        gsfc_data = df_gsfc[df_gsfc['time_ms'] <= t].sort_values(by='time_ms', ascending=False).drop_duplicates(
            subset=['gsfc_id'], keep='first').copy()
    except FileNotFoundError:
        print(f"[ERROR] GSFC log not found: {gsfc_log_path}")
        gsfc_data = pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Reading GSFC log failed: {e}")
        gsfc_data = pd.DataFrame()

    # 2. SAT 데이터 로드 (t 시점의 데이터)
    try:
        df_sat = pd.read_csv(sat_log_path)
        # t 시점의 데이터만 사용 (위성 위치, VNF, 인접 위성 목록)
        sat_data = df_sat[df_sat['time_ms'] == t].copy()
        # sat_data가 비어있다면, t=0 시점의 초기 토폴로지를 찾아야 할 수 있습니다. (여기서는 간단히 t=t 데이터만 로드)
    except FileNotFoundError:
        print(f"[ERROR] SAT log not found: {sat_log_path}")
        sat_data = pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Reading SAT log failed: {e}")
        sat_data = pd.DataFrame()

    # 3. VSG 데이터 로드 (t 시점의 데이터)
    try:
        df_vsg = pd.read_csv(vsg_log_path)
        # t 시점의 데이터만 사용 (경계, 할당 VNF)
        vsg_data = df_vsg[df_vsg['time_ms'] == t].copy()
    except FileNotFoundError:
        print(f"[ERROR] VSG log not found: {vsg_log_path}")
        vsg_data = pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Reading VSG log failed: {e}")
        vsg_data = pd.DataFrame()

    # 4. 데이터 구조 변환 (ID를 키로 하는 딕셔너리 리스트)
    # GSFC 데이터 (하나의 GSFC ID만 추적)
    gsfc_dict = {}
    if not gsfc_data.empty:
        # 'satellite_path'와 'processed_satellite_path'를 객체로 변환
        gsfc_data['satellite_path'] = gsfc_data['satellite_path'].apply(safe_eval)
        gsfc_data['processed_satellite_path'] = gsfc_data['processed_satellite_path'].apply(safe_eval)
        gsfc_dict = gsfc_data.set_index('gsfc_id', drop=False).to_dict('index')

    # 위성 데이터
    sat_list_dict = {}
    if not sat_data.empty:
        # VNF 리스트와 인접 목록 파싱
        sat_data['vnf list'] = sat_data['vnf list'].apply(safe_eval)
        sat_data['adjacent sat index'] = sat_data['adjacent sat index'].apply(safe_eval)
        sat_list_dict = sat_data.set_index('sat_id', drop=False).to_dict('index')

    # VSG 데이터
    vsg_list_dict = {}
    if not vsg_data.empty:
        # 할당 VNF와 위성 목록 파싱
        vsg_data['assigned vnfs'] = vsg_data['assigned vnfs'].apply(safe_eval)
        vsg_data['satellites'] = vsg_data['satellites'].apply(safe_eval)  # 여기에 sat_id 리스트가 들어있다고 가정
        vsg_list_dict = vsg_data.set_index('vsg_id', drop=False).to_dict('index')

    return gsfc_dict, sat_list_dict, vsg_list_dict

def safe_load_path(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "" or raw == "[]":
            return []
        try:
            return json.loads(raw)
        except:
            print("[WARN] json.loads 실패 → 빈 리스트 반환", raw)
            return []
    return []

def animate_one_gsfc(gsfc_id, modes, csv_dir_path):
    for mode in modes:
        video_writer = None
        video_fps = 30
        video_path = os.path.join(csv_dir_path, f"{mode}_gsfc{gsfc_id}_network_constellation.mp4")

        graph_file_path = os.path.join(csv_dir_path, f"{mode}_network_G.pkl")
        G = load_networkx_graph(graph_file_path)

        print(f"[INFO] Starting animation for mode {mode}, GSFC {gsfc_id}")

        try:
            df_gsfc_check = pd.read_csv(os.path.join(csv_dir_path, f"{mode}_gsfc_log.csv"))
            MAX_T = df_gsfc_check['time_ms'].max()
        except:
            MAX_T = 50

        for t in range(MAX_T):
            gsfc_dict, sat_dict, vsg_dict = load_data_for_time_tick(t, mode, csv_dir_path)

            gsfc_record = gsfc_dict.get(gsfc_id)

            sat_list = list(sat_dict.values())
            vsg_list = list(vsg_dict.values())

            # 컬러맵 생성 (VSG별 색상)
            cmap = cm.get_cmap('tab20', len(vsg_list))
            vsg_colors = {vsg['vsg_id']: cmap(vsg['vsg_id']) for vsg in vsg_list}

            fig = plt.figure(figsize=(24, 12))
            ax = plt.gca()

            # 0. VSG 영역 표현
            for vsg in vsg_list:
                rect = Rectangle((vsg['lon min'], vsg['lat min']), LON_STEP, LAT_STEP,
                                 linewidth=0.8, edgecolor=vsg_colors[vsg['vsg_id']],
                                 facecolor=vsg_colors[vsg['vsg_id']],
                                 alpha=0.4, zorder=0)
                ax.add_patch(rect)

                if vsg['assigned vnfs'] and len(vsg['assigned vnfs']) > 1:
                    ax.annotate(f"VNF {vsg['assigned vnfs']}", (vsg['lon min'] + 1, vsg['lat min'] + 1),
                                 fontsize=13, color='black', alpha=0.8, zorder=12)

            # 1. ISL (adjacency edge) 그리기
            sat_positions = {sat['sat_id']: (sat['lon'], sat['lat']) for sat in sat_list}

            for sat in sat_list:
                # adj_sat_index_list 대신 'adjacent sat index' 사용
                for nbr_id in sat['adjacent sat index']:
                    if nbr_id == -1 or nbr_id not in sat_dict:
                        continue
                    nbr_sat = sat_dict[nbr_id]
                    ax.plot([sat['lon'], nbr_sat['lon']], [sat['lat'], nbr_sat['lat']],
                            color='gray', linewidth=1.0, alpha=0.3, zorder=1)

                # 4. 위성 인덱스 모두 표시
                ax.annotate(str(sat['sat_id']), (sat['lon'], sat['lat']), fontsize=13, alpha=0.7, zorder=5)

            # 2. 위성 산점도 (VSG 영역별 색상)
            for sat in sat_list:
                # sat 객체에 vsg_id가 없으므로, VSG 데이터에서 해당 sat_id를 포함하는 VSG를 찾아 색상을 결정해야 합니다.
                vsg_id_of_sat = next((vsg['vsg_id'] for vsg in vsg_list if sat['sat_id'] in vsg['satellites']),
                                     None)
                color = vsg_colors.get(vsg_id_of_sat, 'blue')  # 기본 색상 설정

                ax.scatter(sat['lon'], sat['lat'], s=100, color=color, edgecolors='black',
                           linewidths=0.8, alpha=0.6, zorder=2)

                # 3. VNF 수행 위성 강조
                if sat['vnf list']:
                    ax.scatter(sat['lon'], sat['lat'], marker='*', s=80, color='red', edgecolors='black',
                               linewidths=0.8,
                               zorder=4)
                    ax.annotate(f"VNF {sat['vnf list']}", (sat['lon'] + 3.0, sat['lat'] + 2.0),
                                fontsize=13, color='darkred', alpha=0.8, zorder=12)

            if gsfc_record:
                satellite_path = safe_load_path(gsfc_record.get('satellite_path', []))
                processed_path = safe_load_path(gsfc_record.get('processed_satellite_path', []))

                # 전체 경로에서 processed_path를 제외한 나머지 경로
                # satellite_path가 전체 경로이고 processed_path가 지나온 경로라면,
                # remaining_path를 계산하기 위해 인덱스를 사용해야 합니다.

                # CSV 저장 방식에 따라 처리 방식이 달라짐.
                # (A) 전체 경로와 현재 인덱스가 저장된 경우
                # (B) 지나온 경로와 전체 경로가 분리되어 저장된 경우

                # 여기서는 'processed_satellite_path'가 `gsfc.satellite_path[:gsfc.cur_path_id]`로 저장되었으므로,
                # 전체 경로와 cur_path_id를 사용합니다.
                if isinstance(satellite_path, list) or isinstance(processed_path, list):
                    # 전체 경로에서 processed_path의 길이만큼이 지나갔다고 가정
                    cur_path_len = len(processed_path)

                    processed_path_ids = satellite_path[:cur_path_len]
                    remain_path_ids = satellite_path[cur_path_len:]

                    processed_edges = []
                    remaining_edges = []

                    processed_sat_ids = sat_ids(processed_path_ids)
                    remaining_sat_ids = sat_ids(remain_path_ids)
                    all_tracked_sat_ids = list(set(processed_sat_ids + remaining_sat_ids))

                    # A. Processed Path 엣지 추출
                    if len(processed_path_ids) >= 2:
                        for i in range(len(processed_path_ids) - 1):
                            prev_sat_id = processed_path_ids[i][0]
                            current_sat_id = processed_path_ids[i + 1][0]
                            if current_sat_id != prev_sat_id:
                                processed_edges.append((prev_sat_id, current_sat_id))

                    # B. Remaining Path 엣지 추출
                    if remain_path_ids:
                        # Processed Path와 Remain Path 연결 엣지
                        if processed_path_ids:
                            last_processed_sat_id = processed_path_ids[-1][0]
                            first_remaining_sat_id = remain_path_ids[0][0]
                            if last_processed_sat_id != first_remaining_sat_id:
                                remaining_edges.append((last_processed_sat_id, first_remaining_sat_id))

                        # 나머지 Remaining Path 엣지
                        if len(remain_path_ids) >= 2:
                            for i in range(len(remain_path_ids) - 1):
                                prev_sat_id = remain_path_ids[i][0]
                                current_sat_id = remain_path_ids[i + 1][0]
                                if current_sat_id != prev_sat_id:
                                    remaining_edges.append((prev_sat_id, current_sat_id))

                    # 하이라이팅 시각화 (NetworkX G를 사용)

                    # 4-1. 노드 하이라이팅 (전체 경로 노드)
                    nx.draw_networkx_nodes(G, pos=sat_positions, nodelist=all_tracked_sat_ids,
                                           node_color='gray', node_size=150, ax=ax)

                    # 4-2. Processed Path 엣지 (초록색)
                    nx.draw_networkx_edges(G, pos=sat_positions, edgelist=processed_edges,
                                           edge_color='green', width=2.5, ax=ax)

                    # 4-3. Remaining Path 엣지 (빨간색)
                    nx.draw_networkx_edges(G, pos=sat_positions, edgelist=remaining_edges,
                                           edge_color='red', width=2.5, ax=ax)

                    # 4-4. 현재 위치 위성 강조
                    if processed_path_ids:
                        current_sat_id = processed_path_ids[-1][0]
                        nx.draw_networkx_nodes(G, pos=sat_positions, nodelist=[current_sat_id],
                                               node_color='yellow', node_size=200, ax=ax)

            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title(f"Network Constellation at {t}ms (GSFC ID: {gsfc_id}, Mode: {mode})")
            plt.grid(True)
            plt.tight_layout()

            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape((height, width, 3))

            # VideoWriter 초기화
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    video_path, fourcc,
                    video_fps,
                    (width, height)
                )

            # OpenCV는 BGR이라 변환 후 write
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

            plt.close(fig)

            # 루프가 끝나고 나서 VideoWriter 해제
        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Saved video to {video_path}")