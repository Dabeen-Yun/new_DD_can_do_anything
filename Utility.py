from Params import *

import numpy as np
import networkx as nx
import csv
import os
import json
import pandas as pd
import glob
import matplotlib.pyplot as plt

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
               getattr(gsfc, "trans_delay_ms", 0) +
               getattr(gsfc, "prop_delay_ms", 0))
    ]

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def load_all_gsfc_logs(modes, data_rate_pairs, base_results_dir="./results"):
    """
    results/{NUM_GSFC}/{mode}/{sat_Mbps}sat_{gs_Mbps}gs/gsfc_log_{mode}.csv
    구조로 저장된 로그를 모두 읽어서 하나의 DataFrame으로 합친다.
    """
    dfs = []
    for sat_rate, gs_rate in data_rate_pairs:
        sat_mbps = sat_rate / 1e6
        gs_mbps = gs_rate / 1e6
        data_rate_label = f"{sat_mbps}sat_{gs_mbps}gs"

        for mode in modes:
            csv_dir = f"{base_results_dir}/{NUM_GSFC}/{mode}/{data_rate_label}/"
            csv_path = f"{csv_dir}{mode}_gsfc_log.csv"

            if not os.path.exists(csv_path):
                print(f"[WARN] CSV not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            # 혹시 mode 컬럼 없으면 붙여주기
            if "mode" not in df.columns:
                df["mode"] = mode

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
        "proc_delay_ms",
        "queue_delay_ms",
        "trans_delay_ms",
        "prop_delay_ms",
    ]
    for col in numeric_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    # e2e_delay_ms 계산 (이미 있으면 덮어써도 됨)
    df_all["e2e_delay_ms"] = (
        df_all.get("proc_delay_ms", 0)
        + df_all.get("queue_delay_ms", 0)
        + df_all.get("trans_delay_ms", 0)
        + df_all.get("prop_delay_ms", 0)
    )

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

def plot_e2e_vs_data_rate(df_succ, out_path="e2e_vs_data_rate_per_mode.png"):
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
    for mode in modes:
        df_mode = df_succ[df_succ["mode"] == mode]
        # mode + data_rate_label 그룹으로 평균 E2E
        grp = (
            df_mode.groupby("data_rate_label")["e2e_delay_ms"]
            .mean()
            .reindex(labels)
        )
        plt.plot(labels, grp.values, marker="o", label=mode)

    plt.xlabel("Data rate pair (sat_Mbps / gs_Mbps)")
    plt.ylabel("Average E2E delay [ms]")
    plt.title("E2E delay vs data_rate_pair (per mode)")
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

    # data_rate_label 단위로 loop
    for data_rate_label, df_pair in df_succ.groupby("data_rate_label"):
        modes = sorted(df_pair["mode"].dropna().unique())
        avg_e2e = (
            df_pair.groupby("mode")["e2e_delay_ms"]
            .mean()
            .reindex(modes)
        )

        x = np.arange(len(modes))

        plt.figure()
        plt.bar(x, avg_e2e.values)
        plt.xticks(x, modes)
        plt.ylabel("Average E2E delay [ms]")
        plt.title(f"E2E delay by mode (data_rate={data_rate_label})")
        plt.grid(True, axis="y")
        plt.tight_layout()

        filename = f"{prefix}{data_rate_label}.png"
        out_path = os.path.join(out_dir, filename)
        plt.savefig(out_path, dpi=300)
        # plt.show()
        print(f"[INFO] Saved {out_path}")

def plot_e2e_summary(modes, data_rate_pairs, base_results_dir="./results"):
    """
    main.py에서 시뮬 다 돈 뒤에 한 번 호출:
    - 모든 mode × data_rate_pair CSV를 읽고
    - is_succeed == True + 마지막 시점 row만 사용해서
    - 1) mode 고정, data_rate_pair 비교 (라인 그래프)
    - 2) data_rate_pair 고정, mode 비교 (bar 그래프)
    """
    df_all = load_all_gsfc_logs(modes, data_rate_pairs, base_results_dir)
    df_succ = preprocess_gsfc_logs(df_all)

    if df_succ.empty:
        print("[WARN] No successful GSFCs to plot")
        return

    # out_dir은 결과 위치 통일해서 관리
    out_dir = os.path.join(base_results_dir, str(NUM_GSFC), "plots")
    os.makedirs(out_dir, exist_ok=True)

    # 1) [모드별] data_rate_pair 비교
    plot_e2e_vs_data_rate(
        df_succ,
        out_path=os.path.join(out_dir, "e2e_vs_data_rate_per_mode.png"),
    )

    # 2) [data_rate_pair별] mode 비교
    plot_e2e_vs_mode(
        df_succ,
        out_dir=out_dir,
        prefix="e2e_vs_mode_",
    )
