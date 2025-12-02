# constellation
POLAR_LATITUDE = 70
CONSTELLATION_PARAM_F = 0
ORBIT_ALTITUDE = 780000    # m
R_EARTH_RADIUS = 6371000   # m
SATELLITE_ORBITAL_PERIOD = 5400 #sec (현재 사용 안 함 ms 단위로 변환 필요?)
NUM_ORBITS = 6
NUM_SATELLITES = 66
NUM_SATELLITES_PER_ORBIT = NUM_SATELLITES // NUM_ORBITS
SATELLITE_SPEED = 7.4 #km/s 27000 km/h
DEGREE_TO_KM = 111 # 6378 지구 반지름에 따른 위도 1도 거리
LAT_RANGE = [-90, 90] #[-30, 30] #[-90, 90]
LON_RANGE = [-180, 180] #[-180, -108] #[-180, 180]

LAT_STEP = 60
LON_STEP = 72 #36, 72
PARAM_C = 299792458 # [m/s]

VNF_SIZE = 1e6 # [bit]  ≈125 KB

# simulation
NUM_ITERATIONS = 50 # ms
NUM_GSFC = 1 #33 # int(2*1024*1024/SFC_SIZE*8) #[per ms]

# VSG
# 0: FW
# 1: DPI
# 2: UE-UE local forwarder
# 3: IP-over-IP
# 4: compress
# 5: security wrapper
# 6: egress
VNF_TYPES_PER_VSG = (1, 7)

NUM_VNFS_PER_SAT = 4 #10
NUM_VNFS_PER_VSG = 4 #2

# satellite capacity
# SAT_QUEUE_SIZE = (SFC_SIZE // 8) * 50 * 8 # [bit] (SFC_SIZE[B] * 50) * 8
SAT_LINK_CAPACITY = 250 * 1e6 # [Mbps] -> [bps]
# SAT_PROCESSING_RATE: 50 MB/S -> 50 * 8 Mbps = 400 Mbps
SAT_PROCESSING_RATE = 400 * 1e6 # [Mbps] -> [bps]
SAT_NUM_PROCESS_VNF = 2 # 동시에 처리 가능한 VNF 수

# Gserver capacity
# GSERVER_QUEUE_SIZE = (SFC_SIZE // 8) * 100 * 8 # [bit] (SFC_SIZE[B] * 100) * 8
GSERVER_LINK_CAPACITY = 500 * 1e6 # [bps]
# GSERVER_PROCESSING_RATE: 200 MB/S -> 200 * 8 Mbps = 1600 Mbps
GSERVER_PROCESSING_RATE = 1600 * 1e6 # [bps]
GSERVER_NUM_PROCESS_VNF = 4 # 동시에 처리 가능한 VNF 수

# mMTC(S&F): DDoS[1] → FW[2] → TCP[3] → NAT[4]
# uRLLC(Local): FW[2]
# eMBB(UE-SAT-UE): DDoS[1] → FW[2] → LB[5] → TCP[3] → VideoOpt[6]
SFC_EMBB_SEQ = ['1', '2', '5', '3', '6']         # 고처리량, 보안, 주소변환
SFC_URLLC_SEQ = ['2', '5']     # 저지연, 경로 최적화
SFC_MMTC_SEQ = ['1', '2', '3', '4']             # 대규모 연결 관리

# eMBB 파라미터
EMBB_ARRIVAL_RATE = 1800    # packets per second (Lambda)
EMBB_LATENCY_LIMIT = 500    # ms
EMBB_PACKET_MIN_SIZE = 50   # bytes
EMBB_PACKET_MAX_SIZE = 600  # bytes
EMBB_PARETO_SHAPE = 1.5     # Pareto shape parameter (일반적인 인터넷 트래픽)

# URLLC 파라미터
URLLC_PERIOD = 2          # ms (Arrival interval)
URLLC_LATENCY_LIMIT = 100 # ms (0.5 ~ 1ms)
URLLC_PACKET_SIZE = 32    # bytes (Fixed)

# mMTC 파라미터
MMTC_PACKET_SIZE = 100 # bytes
MMTC_LATENCY_LIMIT = 10000 # ms (10 seconds - Delay Tolerant)
MMTC_DENSITY_FACTOR = 0.5 # 배경 트래픽 밀도 조절용 상수

# VSG size 업데이트
VSG_EVAL_INTERVAL_TIC = 1000       # 예: 1000 tic마다 한 번 평가
VSG_EVAL_WINDOW_SIZE = 200        # 예: 최근 200개의 GSFC 결과 기준
VSG_FAIL_RATIO_THRESHOLD = 0.2    # 이 이상 실패하면 VSG 분할 후보
VSG_SUCCESS_RATIO_THRESHOLD = 0.98  # 이 이상이면 병합 후보 (성공률 매우 좋을 때)

BATCH_SIZE = 30
TIME_TO_CHECK_MERGE = 50