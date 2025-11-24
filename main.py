from tqdm import tqdm
from Params import *
from Simulation import *
from Utility import *

# 실행 프롬프트 main - NUM_ITERATIONS 101, NUM_GSFC 1, VNF 개수 줄여서 'dd', 'basic', 'sd' 기법 실행

class Main:
    data_rate_pairs = [  # [sat, gs] 단위 bps 1e6
        # (250, 250e3),
        # (2500, 250e3),
        # (250e2, 250e3),
        # (40e3, 100e3), # 100Mbps
        (250e3, 250e3), # 250Mbps
        # (40e3, 400e3),  # A 10배
        # (80e3, 320e3),  # B 10배
        # (100e3, 300e3),
        # (160e6, 640e6),  # C 10배
        # (320e3, 1280e3)
    ]

    modes = ['dd','basic','sd']  # dd, basic, sd, upgrade_sd

    for pair in tqdm(data_rate_pairs):
        for mode in modes:
            csv_dir_path = f"./results/{NUM_GSFC*NUM_ITERATIONS}/{mode}/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/"

            simulation = Simulation()
            simulation.simulation_proceeding(mode, pair, csv_dir_path)

    # for pair in tqdm(data_rate_pairs):
    #     for mode in modes:
    #         csv_dir_path = f"./results/{NUM_GSFC * NUM_ITERATIONS}/{mode}/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/"
    #
    #         animate_one_gsfc(0, modes, csv_dir_path)

    # 시뮬 다 돌리고 나서 한 번만 호출
    plot_e2e_summary(
        modes=modes,
        data_rate_pairs=data_rate_pairs,
        base_results_dir="./results",
    )