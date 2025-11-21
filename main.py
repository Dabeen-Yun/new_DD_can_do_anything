from tqdm import tqdm
from Params import *
from Simulation import *
from Utility import *

class Main:
    data_rate_pairs = [  # [sat, gs] 단위 bps
        # # (40e6, 400e6),  # A 10배
        # (80e6, 320e6),  # B 10배
        # (160e6, 640e6),  # C 10배
        (320e6, 1280e6)
    ]

    modes = ['basic', 'sd']  # dd, basic, sd

    for pair in tqdm(data_rate_pairs):
        for mode in modes:
            csv_dir_path = f"./results/{NUM_GSFC}/{mode}/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/"

            simulation = Simulation()
            simulation.simulation_proceeding(mode, pair, csv_dir_path)

    # 시뮬 다 돌리고 나서 한 번만 호출
    plot_e2e_summary(
        modes=modes,
        data_rate_pairs=data_rate_pairs,
        base_results_dir="./results",
    )