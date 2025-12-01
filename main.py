from tqdm import tqdm
from Params import *
from Simulation import *
from Utility import *

# TODO. vnf는 위성에 탑재되는 것, gsfc는 vnf 탑재가 아니라 vnf 수행이 필요 -> processing rate에서는 vnf size가 아니라 해당 패킷 사이즈가 필요


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

    modes = ['basic']  # dd, basic, sd, upgrade_sd
    #main: 15, dd: 30

    for pair in tqdm(data_rate_pairs):
        for mode in modes:
            csv_dir_path = f"./results/{NUM_GSFC*NUM_ITERATIONS}/{mode}/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/"

            simulation = Simulation()
            simulation.simulation_proceeding(mode, pair, csv_dir_path)

    for pair in tqdm(data_rate_pairs):
        for mode in modes:
            csv_dir_path = f"./results/{NUM_GSFC * NUM_ITERATIONS}/{mode}/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/{mode}_gsfc_log.csv"
            calculate_additional_path_stats(csv_dir_path)
            calculate_success_hop_stats(csv_dir_path)
            # animate_one_gsfc(0, modes, csv_dir_path)

    # 시뮬 다 돌리고 나서 한 번만 호출
    plot_e2e_summary(
        modes=modes,
        data_rate_pairs=data_rate_pairs,
        base_results_dir="./results",
    )