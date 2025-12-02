from tqdm import tqdm
from Params import *
from Simulation import *
# from Simulation_small import *
from Utility import *

# TODO. vnf는 위성에 탑재되는 것, gsfc는 vnf 탑재가 아니라 vnf 수행이 필요 -> processing rate에서는 vnf size가 아니라 해당 패킷 사이즈가 필요

class Main:
    data_rate_pairs = [  # [sat, gs] 단위 bps 1e6
        (250e3, 250e3), # 250Mbps
    ]

    modes = ['basic', 'dd'] #['basic', 'dd']  # dd, basic, sd, upgrade_sd
    lon_steps = [36, 72, 180] #[36, 72]
    NUM_RUNS = 10

    for run_idx in range(NUM_RUNS):
        seed = 1000 + run_idx
        for pair in tqdm(data_rate_pairs):
            for lon_step in tqdm(lon_steps):
                for mode in modes:
                    csv_dir_path = f"./results/run_{run_idx}/{lon_step}_{NUM_GSFC*NUM_ITERATIONS}/{mode}/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/"

                    simulation = Simulation()
                    simulation.simulation_proceeding(mode, lon_step, pair, csv_dir_path)

    for run_idx in range(NUM_RUNS):
        for pair in tqdm(data_rate_pairs):
            for lon_step in tqdm(lon_steps):
                for mode in modes:
                    csv_dir_path = f"./results/run_{run_idx}/{lon_step}_{NUM_GSFC * NUM_ITERATIONS}/{mode}/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/{mode}_gsfc_log.csv"
                    calculate_additional_path_stats(csv_dir_path)
                    calculate_success_hop_stats(csv_dir_path)

    # 시뮬 다 돌리고 나서 한 번만 호출
    plot_e2e_summary(
        modes=modes,
        lon_steps=lon_steps,
        seed_nums = list(range(NUM_RUNS)),
        data_rate_pairs=data_rate_pairs,
        base_results_dir="./results",
    )