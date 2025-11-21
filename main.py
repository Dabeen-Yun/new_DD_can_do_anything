from tqdm import tqdm
from Params import *
from Simulation import *

class Main:
    data_rate_pairs = [  # [sat, gs] 단위 bps
        # # (40e6, 400e6),  # A 10배
        # (80e6, 320e6),  # B 10배
        # (160e6, 640e6),  # C 10배
        (320e6, 1280e6)
    ]

    modes = ['basic']  # dd, basic, noname

    for pair in tqdm(data_rate_pairs):
        for mode in modes:
            csv_dir_path = f"./results/{NUM_GSFC}/{mode}/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/"

            simulation = Simulation()
            simulation.simulation_proceeding(mode, pair, csv_dir_path)
