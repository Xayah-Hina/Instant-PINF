import torch
import pathlib

import src.datamanager as datamanager


def load_data(device: torch.device, data_path: pathlib.Path) -> datamanager.HyFluidNeRFDataManager:
    config = datamanager.HyFluidNeRFDataManagerConfig(dataparser=datamanager.HyFluidDataParserConfig(data=data_path))
    manager = config.setup(device=device)
    return manager


if __name__ == '__main__':
    path = pathlib.Path("C:/Users/imeho/Documents/DataSets/InstantPINF/ScalarReal")
    manager = load_data(torch.device("cuda"), data_path=path)
