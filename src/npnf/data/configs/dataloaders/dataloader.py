from hydra_zen import builds, store
from torch.utils.data import DataLoader

from npnf.configs.utils import populate_builds
from npnf.data.configs.process.collate import CollateFip1HeightsFnConfig

DataLoaderConfig = populate_builds(
    DataLoader, batch_size="${per_device_batch_size}", num_workers=0, pin_memory=False
)

TrainBaseDataloaderConfig = builds(
    DataLoader,
    collate_fn=CollateFip1HeightsFnConfig,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    builds_bases=(DataLoaderConfig,),
)
ValBaseDataloaderConfig = builds(
    DataLoader, collate_fn=CollateFip1HeightsFnConfig, builds_bases=(DataLoaderConfig,)
)

# Specialized base configs with FIP1 Heights collate function built-in
TrainFip1HeightsDataloaderConfig = builds(
    DataLoader, builds_bases=(TrainBaseDataloaderConfig,)
)

ValFip1HeightsDataloaderConfig = builds(
    DataLoader, builds_bases=(ValBaseDataloaderConfig,)
)

TestFip1HeightsDataloaderConfig = builds(
    DataLoader, num_workers=4, builds_bases=(ValFip1HeightsDataloaderConfig,)
)

dataloader_store = store(group="dataloader")
dataloaders_store = store(group="dataloaders")
