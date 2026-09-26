from hydra_zen import make_config

from npnf.data.configs.dataloaders.dataloader import TrainFip1HeightsDataloaderConfig
from npnf.data.configs.dataloaders.fip1 import (
    ValHeightDataLoaderConfig as Fip1ValHeightDataLoaderConfig,
)
from npnf.data.configs.dataloaders.synthetic import (
    ValDataLoader as SyntheticValDataLoaderConfig,
)
from npnf.data.configs.datasets.combined import (
    TrainCombinedDatasetConfig,
    WeightedYearsiteTrainInverseFreqConfig,
    WeightedYearsiteTrainUniformConfig,
)

# Combined Dataset configurations
TrainCombinedDataLoaderConfig = TrainFip1HeightsDataloaderConfig(
    dataset=TrainCombinedDatasetConfig
)


# Default: Use combined dataset with FIP1 validation
CombinedDataLoadersConfig = make_config(
    train=TrainCombinedDataLoaderConfig, validation=Fip1ValHeightDataLoaderConfig
)

# Alternative configurations
CombinedSyntheticValDataLoadersConfig = make_config(
    train=TrainCombinedDataLoaderConfig, validation=SyntheticValDataLoaderConfig
)


# Weighted yearsite dataloader configs
WeightedYearsiteTrainUniformDataLoaderConfig = TrainFip1HeightsDataloaderConfig(
    dataset=WeightedYearsiteTrainUniformConfig
)

WeightedYearsiteTrainInverseFreqDataLoaderConfig = TrainFip1HeightsDataloaderConfig(
    dataset=WeightedYearsiteTrainInverseFreqConfig
)


# Complete dataloader configuration combinations
WeightedYearsiteUniformDataLoadersConfig = make_config(
    train=WeightedYearsiteTrainUniformDataLoaderConfig,
    validation=Fip1ValHeightDataLoaderConfig,
)

WeightedYearsiteInverseFreqDataLoadersConfig = make_config(
    train=WeightedYearsiteTrainInverseFreqDataLoaderConfig,
    validation=Fip1ValHeightDataLoaderConfig,
)
