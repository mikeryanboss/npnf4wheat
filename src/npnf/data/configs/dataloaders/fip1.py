from npnf.data.configs.dataloaders.dataloader import (
    TrainBaseDataloaderConfig,
    ValBaseDataloaderConfig,
)
from npnf.data.configs.datasets.fip1 import (
    AllSplitsHeightDatasetConfig as Fip1AllSplitsHeightDatasetConfig,
    BalancedTrainHeightDatasetConfig as Fip1BalancedTrainHeightDatasetConfig,
    BalancedTrainHeightInverseFreqDatasetConfig as Fip1BalancedTrainHeightInverseFreqDatasetConfig,  # noqa: E501
    TestAllHeightDatasetConfig as Fip1TestAllHeightDatasetConfig,
    TestEnvironmentHeightDatasetConfig as Fip1TestEnvironmentHeightDatasetConfig,
    TestGenotypeEnvironmentHeightDatasetConfig as Fip1TestGenotypeEnvironmentHeightDatasetConfig,  # noqa: E501
    TestGenotypeHeightDatasetConfig as Fip1TestGenotypeHeightDatasetConfig,
    TestPlotHeightDatasetConfig as Fip1TestPlotHeightDatasetConfig,
    TrainHeightDatasetConfig as Fip1TrainHeightDatasetConfig,
    ValHeightDatasetConfig as Fip1ValHeightDatasetConfig,
)

TrainHeightDataLoaderConfig = TrainBaseDataloaderConfig(
    dataset=Fip1TrainHeightDatasetConfig, num_workers=12, drop_last=True
)
BalancedTrainHeightDataLoaderConfig = TrainBaseDataloaderConfig(
    dataset=Fip1BalancedTrainHeightDatasetConfig, num_workers=12, drop_last=True
)
BalancedTrainHeightInverseFreqDataLoaderConfig = TrainBaseDataloaderConfig(
    dataset=Fip1BalancedTrainHeightInverseFreqDatasetConfig,
    num_workers=12,
    drop_last=True,
)
ValHeightDataLoaderConfig = ValBaseDataloaderConfig(dataset=Fip1ValHeightDatasetConfig)
TestPlotHeightDataLoaderConfig = ValBaseDataloaderConfig(
    dataset=Fip1TestPlotHeightDatasetConfig
)
TestGenotypeHeightDataLoaderConfig = ValBaseDataloaderConfig(
    dataset=Fip1TestGenotypeHeightDatasetConfig
)
TestEnvironmentHeightDataLoaderConfig = ValBaseDataloaderConfig(
    dataset=Fip1TestEnvironmentHeightDatasetConfig
)
TestGenotypeEnvironmentHeightDataLoaderConfig = ValBaseDataloaderConfig(
    dataset=Fip1TestGenotypeEnvironmentHeightDatasetConfig
)
TestAllHeightDataLoaderConfig = ValBaseDataloaderConfig(
    dataset=Fip1TestAllHeightDatasetConfig
)
AllSplitsHeightDataLoaderConfig = ValBaseDataloaderConfig(
    dataset=Fip1AllSplitsHeightDatasetConfig
)
