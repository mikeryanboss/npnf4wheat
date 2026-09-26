from npnf.configs.utils import builds
from npnf.data.configs.datasets.fip1 import (
    BalancedTrainHeightDatasetConfig as Fip1BalancedTrainHeightDatasetConfig,
    BalancedTrainHeightInverseFreqDatasetConfig as Fip1BalancedTrainInverseFreqConfig,
    TrainHeightDatasetConfig as Fip1TrainHeightDatasetConfig,
)
from npnf.data.configs.datasets.synthetic import (
    TrainConfig_64k as SyntheticTrainDatasetConfig,
)
from npnf.data.datasets.combined import WeightedYearsiteCombinedDataset

TrainCombinedDatasetConfig = builds(
    WeightedYearsiteCombinedDataset,
    dataset_real=Fip1TrainHeightDatasetConfig,
    dataset_synth=SyntheticTrainDatasetConfig,
    virtual_size=10000,
    p_synth=0.5,
)


# Combined datasets with balanced FIP1 datasets
WeightedYearsiteTrainUniformConfig = builds(
    WeightedYearsiteCombinedDataset,
    dataset_real=Fip1BalancedTrainHeightDatasetConfig,  # Balanced uniform FIP1
    dataset_synth=SyntheticTrainDatasetConfig,
    virtual_size=10000,
)

WeightedYearsiteTrainInverseFreqConfig = builds(
    WeightedYearsiteCombinedDataset,
    dataset_real=Fip1BalancedTrainInverseFreqConfig,  # Balanced inverse freq FIP1
    dataset_synth=SyntheticTrainDatasetConfig,
    virtual_size=10000,
)
