from npnf.configs.utils import builds
from npnf.data.datasets.fip1 import BalancedFIP1Dataset, Fip1Facts, get_heights_dataset

TrainHeightDatasetConfig = builds(
    get_heights_dataset,
    split="train",
    datasets_offline_path="${oc.select:datasets_offline_path,null}",
)
ValHeightDatasetConfig = builds(
    get_heights_dataset,
    split="validation",
    datasets_offline_path="${oc.select:datasets_offline_path,null}",
)
TestPlotHeightDatasetConfig = builds(
    get_heights_dataset,
    split="test_plot",
    datasets_offline_path="${oc.select:datasets_offline_path,null}",
)
TestGenotypeHeightDatasetConfig = builds(
    get_heights_dataset,
    split="test_genotype",
    datasets_offline_path="${oc.select:datasets_offline_path,null}",
)
TestEnvironmentHeightDatasetConfig = builds(
    get_heights_dataset,
    split="test_environment",
    datasets_offline_path="${oc.select:datasets_offline_path,null}",
)
TestGenotypeEnvironmentHeightDatasetConfig = builds(
    get_heights_dataset,
    split="test_genotype_environment",
    datasets_offline_path="${oc.select:datasets_offline_path,null}",
)
TestAllHeightDatasetConfig = builds(
    get_heights_dataset,
    split=list(Fip1Facts().test_splits),
    datasets_offline_path="${oc.select:datasets_offline_path,null}",
)
AllSplitsHeightDatasetConfig = builds(
    get_heights_dataset,
    split=list(Fip1Facts().splits),
    datasets_offline_path="${oc.select:datasets_offline_path,null}",
)

# Balanced FIP1 datasets (train only - no need to balance validation/test sets)
BalancedTrainHeightDatasetConfig = builds(
    BalancedFIP1Dataset,
    split="train",
    weight_strategy="uniform",
    datasets_offline_path="${oc.select:datasets_offline_path,null}",
)

BalancedTrainHeightInverseFreqDatasetConfig = builds(
    BalancedFIP1Dataset,
    split="train",
    weight_strategy="inverse_frequency",
    datasets_offline_path="${oc.select:datasets_offline_path,null}",
)
