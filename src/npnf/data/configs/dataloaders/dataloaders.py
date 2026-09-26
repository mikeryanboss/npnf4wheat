from hydra_zen import make_config

from npnf.data.configs.dataloaders.combined import (
    CombinedDataLoadersConfig,
    CombinedSyntheticValDataLoadersConfig,
    WeightedYearsiteInverseFreqDataLoadersConfig,
    WeightedYearsiteUniformDataLoadersConfig,
)
from npnf.data.configs.dataloaders.dataloader import dataloader_store, dataloaders_store
from npnf.data.configs.dataloaders.fip1 import (
    AllSplitsHeightDataLoaderConfig as Fip1AllSplitsHeightDataLoaderConfig,
    BalancedTrainHeightDataLoaderConfig as Fip1BalancedTrainHeightDataLoaderConfig,
    BalancedTrainHeightInverseFreqDataLoaderConfig as Fip1BalancedTrainHeightInverseFreqDataLoaderConfig,  # noqa: E501
    TestAllHeightDataLoaderConfig as Fip1TestAllHeightDataLoaderConfig,
    TestEnvironmentHeightDataLoaderConfig as Fip1TestEnvironmentHeightDataLoaderConfig,
    TestGenotypeEnvironmentHeightDataLoaderConfig as Fip1TestGenotypeEnvironmentHeightDataLoaderConfig,  # noqa: E501
    TestGenotypeHeightDataLoaderConfig as Fip1TestGenotypeHeightDataLoaderConfig,
    TestPlotHeightDataLoaderConfig as Fip1TestPlotHeightDataLoaderConfig,
    TrainHeightDataLoaderConfig as Fip1TrainHeightDataLoaderConfig,
    ValHeightDataLoaderConfig as Fip1ValHeightDataLoaderConfig,
)
from npnf.data.configs.dataloaders.synthetic import (
    ShiftedEnvDataLoaders,
    ShiftedGenoDataLoaders,
    ShiftedSeenDataLoaders,
    ShiftedUnseenDataLoaders,
    TestAllDataLoaders,
    TestAllNoLodgingDataLoaders,
    TestEnvironmentDataLoaders,
    TestEnvironmentLodging1_3DataLoaders,
    TestEnvironmentLodging1_5DataLoaders,
    TestEnvironmentNoise0_02DataLoaders,
    TestEnvironmentNoLodgingDataLoaders,
    TestGenotypeDataLoaders,
    TestGenotypeLodging1_3DataLoaders,
    TestGenotypeLodging1_5DataLoaders,
    TestGenotypeNoise0_02DataLoaders,
    TestGenotypeNoLodgingDataLoaders,
    TestPlotDataLoaders,
    TestPlotLodging1_3DataLoaders,
    TestPlotLodging1_5DataLoaders,
    TestPlotNoise0_02DataLoaders,
    TestPlotNoLodgingDataLoaders,
    TestSiteDataLoaders,
    TestSiteNoLodgingDataLoaders,
    TestUnseenDataLoaders,
    TestUnseenLodging1_3DataLoaders,
    TestUnseenLodging1_5DataLoaders,
    TestUnseenNoise0_02DataLoaders,
    TestUnseenNoLodgingDataLoaders,
    TestYearDataLoaders,
    TestYearNoLodgingDataLoaders,
    TrainDataLoaders_1k,
    TrainDataLoaders_8k,
    TrainDataLoaders_64k,
    TrainDataLoaders_512k,
)

# Synthetic dataset size scaling dataloaders
dataloaders_store(TrainDataLoaders_1k, name="synth_train_1k_dataloaders")
dataloaders_store(TrainDataLoaders_8k, name="synth_train_8k_dataloaders")
dataloaders_store(TrainDataLoaders_64k, name="synth_train_64k_dataloaders")
dataloaders_store(TrainDataLoaders_512k, name="synth_train_512k_dataloaders")

# Shifted synthetic test dataloaders: an optional test set
dataloaders_store(ShiftedSeenDataLoaders, name="synth_shifted_seen_dataloaders")
dataloaders_store(ShiftedGenoDataLoaders, name="synth_shifted_geno_dataloaders")
dataloaders_store(ShiftedEnvDataLoaders, name="synth_shifted_env_dataloaders")
dataloaders_store(ShiftedUnseenDataLoaders, name="synth_shifted_unseen_dataloaders")

# Legacy synthetic test dataloaders (cached pools for consistent evaluation)
dataloaders_store(TestPlotDataLoaders, name="synth_test_plot_dataloaders")
dataloaders_store(TestGenotypeDataLoaders, name="synth_test_genotype_dataloaders")
dataloaders_store(TestSiteDataLoaders, name="synth_test_site_dataloaders")
dataloaders_store(TestYearDataLoaders, name="synth_test_year_dataloaders")
dataloaders_store(TestEnvironmentDataLoaders, name="synth_test_environment_dataloaders")
dataloaders_store(TestUnseenDataLoaders, name="synth_test_unseen_dataloaders")
dataloaders_store(TestAllDataLoaders, name="synth_test_all_dataloaders")

dataloaders_store(
    TestPlotNoLodgingDataLoaders, name="synth_test_plot_no_lodging_dataloaders"
)
dataloaders_store(
    TestGenotypeNoLodgingDataLoaders, name="synth_test_genotype_no_lodging_dataloaders"
)
dataloaders_store(
    TestSiteNoLodgingDataLoaders, name="synth_test_site_no_lodging_dataloaders"
)
dataloaders_store(
    TestYearNoLodgingDataLoaders, name="synth_test_year_no_lodging_dataloaders"
)
dataloaders_store(
    TestEnvironmentNoLodgingDataLoaders,
    name="synth_test_environment_no_lodging_dataloaders",
)
dataloaders_store(
    TestUnseenNoLodgingDataLoaders, name="synth_test_unseen_no_lodging_dataloaders"
)
dataloaders_store(
    TestAllNoLodgingDataLoaders, name="synth_test_all_no_lodging_dataloaders"
)

dataloaders_store(
    TestPlotLodging1_3DataLoaders, name="synth_test_plot_lodging1_3_dataloaders"
)
dataloaders_store(
    TestGenotypeLodging1_3DataLoaders, name="synth_test_genotype_lodging1_3_dataloaders"
)
dataloaders_store(
    TestEnvironmentLodging1_3DataLoaders,
    name="synth_test_environment_lodging1_3_dataloaders",
)
dataloaders_store(
    TestUnseenLodging1_3DataLoaders, name="synth_test_unseen_lodging1_3_dataloaders"
)
dataloaders_store(
    TestPlotLodging1_5DataLoaders, name="synth_test_plot_lodging1_5_dataloaders"
)
dataloaders_store(
    TestGenotypeLodging1_5DataLoaders, name="synth_test_genotype_lodging1_5_dataloaders"
)
dataloaders_store(
    TestEnvironmentLodging1_5DataLoaders,
    name="synth_test_environment_lodging1_5_dataloaders",
)
dataloaders_store(
    TestUnseenLodging1_5DataLoaders, name="synth_test_unseen_lodging1_5_dataloaders"
)
dataloaders_store(
    TestPlotNoise0_02DataLoaders, name="synth_test_plot_noise0_02_dataloaders"
)
dataloaders_store(
    TestGenotypeNoise0_02DataLoaders, name="synth_test_genotype_noise0_02_dataloaders"
)
dataloaders_store(
    TestEnvironmentNoise0_02DataLoaders,
    name="synth_test_environment_noise0_02_dataloaders",
)
dataloaders_store(
    TestUnseenNoise0_02DataLoaders, name="synth_test_unseen_noise0_02_dataloaders"
)

Fip1TrainHeightDataLoadersConfig = make_config(
    train=Fip1TrainHeightDataLoaderConfig, validation=Fip1ValHeightDataLoaderConfig
)
Fip1TestHeightDataLoadersConfig = make_config(
    test_all=Fip1TestAllHeightDataLoaderConfig
)
Fip1TestHeightIndividualDataLoadersConfig = make_config(
    test_plot=Fip1TestPlotHeightDataLoaderConfig,
    test_genotype=Fip1TestGenotypeHeightDataLoaderConfig,
    test_environment=Fip1TestEnvironmentHeightDataLoaderConfig,
    test_genotype_environment=Fip1TestGenotypeEnvironmentHeightDataLoaderConfig,
)

# One dataloaders config per FIP1 test split, mirroring the synthetic
# ``synth_test_*_dataloaders`` entries so prediction results land in a parallel tree.
Fip1TestPlotHeightDataLoadersConfig = make_config(
    test_plot=Fip1TestPlotHeightDataLoaderConfig
)
Fip1TestGenotypeHeightDataLoadersConfig = make_config(
    test_genotype=Fip1TestGenotypeHeightDataLoaderConfig
)
Fip1TestEnvironmentHeightDataLoadersConfig = make_config(
    test_environment=Fip1TestEnvironmentHeightDataLoaderConfig
)
Fip1TestGenotypeEnvironmentHeightDataLoadersConfig = make_config(
    test_genotype_environment=Fip1TestGenotypeEnvironmentHeightDataLoaderConfig
)

dataloader_store(Fip1TrainHeightDataLoaderConfig, name="fip1_train_dataloader")
dataloader_store(
    Fip1BalancedTrainHeightDataLoaderConfig, name="fip1_balanced_train_dataloader"
)
dataloader_store(
    Fip1BalancedTrainHeightInverseFreqDataLoaderConfig,
    name="fip1_balanced_inverse_freq_train_dataloader",
)
dataloader_store(Fip1ValHeightDataLoaderConfig, name="fip1_val_dataloader")
dataloader_store(Fip1AllSplitsHeightDataLoaderConfig, name="fip1_all_splits_dataloader")


Fip1BalancedTrainHeightDataLoadersConfig = make_config(
    train=Fip1BalancedTrainHeightDataLoaderConfig,
    validation=Fip1ValHeightDataLoaderConfig,
)
Fip1BalancedInverseFreqTrainHeightDataLoadersConfig = make_config(
    train=Fip1BalancedTrainHeightInverseFreqDataLoaderConfig,
    validation=Fip1ValHeightDataLoaderConfig,
)

dataloaders_store(Fip1TrainHeightDataLoadersConfig, name="fip1_train_dataloaders")
dataloaders_store(
    Fip1BalancedTrainHeightDataLoadersConfig, name="fip1_balanced_train_dataloaders"
)
dataloaders_store(
    Fip1BalancedInverseFreqTrainHeightDataLoadersConfig,
    name="fip1_balanced_inverse_freq_train_dataloaders",
)
dataloaders_store(Fip1TestHeightDataLoadersConfig, name="fip1_test_dataloaders")
dataloaders_store(
    Fip1TestHeightIndividualDataLoadersConfig, name="fip1_test_individual_dataloaders"
)
dataloaders_store(
    Fip1TestPlotHeightDataLoadersConfig, name="fip1_test_plot_dataloaders"
)
dataloaders_store(
    Fip1TestGenotypeHeightDataLoadersConfig, name="fip1_test_genotype_dataloaders"
)
dataloaders_store(
    Fip1TestEnvironmentHeightDataLoadersConfig, name="fip1_test_environment_dataloaders"
)
dataloaders_store(
    Fip1TestGenotypeEnvironmentHeightDataLoadersConfig,
    name="fip1_test_genotype_environment_dataloaders",
)


# Combined dataloaders
dataloaders_store(CombinedDataLoadersConfig, name="combined_train_dataloaders")
dataloaders_store(
    CombinedSyntheticValDataLoadersConfig, name="combined_train_synth_val_dataloaders"
)


# Weighted yearsite dataloaders
dataloaders_store(
    WeightedYearsiteUniformDataLoadersConfig,
    name="weighted_yearsite_uniform_dataloaders",
)
dataloaders_store(
    WeightedYearsiteInverseFreqDataLoadersConfig,
    name="weighted_yearsite_inverse_freq_dataloaders",
)
