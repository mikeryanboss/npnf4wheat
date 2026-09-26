"""Dataloader configs for SyntheticDataset."""

from hydra_zen import make_config

from npnf.data.configs.dataloaders.dataloader import (
    TestFip1HeightsDataloaderConfig,
    TrainFip1HeightsDataloaderConfig,
    ValFip1HeightsDataloaderConfig,
)
from npnf.data.configs.datasets.shifted import (
    ShiftedConfig_Env,
    ShiftedConfig_Geno,
    ShiftedConfig_Seen,
    ShiftedConfig_Unseen,
)
from npnf.data.configs.datasets.synthetic import (
    TestConfig_Environment,
    TestConfig_Environment_Lodging1_3,
    TestConfig_Environment_Lodging1_5,
    TestConfig_Environment_Noise0_02,
    TestConfig_Environment_NoLodging,
    TestConfig_Genotype,
    TestConfig_Genotype_Lodging1_3,
    TestConfig_Genotype_Lodging1_5,
    TestConfig_Genotype_Noise0_02,
    TestConfig_Genotype_NoLodging,
    TestConfig_Plot,
    TestConfig_Plot_Lodging1_3,
    TestConfig_Plot_Lodging1_5,
    TestConfig_Plot_Noise0_02,
    TestConfig_Plot_NoLodging,
    TestConfig_Site,
    TestConfig_Site_NoLodging,
    TestConfig_Unseen,
    TestConfig_Unseen_Lodging1_3,
    TestConfig_Unseen_Lodging1_5,
    TestConfig_Unseen_Noise0_02,
    TestConfig_Unseen_NoLodging,
    TestConfig_Year,
    TestConfig_Year_NoLodging,
    TrainConfig_1k,
    TrainConfig_8k,
    TrainConfig_64k,
    TrainConfig_512k,
    ValConfig,
)

# =============================================================================
# Validation Dataloader (shared across all training sizes)
# =============================================================================

ValDataLoader = ValFip1HeightsDataloaderConfig(dataset=ValConfig, drop_last=True)


# =============================================================================
# Training Dataloaders (standard weather data)
# =============================================================================

TrainDataLoader_1k = TrainFip1HeightsDataloaderConfig(dataset=TrainConfig_1k)
TrainDataLoader_8k = TrainFip1HeightsDataloaderConfig(dataset=TrainConfig_8k)
TrainDataLoader_64k = TrainFip1HeightsDataloaderConfig(dataset=TrainConfig_64k)
TrainDataLoader_512k = TrainFip1HeightsDataloaderConfig(dataset=TrainConfig_512k)


# =============================================================================
# Combined Train + Val Dataloader Configs (standard weather data)
# =============================================================================

TrainDataLoaders_1k = make_config(train=TrainDataLoader_1k, validation=ValDataLoader)
TrainDataLoaders_8k = make_config(train=TrainDataLoader_8k, validation=ValDataLoader)
TrainDataLoaders_64k = make_config(train=TrainDataLoader_64k, validation=ValDataLoader)
TrainDataLoaders_512k = make_config(
    train=TrainDataLoader_512k, validation=ValDataLoader
)


# =============================================================================
# Test Dataloaders (synthetic test splits with cached pools)
# =============================================================================

TestPlotDataLoader = TestFip1HeightsDataloaderConfig(
    dataset=TestConfig_Plot, drop_last=True
)
TestGenotypeDataLoader = TestFip1HeightsDataloaderConfig(
    dataset=TestConfig_Genotype, drop_last=True
)
TestSiteDataLoader = TestFip1HeightsDataloaderConfig(
    dataset=TestConfig_Site, drop_last=True
)
TestYearDataLoader = TestFip1HeightsDataloaderConfig(
    dataset=TestConfig_Year, drop_last=True
)
TestEnvironmentDataLoader = TestFip1HeightsDataloaderConfig(
    dataset=TestConfig_Environment, drop_last=True
)
TestUnseenDataLoader = TestFip1HeightsDataloaderConfig(
    dataset=TestConfig_Unseen, drop_last=True
)

TestPlotNoLodgingDataLoader = TestFip1HeightsDataloaderConfig(
    dataset=TestConfig_Plot_NoLodging, drop_last=True
)
TestGenotypeNoLodgingDataLoader = TestFip1HeightsDataloaderConfig(
    dataset=TestConfig_Genotype_NoLodging, drop_last=True
)
TestSiteNoLodgingDataLoader = TestFip1HeightsDataloaderConfig(
    dataset=TestConfig_Site_NoLodging, drop_last=True
)
TestYearNoLodgingDataLoader = TestFip1HeightsDataloaderConfig(
    dataset=TestConfig_Year_NoLodging, drop_last=True
)
TestEnvironmentNoLodgingDataLoader = TestFip1HeightsDataloaderConfig(
    dataset=TestConfig_Environment_NoLodging, drop_last=True
)
TestUnseenNoLodgingDataLoader = TestFip1HeightsDataloaderConfig(
    dataset=TestConfig_Unseen_NoLodging, drop_last=True
)


# =============================================================================
# Combined Test Dataloader Configs
# =============================================================================

TestPlotDataLoaders = make_config(test_plot=TestPlotDataLoader)
TestGenotypeDataLoaders = make_config(test_genotype=TestGenotypeDataLoader)
TestSiteDataLoaders = make_config(test_site=TestSiteDataLoader)
TestYearDataLoaders = make_config(test_year=TestYearDataLoader)
TestEnvironmentDataLoaders = make_config(test_environment=TestEnvironmentDataLoader)
TestUnseenDataLoaders = make_config(test_unseen=TestUnseenDataLoader)

TestPlotNoLodgingDataLoaders = make_config(test_plot=TestPlotNoLodgingDataLoader)
TestGenotypeNoLodgingDataLoaders = make_config(
    test_genotype=TestGenotypeNoLodgingDataLoader
)
TestSiteNoLodgingDataLoaders = make_config(test_site=TestSiteNoLodgingDataLoader)
TestYearNoLodgingDataLoaders = make_config(test_year=TestYearNoLodgingDataLoader)
TestEnvironmentNoLodgingDataLoaders = make_config(
    test_environment=TestEnvironmentNoLodgingDataLoader
)
TestUnseenNoLodgingDataLoaders = make_config(test_unseen=TestUnseenNoLodgingDataLoader)

# All test splits combined
TestAllDataLoaders = make_config(
    test_plot=TestPlotDataLoader,
    test_genotype=TestGenotypeDataLoader,
    test_site=TestSiteDataLoader,
    test_year=TestYearDataLoader,
    test_environment=TestEnvironmentDataLoader,
    test_unseen=TestUnseenDataLoader,
)

TestAllNoLodgingDataLoaders = make_config(
    test_plot=TestPlotNoLodgingDataLoader,
    test_genotype=TestGenotypeNoLodgingDataLoader,
    test_site=TestSiteNoLodgingDataLoader,
    test_year=TestYearNoLodgingDataLoader,
    test_environment=TestEnvironmentNoLodgingDataLoader,
    test_unseen=TestUnseenNoLodgingDataLoader,
)


# =============================================================================
# Shifted Test Dataloaders (optional test set, frozen bundle)
# =============================================================================
# No drop_last: 32,000 conditions per split are not a multiple of every prediction
# batch size (e.g. 384), and dropping the remainder would lose test conditions.

ShiftedSeenDataLoaders = make_config(
    test_seen=TestFip1HeightsDataloaderConfig(dataset=ShiftedConfig_Seen)
)
ShiftedGenoDataLoaders = make_config(
    test_geno=TestFip1HeightsDataloaderConfig(dataset=ShiftedConfig_Geno)
)
ShiftedEnvDataLoaders = make_config(
    test_env=TestFip1HeightsDataloaderConfig(dataset=ShiftedConfig_Env)
)
ShiftedUnseenDataLoaders = make_config(
    test_unseen=TestFip1HeightsDataloaderConfig(dataset=ShiftedConfig_Unseen)
)


TestPlotLodging1_3DataLoaders = make_config(
    test_plot=TestFip1HeightsDataloaderConfig(
        dataset=TestConfig_Plot_Lodging1_3, drop_last=True
    )
)
TestGenotypeLodging1_3DataLoaders = make_config(
    test_genotype=TestFip1HeightsDataloaderConfig(
        dataset=TestConfig_Genotype_Lodging1_3, drop_last=True
    )
)
TestEnvironmentLodging1_3DataLoaders = make_config(
    test_environment=TestFip1HeightsDataloaderConfig(
        dataset=TestConfig_Environment_Lodging1_3, drop_last=True
    )
)
TestUnseenLodging1_3DataLoaders = make_config(
    test_unseen=TestFip1HeightsDataloaderConfig(
        dataset=TestConfig_Unseen_Lodging1_3, drop_last=True
    )
)
TestPlotLodging1_5DataLoaders = make_config(
    test_plot=TestFip1HeightsDataloaderConfig(
        dataset=TestConfig_Plot_Lodging1_5, drop_last=True
    )
)
TestGenotypeLodging1_5DataLoaders = make_config(
    test_genotype=TestFip1HeightsDataloaderConfig(
        dataset=TestConfig_Genotype_Lodging1_5, drop_last=True
    )
)
TestEnvironmentLodging1_5DataLoaders = make_config(
    test_environment=TestFip1HeightsDataloaderConfig(
        dataset=TestConfig_Environment_Lodging1_5, drop_last=True
    )
)
TestUnseenLodging1_5DataLoaders = make_config(
    test_unseen=TestFip1HeightsDataloaderConfig(
        dataset=TestConfig_Unseen_Lodging1_5, drop_last=True
    )
)
TestPlotNoise0_02DataLoaders = make_config(
    test_plot=TestFip1HeightsDataloaderConfig(
        dataset=TestConfig_Plot_Noise0_02, drop_last=True
    )
)
TestGenotypeNoise0_02DataLoaders = make_config(
    test_genotype=TestFip1HeightsDataloaderConfig(
        dataset=TestConfig_Genotype_Noise0_02, drop_last=True
    )
)
TestEnvironmentNoise0_02DataLoaders = make_config(
    test_environment=TestFip1HeightsDataloaderConfig(
        dataset=TestConfig_Environment_Noise0_02, drop_last=True
    )
)
TestUnseenNoise0_02DataLoaders = make_config(
    test_unseen=TestFip1HeightsDataloaderConfig(
        dataset=TestConfig_Unseen_Noise0_02, drop_last=True
    )
)
