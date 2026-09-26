from npnf.data.synthetic.height.genotype import GenotypeParams
from npnf.models.configs.utils import model_store, populate_builds
from npnf.models.neural_process import (
    ACNP,
    ANP,
    CNP,
    LNP,
    ANP_NF_Posterior,
    ANP_NF_Prior,
    ANP_NF_Prior_Posterior,
    LNP_NF_Posterior,
    LNP_NF_Prior,
    LNP_NF_Prior_Posterior,
)

# Synthetic-specific configs: marker dim derived from current genotype schema
SYNTH_MARKERS_INPUT_DIM = len(GenotypeParams.field_names())
SYNTH_MARKERS_HIDDEN_CHANNELS = [64, 64]

# FIP1-specific configs: SNP marker dimensions
FIP_MARKERS_INPUT_DIM = 18845
FIP_MARKERS_HIDDEN_CHANNELS = [1024, 1024]

# Base configs (synthetic marker dims)
CNP_Config = populate_builds(
    CNP,
    markers_input_dim=SYNTH_MARKERS_INPUT_DIM,
    markers_hidden_channels=SYNTH_MARKERS_HIDDEN_CHANNELS,
)
ACNP_Config = populate_builds(
    ACNP,
    markers_input_dim=SYNTH_MARKERS_INPUT_DIM,
    markers_hidden_channels=SYNTH_MARKERS_HIDDEN_CHANNELS,
)
ANP_Config = populate_builds(
    ANP,
    markers_input_dim=SYNTH_MARKERS_INPUT_DIM,
    markers_hidden_channels=SYNTH_MARKERS_HIDDEN_CHANNELS,
)
LNP_Config = populate_builds(
    LNP,
    markers_input_dim=SYNTH_MARKERS_INPUT_DIM,
    markers_hidden_channels=SYNTH_MARKERS_HIDDEN_CHANNELS,
)
ANP_NF_Prior_Config = populate_builds(
    ANP_NF_Prior,
    markers_input_dim=SYNTH_MARKERS_INPUT_DIM,
    markers_hidden_channels=SYNTH_MARKERS_HIDDEN_CHANNELS,
)
LNP_NF_Prior_Config = populate_builds(
    LNP_NF_Prior,
    markers_input_dim=SYNTH_MARKERS_INPUT_DIM,
    markers_hidden_channels=SYNTH_MARKERS_HIDDEN_CHANNELS,
)
ANP_NF_Posterior_Config = populate_builds(
    ANP_NF_Posterior,
    markers_input_dim=SYNTH_MARKERS_INPUT_DIM,
    markers_hidden_channels=SYNTH_MARKERS_HIDDEN_CHANNELS,
)
LNP_NF_Posterior_Config = populate_builds(
    LNP_NF_Posterior,
    markers_input_dim=SYNTH_MARKERS_INPUT_DIM,
    markers_hidden_channels=SYNTH_MARKERS_HIDDEN_CHANNELS,
)
ANP_NF_Prior_Posterior_Config = populate_builds(
    ANP_NF_Prior_Posterior,
    markers_input_dim=SYNTH_MARKERS_INPUT_DIM,
    markers_hidden_channels=SYNTH_MARKERS_HIDDEN_CHANNELS,
)
LNP_NF_Prior_Posterior_Config = populate_builds(
    LNP_NF_Prior_Posterior,
    markers_input_dim=SYNTH_MARKERS_INPUT_DIM,
    markers_hidden_channels=SYNTH_MARKERS_HIDDEN_CHANNELS,
)

# FIP1 configs (SNP marker dims)
CNP_FIP_Config = populate_builds(
    CNP,
    markers_input_dim=FIP_MARKERS_INPUT_DIM,
    markers_hidden_channels=FIP_MARKERS_HIDDEN_CHANNELS,
)
ACNP_FIP_Config = populate_builds(
    ACNP,
    markers_input_dim=FIP_MARKERS_INPUT_DIM,
    markers_hidden_channels=FIP_MARKERS_HIDDEN_CHANNELS,
)
ANP_FIP_Config = populate_builds(
    ANP,
    markers_input_dim=FIP_MARKERS_INPUT_DIM,
    markers_hidden_channels=FIP_MARKERS_HIDDEN_CHANNELS,
)
LNP_FIP_Config = populate_builds(
    LNP,
    markers_input_dim=FIP_MARKERS_INPUT_DIM,
    markers_hidden_channels=FIP_MARKERS_HIDDEN_CHANNELS,
)
ANP_NF_Prior_FIP_Config = populate_builds(
    ANP_NF_Prior,
    markers_input_dim=FIP_MARKERS_INPUT_DIM,
    markers_hidden_channels=FIP_MARKERS_HIDDEN_CHANNELS,
)
LNP_NF_Prior_FIP_Config = populate_builds(
    LNP_NF_Prior,
    markers_input_dim=FIP_MARKERS_INPUT_DIM,
    markers_hidden_channels=FIP_MARKERS_HIDDEN_CHANNELS,
)
ANP_NF_Posterior_FIP_Config = populate_builds(
    ANP_NF_Posterior,
    markers_input_dim=FIP_MARKERS_INPUT_DIM,
    markers_hidden_channels=FIP_MARKERS_HIDDEN_CHANNELS,
)
LNP_NF_Posterior_FIP_Config = populate_builds(
    LNP_NF_Posterior,
    markers_input_dim=FIP_MARKERS_INPUT_DIM,
    markers_hidden_channels=FIP_MARKERS_HIDDEN_CHANNELS,
)
ANP_NF_Prior_Posterior_FIP_Config = populate_builds(
    ANP_NF_Prior_Posterior,
    markers_input_dim=FIP_MARKERS_INPUT_DIM,
    markers_hidden_channels=FIP_MARKERS_HIDDEN_CHANNELS,
)
LNP_NF_Prior_Posterior_FIP_Config = populate_builds(
    LNP_NF_Prior_Posterior,
    markers_input_dim=FIP_MARKERS_INPUT_DIM,
    markers_hidden_channels=FIP_MARKERS_HIDDEN_CHANNELS,
)

model_store(CNP_Config, name="CNP")
model_store(ACNP_Config, name="ACNP")
model_store(ANP_Config, name="ANP")
model_store(LNP_Config, name="LNP")
model_store(ANP_NF_Prior_Config, name="ANP-NF-Prior")
model_store(LNP_NF_Prior_Config, name="LNP-NF-Prior")
model_store(ANP_NF_Posterior_Config, name="ANP-NF-Posterior")
model_store(LNP_NF_Posterior_Config, name="LNP-NF-Posterior")
model_store(ANP_NF_Prior_Posterior_Config, name="ANP-NF-Prior-Posterior")
model_store(LNP_NF_Prior_Posterior_Config, name="LNP-NF-Prior-Posterior")
model_store(CNP_FIP_Config, name="CNP-FIP")
model_store(ACNP_FIP_Config, name="ACNP-FIP")
model_store(ANP_FIP_Config, name="ANP-FIP")
model_store(LNP_FIP_Config, name="LNP-FIP")
model_store(ANP_NF_Prior_FIP_Config, name="ANP-NF-Prior-FIP")
model_store(LNP_NF_Prior_FIP_Config, name="LNP-NF-Prior-FIP")
model_store(ANP_NF_Posterior_FIP_Config, name="ANP-NF-Posterior-FIP")
model_store(LNP_NF_Posterior_FIP_Config, name="LNP-NF-Posterior-FIP")
model_store(ANP_NF_Prior_Posterior_FIP_Config, name="ANP-NF-Prior-Posterior-FIP")
model_store(LNP_NF_Prior_Posterior_FIP_Config, name="LNP-NF-Prior-Posterior-FIP")
