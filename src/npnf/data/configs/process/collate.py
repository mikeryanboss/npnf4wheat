from npnf.configs.utils import partial_populate_builds
from npnf.data.process.collate import collate_fn_fip1_heights

CollateFip1HeightsFnConfig = partial_populate_builds(collate_fn_fip1_heights)
