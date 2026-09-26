from hydra_zen import make_custom_builds_fn

builds = make_custom_builds_fn()
populate_builds = make_custom_builds_fn(populate_full_signature=True)
partial_builds = make_custom_builds_fn(zen_partial=True)
partial_populate_builds = make_custom_builds_fn(
    populate_full_signature=True, zen_partial=True
)
