import numpy as np

from experiments.sampling_assisted_r101 import (
    GRID_H,
    GRID_W,
    IMAGE_SIZE,
    MLSP_SPARSE_MEAN,
    SUPPORT_COUNT,
    add_sparse_features,
    build_input_tensor,
    c_from_support,
    free_space_grid,
    frequency_fourier,
    point_grid_locations,
    rssi_to_pathloss_proxy,
    sample_rt_values,
    support_table,
    wall_floor_plan_image,
)
from experiments.baselines import load_samples


def test_free_space_grid_shape_and_rooms():
    free = free_space_grid()
    assert free.shape == (GRID_H, GRID_W)
    assert free[0, 0] == 1
    assert free[8, 10] == 1
    assert free[10, 28] == 1
    assert free[26, 25] == 1
    assert free[20, 5] == 0


def test_wall_floor_plan_is_sparse_boundary_not_occupancy():
    wall_1 = wall_floor_plan_image(1)
    wall_2 = wall_floor_plan_image(2)
    assert wall_1.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert wall_2.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert wall_1.max() == 1
    assert wall_1.min() == 0
    assert wall_1.mean() < 0.08
    assert wall_2.mean() > wall_1.mean()
    assert wall_2.mean() < 0.15


def test_frequency_encoding_is_bounded_and_frequency_dependent():
    a = np.array(frequency_fourier(2417.0))
    b = np.array(frequency_fourier(2462.0))
    assert a.shape == (4,)
    assert np.all(a >= -1.0)
    assert np.all(a <= 1.0)
    assert not np.allclose(a, b)


def test_rssi_to_pathloss_proxy_sign_and_support_centering():
    rssi = np.array([-40.0, -50.0, -60.0])
    c = MLSP_SPARSE_MEAN + rssi.mean()
    pl = rssi_to_pathloss_proxy(rssi, c)
    assert np.isclose(pl.mean(), MLSP_SPARSE_MEAN)
    assert pl[0] < pl[-1]


def test_support_query_split_and_sparse_features():
    samples = load_samples()
    supports = support_table(samples, SUPPORT_COUNT)
    assert int(supports["is_support"].sum()) == 20 * SUPPORT_COUNT
    assert (supports.groupby("setup")["is_support"].sum() == SUPPORT_COUNT).all()

    query = add_sparse_features(samples, supports)
    assert len(query) == len(samples)
    assert int(query["is_support"].sum()) == 20 * SUPPORT_COUNT
    for col in ["nearest_support_rssi", "idw_support_rssi", "nearest_support_dist_m"]:
        assert np.isfinite(query[col]).all()


def test_build_input_tensor_channels_are_well_formed():
    samples = load_samples()
    supports = support_table(samples, SUPPORT_COUNT)
    merged = samples.merge(supports, on=["sample_id", "setup", "point"])
    setup_support = merged[(merged["setup"] == 1) & (merged["is_support"] == 1)]
    c_dbm = c_from_support(setup_support)
    tensor, c_dbm, sparse_grid = build_input_tensor(
        setup=1,
        samples=samples,
        setup_support=setup_support,
        point_locations=point_grid_locations(),
        freq_mhz=2462.0,
        wall_thickness_px=1,
        c_dbm=c_dbm,
    )
    assert tensor.shape == (11, IMAGE_SIZE, IMAGE_SIZE)
    assert np.isfinite(tensor).all()
    assert np.count_nonzero(sparse_grid) == SUPPORT_COUNT
    assert np.count_nonzero(tensor[10]) > 0
    assert tensor[8].max() == 1.0
    assert tensor[8].min() == 0.0
    assert tensor[9].mean() < 0.08
    assert np.isclose(c_dbm, c_from_support(setup_support))


def test_imputed_rt_values_populate_wall_pixels_only():
    samples = load_samples()
    supports = support_table(samples, SUPPORT_COUNT)
    merged = samples.merge(supports, on=["sample_id", "setup", "point"])
    setup_support = merged[(merged["setup"] == 1) & (merged["is_support"] == 1)]
    c_dbm = c_from_support(setup_support)
    tensor, _, _ = build_input_tensor(
        setup=1,
        samples=samples,
        setup_support=setup_support,
        point_locations=point_grid_locations(),
        freq_mhz=2462.0,
        wall_thickness_px=1,
        c_dbm=c_dbm,
        rt_values=sample_rt_values(fold=0, setup=1, draw=0),
    )
    wall = tensor[9] > 0
    assert np.count_nonzero(tensor[0]) == np.count_nonzero(wall)
    assert np.count_nonzero(tensor[1]) == np.count_nonzero(wall)
    assert np.all(tensor[0][~wall] == 0)
    assert np.all(tensor[1][~wall] == 0)
    assert np.isfinite(tensor[0][wall]).all()
    assert np.isfinite(tensor[1][wall]).all()


def test_sampled_rt_values_are_positive_before_normalization():
    for fold in range(3):
        for setup in (1, 7, 20):
            for draw in range(8):
                reflectance, transmittance = sample_rt_values(fold=fold, setup=setup, draw=draw)
                assert reflectance > 0
                assert transmittance > 0
