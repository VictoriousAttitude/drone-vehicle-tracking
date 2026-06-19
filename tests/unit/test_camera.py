from drone_vehicle_tracking.geo.camera import MAVIC_3T_WIDE


def test_gsd_scales_linearly_with_altitude() -> None:
    gsd_low = MAVIC_3T_WIDE.gsd(60.0)
    gsd_high = MAVIC_3T_WIDE.gsd(120.0)
    assert gsd_high == 2 * gsd_low


def test_gsd_is_centimetre_scale_at_typical_altitude() -> None:
    # ~7.6 cm/px at 102 m for the Mavic 3T wide camera.
    gsd = MAVIC_3T_WIDE.gsd(102.0)
    assert 0.05 < gsd < 0.10
