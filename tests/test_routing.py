from graph_design.config import DesignConfig
from graph_design.routing import DensityRoutingPolicy


def test_routing_window_boundaries():
    policy = DensityRoutingPolicy(
        DesignConfig(rho_very_low=0.05, rho_low_mix_end=0.10, rho_low=0.5, rho_high=0.75)
    )

    assert policy.route(0.01).label == "envelope_only"
    assert policy.route(0.049).label == "envelope_only"
    assert policy.route(0.05).label == "both"
    assert policy.route(0.099).label == "both"
    assert policy.route(0.10).label == "both"
    assert policy.route(0.101).label == "cayley_only"
    assert policy.route(0.49).label == "cayley_only"
    assert policy.route(0.50).label == "both"
    assert policy.route(0.75).label == "both"
    assert policy.route(0.751).label == "envelope_only"
