import math
import pytest

# Python engine
from simulator import create_start_conditions as PyCreate

# C engine may be in simulator.simulator.cenv or simulator.cenv
try:
    from simulator.simulator.cenv import create_start_conditions as CCreate  # type: ignore
    CY = True
except Exception:
    try:
        from simulator.cenv import create_start_conditions as CCreate  # type: ignore
        CY = True
    except Exception:
        CCreate = None
        CY = False


def step_n(gameplay, n=1):
    for _ in range(n):
        gameplay.step(1)


@pytest.mark.skipif(not CY, reason="Cython engine not available")
def test_hold_single_shot_and_slowdown_parity():
    py = PyCreate()
    cy = CCreate()

    # Position red player near ball and hold kick
    red_py = py.Pa.D[1]
    red_cy = cy.Pa.D[1]

    # Place ball right of red player
    py.wa.K[0].a.x = -100
    py.wa.K[0].a.y = 0
    red_py.F.a.x = -115
    red_py.F.a.y = 0

    cy.wa.K[0].a.x = -100
    cy.wa.K[0].a.y = 0
    red_cy.F.a.x = -115
    red_cy.F.a.y = 0

    # Hold kick (bit 16)
    red_py.mb |= 16
    red_cy.mb |= 16

    # Step once: both should impart a single impulse to ball
    step_n(py, 1)
    step_n(cy, 1)

    vx_py = py.wa.K[0].M.x
    vx_cy = cy.wa.K[0].M.x

    assert vx_py > 0
    assert vx_cy > 0
    # Same direction and close magnitude
    assert math.isclose(vx_py, vx_cy, rel_tol=1e-5, abs_tol=1e-5)

    # Keep holding and step many times; ball speed should not keep increasing from repeated kicks
    old_py = vx_py
    old_cy = vx_cy
    step_n(py, 5)
    step_n(cy, 5)

    assert py.wa.K[0].M.x <= old_py * 1.05  # allow tiny numerical drift
    assert cy.wa.K[0].M.x <= old_cy * 1.05

    # Check slowdown on player while holding: player velocity grows slower than free running
    # First, measure with holding state already active
    v_hold_py = math.hypot(red_py.F.M.x, red_py.F.M.y)
    v_hold_cy = math.hypot(red_cy.F.M.x, red_cy.F.M.y)

    # Reset to same pose and compare a step with hold vs without hold
    py = PyCreate(); cy = CCreate()
    red_py = py.Pa.D[1]; red_cy = cy.Pa.D[1]
    red_py.mb |= 16; red_cy.mb |= 16
    # Move up-right (bits 2 and 4 unset -> We'll use 2 and 8 mapping depends; here use right (2) and up (4))
    red_py.mb |= 2 | 4
    red_cy.mb |= 2 | 4
    step_n(py, 1); step_n(cy, 1)
    sp_hold_py = math.hypot(red_py.F.M.x, red_py.F.M.y)
    sp_hold_cy = math.hypot(red_cy.F.M.x, red_cy.F.M.y)

    # Same but without hold
    py2 = PyCreate(); cy2 = CCreate()
    red_py2 = py2.Pa.D[1]; red_cy2 = cy2.Pa.D[1]
    red_py2.mb |= 2 | 4
    red_cy2.mb |= 2 | 4
    step_n(py2, 1); step_n(cy2, 1)
    sp_free_py = math.hypot(red_py2.F.M.x, red_py2.F.M.y)
    sp_free_cy = math.hypot(red_cy2.F.M.x, red_cy2.F.M.y)

    assert sp_hold_py < sp_free_py
    assert sp_hold_cy < sp_free_cy
    # And the held speeds are similar across engines
    assert math.isclose(sp_hold_py, sp_hold_cy, rel_tol=1e-5, abs_tol=1e-5)
