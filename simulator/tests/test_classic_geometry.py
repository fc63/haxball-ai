import math

from simulator import create_start_conditions, Vector


def test_classic_dimensions_and_constants():
    gp = create_start_conditions()
    field = gp.U
    phys = gp.wa

    # Field dimensions (Classic)
    assert field.Ed == 370  # half width
    assert field.Dd == 170  # half height
    assert field.Yc == 75   # center circle radius

    # Object radii
    ball, post1, post2, post3, post4, red_obj, blue_obj = phys.K
    assert ball.la == 10
    assert post1.la == 8 and post2.la == 8 and post3.la == 8 and post4.la == 8
    assert red_obj.la == 15 and blue_obj.la == 15

    # Physics constants matching JS defaults
    zb = field.Rd
    assert math.isclose(zb.me, 0.1, rel_tol=1e-6)
    assert math.isclose(zb.Be, 0.07, rel_tol=1e-6)
    assert math.isclose(zb.Ce, 0.96, rel_tol=1e-6)
    assert zb.Kd == 5

    # Damping: ball/posts vs players
    assert math.isclose(ball.Ba, 0.99, rel_tol=1e-6)
    assert math.isclose(red_obj.Ba, 0.96, rel_tol=1e-6)
    assert math.isclose(blue_obj.Ba, 0.96, rel_tol=1e-6)

    # Planes: outer planes (0.1 bCoef, mask 63) and ballArea planes (1.0 bCoef, mask 1)
    planes = field.ha
    outer = [p for p in planes if p.l == 0.1]
    ball_only = [p for p in planes if p.l == 1]
    assert len(outer) == 4
    assert len(ball_only) == 2
    assert all(p.h == 63 for p in outer)
    assert all(p.h == 1 for p in ball_only)


def test_hold_and_kick_impulse_single_shot():
    gp = create_start_conditions()
    phys = gp.wa
    ball = phys.K[0]
    blue = phys.K[6]
    blue_user = gp.Pa.D[2]

    # Place ball within kick range in front of blue
    blue.a.x = 0
    blue.a.y = 0
    ball.a.x = blue.a.x + (blue.la + ball.la - 2)  # within radii+(-2) => inside 4px window
    ball.a.y = 0
    ball.M.x = 0
    ball.M.y = 0

    # Hold kick and step; expect one impulse and latch set
    blue_user.mb = 16
    gp.step(1)
    v1 = math.hypot(ball.M.x, ball.M.y)
    assert v1 > 0
    assert getattr(blue_user, 'kicked_this_hold', False) is True

    # Keep holding and step again; velocity should not jump a second time (no re-impulse)
    prev = v1
    gp.step(1)
    v2 = math.hypot(ball.M.x, ball.M.y)
    # Allow minor numeric differences due to damping/contacts; ensure not a big step jump
    assert v2 <= prev * 1.05

    # Release and hold again should allow another impulse
    blue_user.mb = 0
    gp.step(1)
    # Reposition ball within range again (simulate being close for a new kick)
    ball.a.x = blue.a.x + (blue.la + ball.la - 2)
    ball.a.y = 0
    ball.M.x = 0
    ball.M.y = 0
    blue_user.mb = 16
    gp.step(1)
    v3 = math.hypot(ball.M.x, ball.M.y)
    assert v3 > v2
