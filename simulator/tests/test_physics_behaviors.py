import math

from simulator import create_start_conditions


def step_n(gp, n):
    for _ in range(n):
        gp.step(1)


def test_kickoff_transition_on_ball_move():
    gp = create_start_conditions()
    ball = gp.wa.K[0]
    # start in kickoff
    assert gp.zb == 0
    # Give ball some velocity and step once -> should start play and advance time
    ball.M.x = 5
    ball.M.y = 0
    t0 = gp.Ac
    gp.step(1)
    assert gp.zb == 1
    # Time advances starting the next tick after kickoff becomes active
    gp.step(1)
    assert gp.Ac > t0


def test_ballarea_affects_only_ball():
    gp = create_start_conditions()
    ball = gp.wa.K[0]
    blue = gp.wa.K[6]
    # exit kickoff
    ball.M.x = 0.01
    gp.step(1)
    # Place near lower boundary and push downward (positive y)
    blue.a.x, blue.a.y = 0, 160
    blue.M.x, blue.M.y = 0, 5
    ball.a.x, ball.a.y = 0, 160
    ball.M.x, ball.M.y = 0, 5
    # Run a bit
    step_n(gp, 60)
    # Ball should be constrained within ±170 (center), allowing small tolerance for numeric effects
    assert ball.a.y <= 170 + 1e-3
    # Player can exceed 170 (until outer 200 constraint)
    assert blue.a.y > 170 - 1e-3


def test_post_collision_separation():
    gp = create_start_conditions()
    ball = gp.wa.K[0]
    post = gp.wa.K[1]  # (370, -64), radius 8
    # exit kickoff
    ball.M.x = 0.01
    gp.step(1)

    # Aim ball toward the post
    ball.a.x, ball.a.y = 350, -64
    ball.M.x, ball.M.y = 30, 0
    # step through collision
    step_n(gp, 60)
    dist = math.hypot(ball.a.x - post.a.x, ball.a.y - post.a.y)
    assert dist + 1e-6 >= (ball.la + post.la)


def test_hold_slowdown_reduces_steady_speed():
    gp = create_start_conditions()
    blue = gp.wa.K[6]
    blue_user = gp.Pa.D[2]
    # exit kickoff
    gp.wa.K[0].M.x = 0.01
    gp.step(1)

    # Baseline: move right without hold
    blue.a.x, blue.a.y = 0, 0
    blue.M.x, blue.M.y = 0, 0
    blue_user.mb = 8  # right
    step_n(gp, 120)
    v_no_hold = math.hypot(blue.M.x, blue.M.y)

    # With hold: keep same direction and hold kick
    blue.a.x, blue.a.y = 0, 0
    blue.M.x, blue.M.y = 0, 0
    blue_user.mb = 8 | 16  # right + hold
    step_n(gp, 120)
    v_hold = math.hypot(blue.M.x, blue.M.y)

    assert v_hold < v_no_hold


def test_outer_wall_reflection_and_damping():
    gp = create_start_conditions()
    ball = gp.wa.K[0]
    # exit kickoff
    ball.M.x = 0.01
    gp.step(1)

    # Send ball to right outer wall
    ball.a.x, ball.a.y = 410, 0
    ball.M.x, ball.M.y = 40, 0

    # After some steps, it should be within boundary and have lost energy; velocity may be small/negative
    step_n(gp, 30)
    vx_after = ball.M.x
    # Either bounced back (negative) or damped to ~0
    assert vx_after <= 1e-6
    # Inside or at outer boundary (<= 420 considering radius 10)
    assert ball.a.x <= 420
    # And magnitude should be smaller than initial (damping/restitution)
    assert abs(vx_after) < 40
