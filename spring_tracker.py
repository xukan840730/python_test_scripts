import math


def track_common(vel, curr_val, x_goal, spring_const, mass, in_delta_time1):
    max_step_time = math.sqrt(0.5 / spring_const)
    damping_coef = math.sqrt(4.0 * mass * spring_const)

    new_val = curr_val
    new_vel = vel
    in_delta_time = in_delta_time1
    while in_delta_time > 0.0:
        dt = min(max_step_time, in_delta_time)
        force = -spring_const * (new_val - x_goal) - damping_coef * new_vel
        acc = force / mass
        new_val += (new_vel + acc / 2.0 * dt) * dt
        new_vel += acc * dt
        in_delta_time = max(in_delta_time - dt, 0.0)

    return new_val, new_vel
