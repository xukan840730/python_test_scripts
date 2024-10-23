from euclid3 import *
import math
import matplotlib.pyplot as plt
import copy


def rotate_vector_2d(vector, theta_rad):
    x, y = vector
    # Calculate new coordinates after rotation
    new_x = x * math.cos(theta_rad) - y * math.sin(theta_rad)
    new_y = x * math.sin(theta_rad) + y * math.cos(theta_rad)
    return Vector2(new_x, new_y)


def generate_arc_trajectory(input_angles, input_time_stamps, speed):
    assert len(input_angles) == len(input_time_stamps)

    input_vecs = []
    for i in range(len(input_angles)):
        input_angle_rad = math.radians(input_angles[i])
        x = math.cos(input_angle_rad)
        y = math.sin(input_angle_rad)
        input_vec = Vector2(x, y)
        input_vecs.append(input_vec)

    dt_sum = 0.0
    d_angle_rad = 0.0
    for i in range(1, len(input_angles)):
        assert input_time_stamps[i] >= input_time_stamps[i - 1]
        dt_sum += input_time_stamps[i] - input_time_stamps[i - 1]
        d_angle_rad += math.radians(input_angles[i]) - math.radians(input_angles[i - 1])

    assert dt_sum > 0.0
    v_angular = d_angle_rad / dt_sum

    radius = speed / v_angular
    print(radius)

    init_pos = Vector2(0, 0)

    # init_vel = input_vecs[0] * speed
    if d_angle_rad >= 0.0:
        vec_to_center = rotate_vector_2d(input_vecs[0], math.radians(90.0))
    else:
        vec_to_center = rotate_vector_2d(input_vecs[0], math.radians(-90))
    vec_to_center *= radius
    center_pos = init_pos + vec_to_center
    print(f'center_pos: {center_pos}')

    dt = 0.01
    curr_t = input_time_stamps[0]
    curr_theta_rad = 0.0
    res = []
    while curr_t < input_time_stamps[-1]:
        curr_pos = center_pos + rotate_vector_2d(-vec_to_center, curr_theta_rad)
        # print(curr_pos)
        res.append(curr_pos)
        curr_theta_rad += v_angular * dt
        curr_t += dt

    return res


def calc_next_curve_sample(in_step_time, in_pos, in_curr_dir_angle, in_linear_speed, in_angular_speed_degs):
    assert abs(in_angular_speed_degs) > 0.001
    angular_speed_rads_abs = math.radians(in_angular_speed_degs)
    radius = in_linear_speed / angular_speed_rads_abs

    curr_dir = rotate_vector_2d(Vector2(1, 0), math.radians(in_curr_dir_angle))
    if in_angular_speed_degs >= 0.0:
        vec_to_center = rotate_vector_2d(curr_dir, math.radians(90.0))
        turn_dir = +1
    else:
        vec_to_center = rotate_vector_2d(curr_dir, math.radians(-90))
        turn_dir = -1
    vec_to_center *= radius
    center_pos = in_pos + vec_to_center
    print(center_pos)

    theta_rad = angular_speed_rads_abs * turn_dir * in_step_time
    new_pos = center_pos + rotate_vector_2d(-vec_to_center, theta_rad)
    new_dir_angle = in_curr_dir_angle + in_angular_speed_degs * in_step_time
    return new_pos, new_dir_angle


def generate_arc_trajectory2(in_time_horizon, in_step_time, in_curr_pos, in_curr_dir_angle, in_target_dir_angle, in_speed, in_reach_time):
    assert in_reach_time > 0.0
    d_angle_rad = math.radians(in_target_dir_angle - in_curr_dir_angle)
    angular_speed_rad = abs(d_angle_rad) / in_reach_time
    assert angular_speed_rad > 0.0
    radius = in_speed / angular_speed_rad
    print(radius)

    curr_dir = rotate_vector_2d(Vector2(1, 0), math.radians(in_curr_dir_angle))
    # init_vel = input_vecs[0] * speed
    if d_angle_rad >= 0.0:
        vec_to_center = rotate_vector_2d(curr_dir, math.radians(90.0))
        turn_dir = +1
    else:
        vec_to_center = rotate_vector_2d(curr_dir, math.radians(-90))
        turn_dir = -1
    vec_to_center *= radius

    center_pos = in_curr_pos + vec_to_center
    print(f'center_pos: {center_pos}')

    phase1_time = min(in_reach_time, in_time_horizon)

    res = []

    curr_pos = in_curr_pos
    curr_t = 0.0
    curr_theta_rad = 0.0

    # phase 1: circular motion
    while curr_t < phase1_time:
        curr_pos = center_pos + rotate_vector_2d(-vec_to_center, curr_theta_rad)
        res.append(curr_pos)
        curr_theta_rad += angular_speed_rad * turn_dir * in_step_time
        curr_t += in_step_time

    # phase 2: exit to straight motion
    velocity_ws = rotate_vector_2d(Vector2(1, 0), math.radians(in_target_dir_angle)) * in_speed
    while curr_t < in_time_horizon:
        curr_pos = curr_pos + velocity_ws * in_step_time
        res.append(curr_pos)

        curr_t += in_step_time

    return res


# spiral trajectory without fixed center
def generate_spiral_trajectory(in_time_horizon, in_step_time, in_curr_pos, in_curr_dir_angle, in_speed):

    curr_t = in_step_time
    curr_dir_angle = in_curr_dir_angle
    curr_pos = copy.copy(in_curr_pos)
    angular_speed_degs = 180.0

    res = []
    while curr_t < in_time_horizon:
        new_pos, new_dir_angle = calc_next_curve_sample(in_step_time, curr_pos, curr_dir_angle, in_speed, angular_speed_degs)
        res.append(new_pos)

        curr_t += in_step_time
        curr_dir_angle = new_dir_angle
        curr_pos = copy.copy(new_pos)
        angular_speed_degs -= 10.0 * in_step_time

    return res


def test_func():
    input_angles = [0.0, 20.0, 40.0, 60.0, 90.0]
    input_time_stamps = [0.1, 0.15, 0.2, 0.25, 0.3]
    speed = 400.0
    res = generate_arc_trajectory(input_angles, input_time_stamps, speed)
    # input_time_stamps = [0.1, 0.2, 0.3, 0.4]
    # generate_arc_trajectory(input_angles, input_time_stamps, speed)
    xs = []
    ys = []
    for i in range(len(res)):
        xs.append(res[i].x)
        ys.append(res[i].y)
    plt.scatter(xs, ys)
    plt.show()


def test_func2():
    time_horizon = 1.3
    step_time = 1.0/30
    curr_pos = Vector2(0, 0)
    curr_dir_angle = 0.0
    target_dir_angle = -90.0
    speed = 400.0
    reach_time = 0.4

    res = generate_arc_trajectory2(time_horizon, step_time, curr_pos, curr_dir_angle, target_dir_angle, speed, reach_time)
    for i in range(1, len(res)):
        print((res[i] - res[i - 1]).magnitude())

    xs = []
    ys = []
    for i in range(len(res)):
        xs.append(res[i].x)
        ys.append(res[i].y)

    plt.scatter(xs, ys)
    plt.axis('equal')
    plt.show()


def test_spiral_func():
    time_horizon = 10.0
    step_time = 1.0/30
    curr_pos = Vector2(0, 0)
    curr_dir_angle = 0.0
    speed = 400.0

    res = generate_spiral_trajectory(time_horizon, step_time, curr_pos, curr_dir_angle, speed)
    # for i in range(1, len(res)):
    #    print((res[i] - res[i - 1]).magnitude())

    xs = []
    ys = []
    for i in range(len(res)):
        xs.append(res[i].x)
        ys.append(res[i].y)

    plt.scatter(xs, ys)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    # test_func2()
    test_spiral_func()
