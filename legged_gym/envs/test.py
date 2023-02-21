from glob import glob


robot_list = glob(f'../../resources/robots/arclabdog/urdf/default/*.urdf')
print(robot_list)