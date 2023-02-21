from glob import glob


robot_list = glob('../../../resources/robots/arclabdog/urdf/default/*.urdf')
print(robot_list)