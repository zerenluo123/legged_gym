from glob import glob


robot_list = glob('urdf/default/*.urdf')
print(robot_list)