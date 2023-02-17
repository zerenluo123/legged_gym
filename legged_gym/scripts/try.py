import numpy as np

custom_bound_per_env_lower = -1
custom_bound_per_env_upper = 1
random_search = 10
dim = 3


x_random = np.random.uniform(custom_bound_per_env_lower, custom_bound_per_env_upper,
                                         size=(random_search, dim))

print(x_random)