import numpy as np

class leaky_relu:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, z, derivative=False):
        if derivative:
            return np.where(z > 0, 0.2, 0.1)
        return np.where(z < 0, z * 0.1, z * 0.2)

# Tạo một instance của class leaky_relu với alpha = 0.01
leaky_relu_func = leaky_relu(alpha=0.01)

# Tạo một mảng numpy đầu vào
z = np.array([-1, -0.5, 0, 0.5, 1])

# Áp dụng hàm leaky_relu cho mảng z
output = leaky_relu_func(z,True)

print("Output after applying leaky ReLU:", output)