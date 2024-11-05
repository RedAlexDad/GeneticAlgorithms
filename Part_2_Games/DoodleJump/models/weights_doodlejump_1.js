var sensorWeb = [[-250, -445], [-200, -445], [-150, -445], [-100, -445], [-50, -445], [0, -445], [50, -445], [100, -445], [150, -445], [200, -445], [-250, -370], [-200, -370], [-150, -370], [-100, -370], [-50, -370], [0, -370], [50, -370], [100, -370], [150, -370], [200, -370], [-250, -295], [-200, -295], [-150, -295], [-100, -295], [-50, -295], [0, -295], [50, -295], [100, -295], [150, -295], [200, -295], [-250, -220], [-200, -220], [-150, -220], [-100, -220], [-50, -220], [0, -220], [50, -220], [100, -220], [150, -220], [200, -220], [-250, -145], [-200, -145], [-150, -145], [-100, -145], [-50, -145], [0, -145], [50, -145], [100, -145], [150, -145], [200, -145], [-250, -70], [-200, -70], [-150, -70], [-100, -70], [-50, -70], [0, -70], [50, -70], [100, -70], [150, -70], [200, -70], [-250, 5], [-200, 5], [-150, 5], [-100, 5], [-50, 5], [0, 5], [50, 5], [100, 5], [150, 5], [200, 5], [-250, 80], [-200, 80], [-150, 80], [-100, 80], [-50, 80], [0, 80], [50, 80], [100, 80], [150, 80], [200, 80], [-250, 155], [-200, 155], [-150, 155], [-100, 155], [-50, 155], [0, 155], [50, 155], [100, 155], [150, 155], [200, 155], [-250, 230], [-200, 230], [-150, 230], [-100, 230], [-50, 230], [0, 230], [50, 230], [100, 230], [150, 230], [200, 230], [-250, 305], [-200, 305], [-150, 305], [-100, 305], [-50, 305], [0, 305], [50, 305], [100, 305], [150, 305], [200, 305], [-250, 380], [-200, 380], [-150, 380], [-100, 380], [-50, 380], [0, 380], [50, 380], [100, 380], [150, 380], [200, 380]];

var W = [[0.657, -0.863, 0.362, -1.163, -1.453], [-0.909, 0.06, -0.234, 0.56399, 0.002], [-0.812, 0.052, -0.592, -0.95, -0.223], [-0.14099, -0.658, -0.9, 0.271, 0.771], [-0.172, -1.313, -1.875, -0.345, 0.323], [0.708, 1.07, 0.479, 2.25599, 0.555], [0.012, -0.66, 0.233, 0.162, 0.865], [-0.29199, 0.293, -1.939, -0.173, -2.419], [-1.205, 0.597, -1.476, -0.814, 0.755], [1.618, -0.765, 1.73, 0.177, 1.554], [1.394, -0.32, -0.017, -1.133, 0.293], [0.733, -1.373, 0.43, 0.711, 0.539], [-1.12799, -1.449, 0.157, 0.954, 0.993], [0.1, 1.462, 1.313, -0.973, 0.028], [-0.62, -1.03499, -0.466, 0.84, 0.59], [-1.391, 0.105, -1.619, -0.065, -1.737], [-0.849, 0.344, -0.69, -1.422, -0.281], [1.529, 0.764, -0.967, 0.022, 0.614], [-0.314, 0.063, 1.28, -0.487, 0.68], [0.819, -0.552, 0.513, -1.47, 1.407], [0.724, -2.16699, -1.032, -0.994, 0.903], [0.089, -1.626, -0.476, -0.175, 0.587], [0.629, 1.109, 0.219, -1.599, -0.739], [0.227, 0.963, 0.107, -0.658, 1.403], [-2.741, -1.517, 0.742, 0.161, 0.95], [1.641, 0.57399, 0.777, -1.676, 0.56], [1.202, 1.608, -0.278, -0.604, 0.213], [0.436, -0.677, 1.627, 0.495, 0.199], [-1.412, -0.416, 0.71, 0.26, -1.538], [0.927, -0.376, -0.468, -1.681, -0.079], [-0.58399, -0.58099, 0.184, 2.128, 0.084], [-0.514, 0.115, 0.176, -0.587, 0.421], [-0.56499, 0.14599, 0.8, -0.067, 1.198], [1.12599, 1.54, 1.28299, 0.247, 0.148], [-0.57699, 0.58199, 1.965, 1.793, 0.96], [-0.731, 1.286, -0.174, 0.053, -0.758], [0.628, 1.315, 2.226, -0.28, -0.562], [-1.539, 0.156, 1.059, -1.322, 0.047], [1.324, -0.492, 0.391, -1.56, -1.408], [-1.872, 0.228, 0.631, -0.834, -0.313], [1.876, 1.933, -0.029, 0.94, -0.152], [0.78, -0.618, 1.139, 0.714, 0.223], [-0.016, -1.216, 0.68, -1.647, -0.006], [-0.944, 1.213, -1.199, 0.917, -0.99], [-0.387, -1.21, -0.437, 0.802, -1.175], [0.076, 0.166, 0.844, -1.13599, -0.375], [0.778, -1.926, -0.031, 0.232, -1.217], [-0.622, 0.205, 0.622, -0.155, 1.06], [0.488, 0.102, -1.26899, -0.204, 1.691], [-0.002, -0.518, -0.417, -0.847, -0.58199], [1.815, -0.168, 0.56499, -0.393, 0.984], [0.016, -2.088, -1.27899, 0.869, 1.2], [0.735, 0.718, 0.438, 0.91, -0.946], [-0.993, -1.177, 1.087, 0.794, -0.339], [-0.269, -0.65, 0.132, -0.239, 1.568], [0.187, 1.793, -0.953, 0.655, -2.121], [0.691, -0.545, 0.799, -1.227, -0.704], [-1.183, 0.807, -0.413, 0.178, -0.314], [-0.00899, -0.052, 0.279, -0.177, 1.325], [0.253, 0.732, 2.821, -0.846, 0.558], [1.054, 0.558, 0.046, 1.606, 0.952], [0.07099, 0.013, 1.03299, 0.973, 0.17], [2.679, 1.483, 1.935, -2.354, 1.478], [-0.428, -0.307, 0.251, -0.657, -0.56999], [-0.006, -2.005, -0.81, 1.129, -0.032], [-1.507, 0.534, 0.52, -1.224, -0.76], [-1.665, 0.384, 0.157, -0.499, 0.162], [1.371, -2.031, -1.222, 0.357, 1.409], [-0.232, 0.498, -0.147, 0.732, -0.301], [1.057, -0.671, -0.708, -0.349, 0.407], [-0.903, 0.436, -1.08, 0.299, 1.643], [-1.378, 2.631, -1.01, 1.114, 0.017], [-0.266, -1.658, -0.908, 0.765, -0.178], [-1.759, 0.407, 1.106, 0.918, 0.198], [0.816, -1.882, 0.089, -0.319, 0.597], [0.275, -0.985, -0.263, -1.006, -0.191], [0.744, -0.772, -0.2, 0.083, -1.096], [-0.236, 0.001, -1.28, -0.07299, 0.525], [-0.534, -1.01, 1.014, 0.427, -0.995], [-0.734, 0.665, -0.553, 0.502, 0.008], [-0.446, 1.006, -1.863, 0.059, 1.03899], [1.498, 0.361, -0.057, 0.787, 0.973], [0.333, -1.022, -0.433, -1.779, 0.31], [1.343, -0.839, -1.107, -0.462, -0.057], [-0.907, 0.25, 0.28399, -1.096, 0.049], [-0.134, 0.356, 0.334, -0.133, -0.716], [0.494, -1.893, -0.818, -0.613, 0.687], [-0.749, -0.214, 0.182, 0.591, 1.426], [0.509, -0.525, 1.22, -1.31, -3.105], [-0.149, -0.225, -1.43, -0.637, 0.835], [-1.045, 0.258, -0.497, -0.133, 0.087], [-1.366, 0.434, 0.35, 1.01, -0.592], [-1.184, 0.809, -0.616, 0.861, -1.177], [-0.964, 0.165, -1.932, -1.778, -1.309], [-0.134, -1.607, -0.367, 0.271, -1.016], [0.386, -0.835, -3.047, -0.788, -0.138], [-1.24, -2.232, -1.06, -0.641, -0.211], [1.17, -0.136, 0.657, 0.467, -0.912], [-0.151, -0.701, -0.746, 0.744, -0.179], [-0.628, -0.939, -0.745, -1.503, 0.803], [-2.432, -0.112, -0.9, 1.076, 0.666], [-1.639, -0.594, 0.187, -0.62, 0.255], [-1.203, 0.095, 0.877, -0.107, -2.404], [-0.496, -0.746, 2.052, -0.975, -0.632], [-0.079, -0.081, -2.96, 0.733, -0.613], [0.398, 0.928, -0.069, -0.57199, -0.363], [-1.368, 0.963, 0.851, -0.837, -0.475], [-0.405, -0.193, 0.304, 0.038, -1.308], [-1.992, -2.047, 0.356, 2.021, -0.045], [1.044, 0.537, 0.943, 0.995, 1.496], [-1.25699, -0.384, -2.231, 0.927, -0.799], [0.28799, -0.721, 0.85, 0.451, 0.05], [-1.002, -0.14, 0.534, 0.123, -0.458], [1.561, 0.101, -1.22, -1.096, -0.427], [0.842, -0.805, -0.53, 0.195, -1.085], [0.026, 0.075, -0.787, -0.59, 0.14399], [0.762, -0.8, 0.811, 0.606, 0.526], [-1.107, -1.081, -1.225, -0.007, -0.729], [1.552, 0.019, 0.688, -0.528, 0.897], [1.248, 0.492, 1.26899, -0.905, -0.28799], [-0.547, 0.123, -0.136, -0.058, 1.493], [-0.765, 0.21, 0.963, -0.416, 0.322], [-0.188, -0.59, 0.318, -0.751, 0.821]];

var W2 = [[0.534, 0.307, -1.706], [1.16199, 0.523, 1.572], [-0.668, 0.98, 1.372], [1.22, -0.5, -0.909], [-0.178, 0.45, -1.184]];