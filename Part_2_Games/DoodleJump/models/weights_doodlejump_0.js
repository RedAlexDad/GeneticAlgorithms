var sensorWeb = [[-250, -445], [-200, -445], [-150, -445], [-100, -445], [-50, -445], [0, -445], [50, -445], [100, -445], [150, -445], [200, -445], [-250, -370], [-200, -370], [-150, -370], [-100, -370], [-50, -370], [0, -370], [50, -370], [100, -370], [150, -370], [200, -370], [-250, -295], [-200, -295], [-150, -295], [-100, -295], [-50, -295], [0, -295], [50, -295], [100, -295], [150, -295], [200, -295], [-250, -220], [-200, -220], [-150, -220], [-100, -220], [-50, -220], [0, -220], [50, -220], [100, -220], [150, -220], [200, -220], [-250, -145], [-200, -145], [-150, -145], [-100, -145], [-50, -145], [0, -145], [50, -145], [100, -145], [150, -145], [200, -145], [-250, -70], [-200, -70], [-150, -70], [-100, -70], [-50, -70], [0, -70], [50, -70], [100, -70], [150, -70], [200, -70], [-250, 5], [-200, 5], [-150, 5], [-100, 5], [-50, 5], [0, 5], [50, 5], [100, 5], [150, 5], [200, 5], [-250, 80], [-200, 80], [-150, 80], [-100, 80], [-50, 80], [0, 80], [50, 80], [100, 80], [150, 80], [200, 80], [-250, 155], [-200, 155], [-150, 155], [-100, 155], [-50, 155], [0, 155], [50, 155], [100, 155], [150, 155], [200, 155], [-250, 230], [-200, 230], [-150, 230], [-100, 230], [-50, 230], [0, 230], [50, 230], [100, 230], [150, 230], [200, 230], [-250, 305], [-200, 305], [-150, 305], [-100, 305], [-50, 305], [0, 305], [50, 305], [100, 305], [150, 305], [200, 305], [-250, 380], [-200, 380], [-150, 380], [-100, 380], [-50, 380], [0, 380], [50, 380], [100, 380], [150, 380], [200, 380]];

var W = [[0.423, 0.347, -1.4, -0.248, -0.556], [0.245, -0.944, 1.13199, -0.61, -0.474], [-0.938, 0.082, -0.346, -0.537, 0.038], [1.224, -2.468, -0.219, -0.303, 0.949], [-2.1, 0.424, -1.487, -0.373, -0.128], [-1.127, 0.205, -1.417, 0.18, -0.671], [-1.15399, 0.95, -0.327, 0.877, -0.488], [1.199, -1.117, 0.75, 0.365, -0.171], [0.28899, 1.65, -1.707, 0.112, -2.021], [0.539, 2.26399, 0.538, -0.384, 0.15], [1.397, 1.739, -0.198, -1.29, 0.686], [-0.058, 1.15999, -0.338, 1.397, 0.423], [0.28999, -1.264, 0.405, -2.407, -0.013], [-0.108, 1.081, -0.587, 0.837, -1.175], [1.13799, 0.025, 0.161, 1.235, -0.437], [0.136, -0.972, -0.57599, 0.058, 0.07], [-1.404, 0.014, 0.178, 2.094, -1.022], [0.311, -0.465, -0.052, -1.787, -0.252], [0.907, 0.254, 0.476, 1.119, -0.967], [-1.413, -0.905, 1.862, -0.868, 0.137], [-0.828, 0.465, -0.904, 1.234, -0.131], [-0.244, 1.01, -0.904, -0.67, -1.594], [-0.252, 0.28999, -0.926, -0.04, 1.305], [0.607, -0.494, 0.317, -0.296, 1.771], [1.748, 0.486, 1.657, 0.102, -0.813], [-0.34, 0.23, -0.059, -1.075, -0.212], [0.341, 0.246, -2.084, 0.097, -0.756], [0.778, -0.033, 1.611, 1.72, 1.04], [-0.07, 0.258, 0.423, 0.318, -0.675], [0.951, 0.116, -0.328, 0.845, -0.413], [0.767, 0.898, 1.347, 1.695, -0.386], [0.866, -0.605, -0.183, -1.901, 0.954], [-0.148, 0.671, -0.56999, 0.954, -0.531], [2.013, -0.37, -1.524, 0.516, 0.657], [-1.486, 1.929, -0.507, 1.018, 0.363], [0.199, -0.703, -0.402, 0.411, 1.676], [0.28199, 1.115, 0.01, -0.255, -0.021], [0.702, 0.329, 0.831, 1.37, -1.095], [0.553, 0.717, -0.838, 0.177, -0.206], [-0.755, 0.084, 0.904, -0.193, 1.135], [0.227, -0.368, -0.9, 2.02, -0.696], [1.274, -1.95, -1.577, 1.0, 0.991], [0.076, 0.0, -0.326, -1.578, 0.801], [-0.849, 0.689, -1.12999, -0.541, 1.131], [-0.819, 0.01799, -1.303, 0.206, -0.132], [0.415, -0.189, 1.002, 1.511, 1.951], [0.075, -1.343, -1.04299, 0.06, 0.229], [1.106, -0.201, -0.459, -0.688, 1.323], [0.34, -0.939, -1.23, 0.434, -1.698], [0.886, 0.774, 0.314, -1.431, -1.457], [-0.535, -0.844, 0.222, -0.04, -0.18], [2.731, -0.077, 1.728, 0.037, 0.341], [-0.769, -1.226, -0.322, -0.679, 0.953], [0.138, 0.57299, 1.312, 0.113, -0.446], [-0.543, -0.903, -0.687, 1.214, -0.918], [-0.557, -1.249, 0.05, 0.175, 0.726], [0.739, -0.765, -1.192, 0.696, 0.242], [0.96, -0.323, -1.87, -0.95, -0.448], [-0.039, -0.175, 1.41, 0.542, 1.354], [0.8, 1.218, 1.15399, -0.125, -0.998], [-1.088, -0.933, 0.903, -0.241, -0.242], [-2.382, 0.131, -0.715, 0.088, 1.309], [-2.043, -0.679, -0.663, -0.953, -0.136], [0.306, -1.11, -0.052, -0.022, -0.97], [1.599, -0.474, -0.931, -1.104, -0.28399], [-0.58099, -1.499, 0.433, -1.56, -0.991], [-0.102, -0.456, 0.57299, -0.853, 1.096], [-0.637, -0.723, 0.27, -0.156, 1.599], [-0.805, 0.485, -1.905, -1.727, -1.184], [1.161, -0.021, 2.20699, -2.233, 0.476], [-0.369, -0.889, 0.907, -0.068, -1.489], [-0.089, -1.02, 0.743, -0.366, -0.317], [-1.685, -0.424, 0.337, 0.123, 0.621], [0.788, 0.034, 0.034, -1.176, -1.469], [-1.408, 0.12, -0.033, 1.139, 0.14099], [-0.276, -0.055, -0.376, 1.016, 0.248], [-2.353, 0.49, 1.945, -0.766, -1.347], [0.181, 0.617, 0.379, -0.345, -0.198], [-1.203, -0.28399, -0.098, -0.242, 1.125], [0.639, 1.065, -1.16599, -0.169, 0.179], [0.435, -0.795, 0.089, 0.666, -1.09], [-0.486, -0.027, 1.616, -1.264, -1.533], [-1.66, -0.183, -0.246, 1.794, -0.198], [-0.085, -0.653, -1.611, 1.437, 0.662], [-1.77, -0.732, -0.338, -0.939, -0.931], [-0.631, -0.478, 0.592, 0.187, -0.785], [-0.458, 0.455, -0.404, 0.665, -0.532], [0.225, 0.862, -2.05799, -0.675, 1.39], [-0.481, -0.154, -0.001, 0.635, -0.656], [-0.618, -0.728, 0.536, 1.212, -1.412], [-2.096, -0.439, -2.004, 1.367, 2.015], [-0.623, 0.924, -0.055, 1.333, -1.198], [-0.599, 1.463, -0.674, 0.814, 0.57399], [0.795, 0.192, 0.039, -2.102, -1.417], [1.333, -1.008, -0.347, -1.155, 1.753], [1.124, 1.407, -0.503, -0.763, 1.42], [0.902, -0.52, -0.526, 0.226, -1.219], [0.469, 1.176, -0.505, -0.853, 1.116], [2.359, -0.712, -2.807, -0.329, 0.123], [0.682, 1.314, 1.065, -0.934, -0.246], [1.157, 0.708, -0.432, 0.817, 0.46], [1.41, -0.766, 0.238, 0.244, -0.694], [1.726, 0.926, -2.642, -0.97, 1.206], [-1.911, -1.02499, -0.748, -0.456, -0.216], [1.419, -0.406, -0.913, -0.01799, -1.078], [1.12, -1.54, -1.177, 1.03299, 0.034], [2.173, -0.124, 1.161, 1.775, -0.044], [-0.01, -0.313, -0.25, -1.532, 0.996], [1.086, 0.646, 1.022, 1.418, -1.906], [0.273, 1.326, 0.414, 0.753, -1.15999], [-2.009, 1.537, -0.28999, -0.628, 0.724], [-0.193, 0.28299, -0.587, 1.38, -0.757], [0.972, 0.026, 0.369, -0.426, -0.804], [1.239, -0.866, -0.541, 1.837, 0.503], [0.392, 0.56299, -1.276, 1.319, 0.011], [-0.56299, -0.558, -0.861, 0.919, 1.739], [1.05, 1.343, -1.059, -0.806, -0.017], [-0.07099, 0.45, 1.117, -0.516, 0.4], [-0.183, -0.091, -1.092, 0.508, -0.074], [-0.683, -0.33, 0.337, -0.637, -0.65], [-0.361, -0.372, 0.463, 0.56399, 1.819], [-0.997, 0.89, 0.159, 1.234, 0.842], [0.064, -1.095, -0.172, 1.159, -0.087]];

var W2 = [[-1.288, 1.306, 0.022], [-0.179, -1.125, -1.637], [-1.124, -0.198, -0.768], [-0.306, 1.194, 0.181], [-1.083, 1.807, 0.375]];
