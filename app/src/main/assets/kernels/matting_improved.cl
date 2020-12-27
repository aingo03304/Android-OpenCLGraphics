#define WIN_SIZE 9
#define WIN_RAD 1
#define WIN_DIAM 3
#define EPS 1e-7
#define CHANNEL 3

__kernel void calculateLaplacian(__global int *image,
                                 __global float *laplacian,
                                 __global int *col_index,
                                 __global int *row_index,
                                 int height,
                                 int width) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    int c_h = height - 2 * WIN_RAD;
    int c_w = width - 2 * WIN_RAD;

    float winImage[WIN_DIAM][WIN_DIAM][CHANNEL] = {0};
    float mu[CHANNEL] = {0};
    float var[CHANNEL][CHANNEL] = {0};
    float X[WIN_DIAM][WIN_DIAM][CHANNEL] = {0};
    float Y[WIN_DIAM][WIN_DIAM][WIN_DIAM][WIN_DIAM] = {0};

    for (int i = -WIN_RAD; i <= WIN_RAD; i++) {
        for (int j = -WIN_RAD; j <= WIN_RAD; j++) {
            winImage[i+WIN_RAD][j+WIN_RAD][0] = (float)((image[(row + i) * width + col + j] >> 16) & 0xff) / 255.0;
            winImage[i+WIN_RAD][j+WIN_RAD][1] = (float)((image[(row + i) * width + col + j] >> 8) & 0xff) / 255.0;
            winImage[i+WIN_RAD][j+WIN_RAD][2] = (float)((image[(row + i) * width + col + j]) & 0xff) / 255.0;
            mu[0] += winImage[i+WIN_RAD][j+WIN_RAD][0];
            mu[1] += winImage[i+WIN_RAD][j+WIN_RAD][1];
            mu[2] += winImage[i+WIN_RAD][j+WIN_RAD][2];
        }
    }

    mu[0] /= WIN_SIZE; // 1x3 mean vector for window
    mu[1] /= WIN_SIZE;
    mu[2] /= WIN_SIZE;

    for (int i = -WIN_RAD; i <= WIN_RAD; i++) {
        for (int j = -WIN_RAD; j <= WIN_RAD; j++) {
            var[0][0] += winImage[i+WIN_RAD][j+WIN_RAD][0] * winImage[i+WIN_RAD][j+WIN_RAD][0];
            var[0][1] += winImage[i+WIN_RAD][j+WIN_RAD][0] * winImage[i+WIN_RAD][j+WIN_RAD][1];
            var[0][2] += winImage[i+WIN_RAD][j+WIN_RAD][0] * winImage[i+WIN_RAD][j+WIN_RAD][2];

            var[1][0] += winImage[i+WIN_RAD][j+WIN_RAD][1] * winImage[i+WIN_RAD][j+WIN_RAD][0];
            var[1][1] += winImage[i+WIN_RAD][j+WIN_RAD][1] * winImage[i+WIN_RAD][j+WIN_RAD][1];
            var[1][2] += winImage[i+WIN_RAD][j+WIN_RAD][1] * winImage[i+WIN_RAD][j+WIN_RAD][2];

            var[2][0] += winImage[i+WIN_RAD][j+WIN_RAD][2] * winImage[i+WIN_RAD][j+WIN_RAD][0];
            var[2][1] += winImage[i+WIN_RAD][j+WIN_RAD][2] * winImage[i+WIN_RAD][j+WIN_RAD][1];
            var[2][2] += winImage[i+WIN_RAD][j+WIN_RAD][2] * winImage[i+WIN_RAD][j+WIN_RAD][2];
        }
    }

    for (int i = 0; i < CHANNEL; i++) {
        for (int j = 0; j < CHANNEL; j++) {
            var[i][j] /= WIN_SIZE;
            var[i][j] -= mu[i] * mu[j];
        }
    }

    for (int i = 0; i < CHANNEL; i++) {
        for (int j = 0; j < CHANNEL; j++) {
            var[i][j] += (i == j) ? (float)(EPS / WIN_SIZE) : 0.0;
        }
    }

    float var_inv[CHANNEL][CHANNEL];
    float det = var[0][0]*var[1][1]*var[2][2] +
                var[0][1]*var[1][2]*var[2][0] +
                var[0][2]*var[1][0]*var[2][1] -
                var[0][2]*var[1][1]*var[2][0] -
                var[0][1]*var[1][0]*var[2][2] -
                var[0][0]*var[1][2]*var[2][1];

    var_inv[0][0] = (var[1][1]*var[2][2] - var[1][2]*var[2][1]) / det;
    var_inv[0][1] = (var[0][2]*var[2][1] - var[0][1]*var[2][2]) / det;
    var_inv[0][2] = (var[0][1]*var[1][2] - var[0][2]*var[1][1]) / det;
    var_inv[1][0] = (var[1][2]*var[2][0] - var[1][0]*var[2][2]) / det;
    var_inv[1][1] = (var[0][0]*var[2][2] - var[0][2]*var[2][0]) / det;
    var_inv[1][2] = (var[0][2]*var[1][0] - var[0][0]*var[1][2]) / det;
    var_inv[2][0] = (var[1][0]*var[2][1] - var[1][1]*var[2][0]) / det;
    var_inv[2][1] = (var[0][1]*var[2][0] - var[0][0]*var[2][1]) / det;
    var_inv[2][2] = (var[0][0]*var[1][1] - var[0][1]*var[1][0]) / det;

    for (int i = -WIN_RAD; i <= WIN_RAD; i++) {
        winImage[i+WIN_RAD][0][0] -= mu[0];
        winImage[i+WIN_RAD][0][1] -= mu[1];
        winImage[i+WIN_RAD][0][2] -= mu[2];
	
        winImage[i+WIN_RAD][1][0] -= mu[0];
        winImage[i+WIN_RAD][1][1] -= mu[1];
        winImage[i+WIN_RAD][1][2] -= mu[2];

	winImage[i+WIN_RAD][2][0] -= mu[0];
	winImage[i+WIN_RAD][2][1] -= mu[1];
	winImage[i+WIN_RAD][2][2] -= mu[2];
    }

    for (int i = -WIN_RAD; i <= WIN_RAD; i++) {
        for (int j = -WIN_RAD; j <= WIN_RAD; j++) {
	    X[i+WIN_RAD][j+WIN_RAD][0] += winImage[i+WIN_RAD][j+WIN_RAD][0] * var_inv[0][0];
	    X[i+WIN_RAD][j+WIN_RAD][0] += winImage[i+WIN_RAD][j+WIN_RAD][1] * var_inv[1][0];
	    X[i+WIN_RAD][j+WIN_RAD][0] += winImage[i+WIN_RAD][j+WIN_RAD][2] * var_inv[2][0];

	    X[i+WIN_RAD][j+WIN_RAD][1] += winImage[i+WIN_RAD][j+WIN_RAD][0] * var_inv[0][1];
	    X[i+WIN_RAD][j+WIN_RAD][1] += winImage[i+WIN_RAD][j+WIN_RAD][1] * var_inv[1][1];
	    X[i+WIN_RAD][j+WIN_RAD][1] += winImage[i+WIN_RAD][j+WIN_RAD][2] * var_inv[2][1];

	    X[i+WIN_RAD][j+WIN_RAD][2] += winImage[i+WIN_RAD][j+WIN_RAD][0] * var_inv[0][2];
	    X[i+WIN_RAD][j+WIN_RAD][2] += winImage[i+WIN_RAD][j+WIN_RAD][1] * var_inv[1][2];
	    X[i+WIN_RAD][j+WIN_RAD][2] += winImage[i+WIN_RAD][j+WIN_RAD][2] * var_inv[2][2];
        }
    }

    for (int i = -WIN_RAD; i <= WIN_RAD; i++) {
        for (int j = -WIN_RAD; j <= WIN_RAD; j++) {
            for (int k = -WIN_RAD; k <= WIN_RAD; k++) {
                for (int l = -WIN_RAD; l <= WIN_RAD; l++) {
		    Y[i+WIN_RAD][j+WIN_RAD][k+WIN_RAD][l+WIN_RAD] +=
			    X[i+WIN_RAD][j+WIN_RAD][0] *
			    winImage[k+WIN_RAD][l+WIN_RAD][0];

		    Y[i+WIN_RAD][j+WIN_RAD][k+WIN_RAD][l+WIN_RAD] +=
			    X[i+WIN_RAD][j+WIN_RAD][1] *
			    winImage[k+WIN_RAD][l+WIN_RAD][1];

		    Y[i+WIN_RAD][j+WIN_RAD][k+WIN_RAD][l+WIN_RAD] +=
			    X[i+WIN_RAD][j+WIN_RAD][0] *
			    winImage[k+WIN_RAD][l+WIN_RAD][1];

                    Y[i+WIN_RAD][j+WIN_RAD][k+WIN_RAD][l+WIN_RAD] += 1;
                    Y[i+WIN_RAD][j+WIN_RAD][k+WIN_RAD][l+WIN_RAD] /= WIN_SIZE;
                    Y[i+WIN_RAD][j+WIN_RAD][k+WIN_RAD][l+WIN_RAD] = ((i == k && j == l) ? 1.0 : 0.0) -
                                                                    Y[i+WIN_RAD][j+WIN_RAD][k+WIN_RAD][l+WIN_RAD];
                }
            }
        }
    }

    for (int k = 0; k < WIN_SIZE; k++) {
        for (int i = -WIN_RAD; i <= WIN_RAD; i++) {
            for (int j = -WIN_RAD; j <= WIN_RAD; j++) {
		int temp_col = (i+WIN_RAD) * WIN_DIAM + (j+WIN_RAD);
		int temp_offset = k * WIN_SIZE + temp_col;
	        col_index[temp_offset * WIN_SIZE * WIN_SIZE + row * c_w + col] = (i+WIN_RAD) * width + (j+WIN_RAD);
            }
        }
    }

    for (int i = -WIN_RAD; i <= WIN_RAD; i++) {
        for (int j = -WIN_RAD; j <= WIN_RAD; j++) {
	    int temp_row = (i+WIN_RAD) * WIN_DIAM + (j+WIN_RAD);
            for (int k = 0; k < WIN_SIZE; k++) {
		int temp_offset = temp_row * WIN_SIZE + k;
	        row_index[temp_offset * WIN_SIZE * WIN_SIZE + row * c_w + col] = (i+WIN_RAD) * width + (j+WIN_RAD);
            }
        }
    }

    for (int i = -WIN_RAD; i <= WIN_RAD; i++) {
        for (int j = -WIN_RAD; j <= WIN_RAD; j++) {
            int temp_row = (i+WIN_RAD) * WIN_DIAM + (j+WIN_RAD);
            for (int k = -WIN_RAD; k <= WIN_RAD; k++) {
                for (int l = -WIN_RAD; l <= WIN_RAD; l++) {
	            int temp_col = (k+WIN_RAD) * WIN_DIAM + (l+WIN_RAD);
		    int temp_offset = temp_row * WIN_SIZE + temp_col;
		    laplacian[temp_offset * WIN_SIZE * WIN_SIZE + row * c_w + col] = Y[i+WIN_RAD][j+WIN_RAD][k+WIN_RAD][l+WIN_RAD];
                }
            }
        }
    }
}
