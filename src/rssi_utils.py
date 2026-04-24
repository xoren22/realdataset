import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import re
import h5py
import datetime


def get_rssi(df):
    test = df['Strength'].values
    list_values = []
    for i in np.arange(len(test)):
        match = re.search(r'\d+', test[i]).group()
        list_values.append(-float(match))

    return list_values


def rssi_contour_plot(matrix, matrix_ind, ap_location, setup_num, save_figure=True):
    # Get the coordinates of nonzero values

    matrix = np.flipud(matrix)
    matrix_ind = np.flipud(matrix_ind)

    y_indices, x_indices = np.nonzero(matrix)  # (row, column)
    z_values = matrix[y_indices, x_indices]  # Extract nonzero values

    # Create a contour plot
    plt.tricontourf(x_indices, y_indices, z_values, levels=32, cmap="viridis")
    plt.colorbar(label="dBm")

    # Scatter plot for known data points
    plt.scatter(x_indices, y_indices, color="red", label="Measurement Points")

    # Add enumeration at each data point
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x, y] != 0:
                plt.text(y + 0.2, x + 0.2, str(int(matrix_ind[x,y])), fontsize=8, color='cyan')

    # Set axis limits to show more space
    padding = 2  # Adjust this value to control extra space
    plt.xlim(min(x_indices) - padding, max(x_indices) + padding)
    plt.ylim(min(y_indices) - padding, max(y_indices) + padding)

    # Labels and title
    plt.title("RSSI Values, setup {}, AP location {}".format(setup_num, ap_location))
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()

    if save_figure:
        plt.savefig("rssi_{}.png".format(setup_num), dpi=300, bbox_inches="tight", transparent=True)

    # Show plot
    plt.show()


def extract_values(df, setup_num=1):
    # Generate a 26x28 grid with 1m spacing
    matrix = np.zeros((28, 30))
    matrix_ind = np.zeros((28, 30))
    filtered_df = df[df['Setup'] == str(setup_num)]

    # Office
    values = get_rssi(filtered_df)
    values_real = get_rssi(filtered_df)

    if setup_num == 14 or setup_num==7 or setup_num==5: # Missing value in list 2 at pos 1. Adding interpolated value.
        values.insert(0, (values[0]+values[1])/2)
        values_real.insert(0, None)


    extended_list, _ = extend_list(values, values_real, 53)
    matrix[0, 0:12:2] = np.array(values[0:6])[::-1]
    matrix_ind[0, 0:12:2] = np.arange(0, 6, 1)[::-1]+1

    matrix[0,1] = matrix[0,0]
    matrix_ind[0,1] = matrix_ind[0,0]
    matrix[0,0] = 0
    matrix_ind[0,0] = 0

    matrix[2, 0:12:2] = np.array(values[6:12])
    matrix_ind[2, 0:12:2] = np.arange(6, 12, 1)+1

    matrix[4, 0:12:2] = np.array(values[12:18])[::-1]
    matrix_ind[4, 0:12:2] = np.arange(12, 18, 1)[::-1]+1

    matrix[6, 0:12:2] = np.array(values[18:24])
    matrix_ind[6, 0:12:2] = np.arange(18, 24, 1)+1

    matrix[8, 0:12:2] = np.array(values[24:30])[::-1]
    matrix_ind[8, 0:12:2] = np.arange(24, 30, 1)[::-1]+1

    # Corridor
    matrix[10, 0:30:2] = np.array(values[30:45])
    matrix_ind[10, 0:30:2] = np.arange(30, 45, 1)+1
    # Elevator hall
    matrix[12:34:2, 25] = np.array(values[45:])
    matrix_ind[12:34:2, 25] = np.arange(45, 53, 1)+1

    return matrix, matrix_ind


def extend_list(lst, lst_real, target_length):
    if len(lst) < target_length:
        for i in range(target_length-len(lst)):
            fill_value2 = lst[-1]
            lst_real.append(None)
            lst.extend([fill_value2] * (target_length - len(lst)))
    return lst, lst_real


def generate_dataset_file(df, filename='WiFi_RSSI_data'):
    ap_locations = [1, 3, 5, 18, 30, 26, 25, 12, 33, 37, 44, 49, 1, 18, 26, 30, 20, 14, 35, 47]

    # Generate unique time-stamped filename for the data
    # Current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Base path and filename
    filename = filename + timestamp + '.h5'

    # Dataset parameters
    num_setups = 20
    dataset_list = []
    setup_list = []
    indices_list = []

    for i in range(num_setups):
        matrix, matrix_ind = extract_values(df, setup_num=i + 1)
        dataset_list.append(matrix)
        setup_list.append(i+1)
        indices_list.append(matrix_ind)

    # Create the HDF5 file
    with h5py.File(filename, 'w') as f:
        # Create groups to organize entries
        f.create_dataset('data', data=dataset_list)
        f.create_dataset('setup', data=setup_list)
        f.create_dataset('indices', data=indices_list)
        f.create_dataset('ap_locations', data=ap_locations)

    print("HDF5 dataset file created.")
