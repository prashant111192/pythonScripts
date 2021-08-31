# /usr/bin/python3

import os
from typing import TYPE_CHECKING, Tuple
import numpy as np
from glob import glob
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from scipy.spatial import distance

if TYPE_CHECKING:
    from numpy import ndarray

pool = mp.Pool(mp.cpu_count()) 


def find_number_files(path: str) -> int:
    """Return total number of csv files."""
    return len(glob(f"{path}/*.txt"))


def read_csv(number_files: int, set_number: int, path: str) -> "ndarray":
    """Read required CSV(text files)."""
    # preparing the name of the files which is iterated upon (the names of the files start from 1)
    name_file = f"{path}H{set_number}_{(1):04d}.txt"
    # 0x [m],1y [m],2u [m/s],3v [m/s],4vorticity [1/s],5magnitude [m/s],6divergence [1/s],7dcev [1],8simple shear [1/s],9simple strain [1/s],10vector direction [degrees]
    arr = loadAndVel(name_file)

    # reshape the arr prepared to make it 3d
    arr = arr.reshape(arr.shape[0], arr.shape[1], 1)

    # +1 becuase range does not iterate obver the last value
    for _ in range(2, number_files + 1):
        name_file = f"{path}H{set_number}_{(1):04d}.txt"
        arr_temp = loadAndVel(name_file)
        arr_temp = arr_temp.reshape(arr_temp.shape[0], arr_temp.shape[1], 1)
        arr = np.append(arr, arr_temp, axis=2)

    return arr


def vel(arr: "ndarray") -> "ndarray":
    """Adding vel magnitude column."""
    u = arr[:, 4]
    v = arr[:, 6]
    w = arr[:, 6]
    sqrt_vec = np.vectorize(math.sqrt)
    vel_mag = sqrt_vec((u*u) + (v*v) + (w*w))
    vel_mag = vel_mag.reshape(len(vel_mag), 1)
    return np.append(arr, vel_mag, axis=1)


def read_csv_2(path: str) -> Tuple["ndarray", int]:
    """Read multiple csv files."""
    # 0_Pos.x[m];1_Pos.y[m];2_Pos.z[m];3_Idp;4_Vel.x[m/s];5_Vel.y[m/s];6_Vel.z[m/s];7_Rhop[kg/m^3];8_Type;
    arr = np.loadtxt(f"{path}/csv_0400.csv", dtype=float, delimiter=";", skiprows=5, usecols=(0, 1, 2, 3, 4, 5, 6, 8))
    # 1Pos.x[m];Pos.y[m];Pos.z[m];Idp;Vel.x[m/s];Vel.y[m/s];Vel.z[m/s];Rhop[kg/m^3];Type;

    # picked 0_x,1_y,2_z, 3_idp 4_vx, 5vy, 6_vz, 7_type
    # extract only the fluids
    low = min(arr[:, 0])
    arr = arr[arr[:, 7] == 3]
    # picked 0_x,1_y,2_z, 3_idp 4_vx, 5vy, 6_vz
    arr = vel(np.delete(arr, 7, 1))
    # arr2 = arr2[arr2[:, 7] == 3]
    # arr2 = np.delete(arr2, 7, 1)
    return arr, low


def loadAndVel(name_file: str) -> "ndarray":
    """TODO."""
    arr = np.loadtxt(name_file, dtype=float, delimiter=",", skiprows=3, usecols=(0, 1, 2, 3, 5))
    # arr = velocities(arr)
    return arr[~np.isnan(arr).any(axis=1)]


def velocities(arr: "ndarray") -> "ndarray":
    """TODO."""
    rad_vec = np.vectorize(math.radians)
    cos_vec = np.vectorize(math.cos)
    sin_vec = np.vectorize(math.sin)
    rads = rad_vec(arr[:, 3])
    u = arr[:, 2]*cos_vec(rads)
    v = arr[:, 2]*sin_vec(rads)
    arr = np.insert(arr, 2, u, axis=1)
    return np.insert(arr, 3, v, axis=1)


def closest_point_distance(piv_arr: "ndarray", sph_arr: "ndarray", height: float) -> Tuple["ndarray", "ndarray"]:
    height = np.ones(len(piv_arr)) * height
    height = height.reshape(len(piv_arr), 1)
    # adding heights to piv data i.e., making them 3d
    piv_arr = np.insert(piv_arr, 2, height, axis=1)
    closest_index = distance.cdist(piv_arr[:, :3, 0], sph_arr[:, :3], metric="euclidean").argmin(1)
    return closest_index, piv_arr


def closest_point_kdtree(piv_arr: "ndarray", sph_arr: "ndarray") -> "ndarray":
    "TODO."
    kdtree = KDTree(sph_arr[:, :2])
    _, closest_index = kdtree.query(piv_arr[:, :2])
    return closest_index


# def cKDTreeMethod(data, num, timestep):
#     tree = cKDTree(arr[:,])
#     nn_dist, index = tree.query()
#     return (nn_dist, index)

# def interpolate_data(arr):
#     interpolator = interp.CloughTocher2DInterpolator(np.array([x,y]).T, z)


def subtract_plt(piv_arr: "ndarray", sph_arr: "ndarray", vel_piv_index: int, vel_sph_index: int, closest_index_sph: "ndarray") -> Tuple["ndarray", "ndarray"]:
    """TODO."""
    # sub_array = abs(piv_arr[:, vel_piv_index, 0] - sph_arr[closest_index_sph, vel_sph_index])
    sub_array = (piv_arr[:, vel_piv_index, 0] - sph_arr[closest_index_sph, vel_sph_index])
    percent_arr = (sub_array * 100) / piv_arr[:, vel_piv_index, 0]
    return percent_arr, sub_array


def plot_histogram(c_: "ndarray", name: str, title: str) -> None:
    "TODO."
    plt.boxplot(c_, showfliers=False)
    # plt.hist(c_,)
    # plt.show()
    plt.savefig(f"./figs2/{title}{str(name)}.png")
    plt.clf()


def plot_graph(x: "ndarray", y: "ndarray", c_: "ndarray", name: str, height: str, title: str) -> None:
    """TODO."""
    plt.scatter(x, y, s=15, c=c_)
    plt.colorbar()
    # plt.clim(0, 100)
    # plt.gca().set_aspect("equal")
    plt.savefig(f"./figs2/{title}{name}{height}.png")
    plt.clf()
    # plt.show()


def plot_2_together(piv_arr: "ndarray", sph_arr: "ndarray", closest: "ndarray", name: str, height: str, title: str) -> None:
    """TODO."""
    # plt.scatter(sph_arr[:,0], sph_arr[:,2],marker = "." )
    # plt.scatter(sph_arr[closest,0], sph_arr[closest,1], sph_arr[closest, 2], marker= ",")
    # plt.scatter(piv_arr[:,0,0], piv_arr[:,1,0], piv_arr[:,2,0])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(sph_arr[closest, 0], sph_arr[closest, 1], sph_arr[closest, 2], s=3)
    # ax.scatter(sph_arr[:, 0], sph_arr[:, 1], sph_arr[:, 2], s=0.1)
    if closest is None:
        plt.scatter(sph_arr[:, 0], sph_arr[:, 1], s=10)
    else:
        plt.scatter(sph_arr[closest, 0], sph_arr[closest, 1], s=10)
    # print(piv_arr[:, 2, 0])
    # ax.scatter(piv_arr[:, 0, 0], piv_arr[:, 1, 0], piv_arr[:, 2, 0], s=1)
    plt.scatter(piv_arr[:, 0, 0], piv_arr[:, 1, 0], s=5)
    plt.gca().set_aspect("equal")
    #ax.scatter(piv_arr[:, 0, 0], piv_arr[:, 1, 0], np.ones(len(piv_arr[:, 1, 0])))
    plt.savefig(f"./figs2/{title}{name}{height}.png")
    plt.clf()
    # plt.savefig("./fig.png")
    # plt.show()


def quartile_range(percent: "ndarray") -> "ndarray":
    "TODO."
    q1 = np.quantile(percent, 0.25)
    q3 = np.quantile(percent, 0.75)
    IQR = q3 - q1
    outliers_idx = (percent < (q1 - 1.5 * IQR)) | (percent > (q3 + 1.5 * IQR))
    return outliers_idx

def par(path, set_number, heights_array, y_shift_array, average_percent_arr, idy):
    for idx in range(12):
        path_file = f"{path}/H{set_number[idx]}/"
        number_files = find_number_files(path_file)
        # Final array is a 3d array with (points, data, time steps). The data is as follows; x,y,u,v,vel magnitude, vel degree
        data = read_csv(number_files, set_number[idx], path_file)
        pos_x_piv_min = min(data[:, 0, 0])
        pos_x_piv_max = max(data[:, 0, 0])
        sph_arr, low = read_csv_2(path)
        posy = min(sph_arr[:, 1])
        posx = min(sph_arr[:, 0])
        height = min(sph_arr[:, 2]) + heights_array[idx]
        shifted_data = np.copy(data)
        shifted_data[:, 1, :] += posy * y_shift_array[idy]
        shifted_data[:, 0] = shifted_data[:, 0] - pos_x_piv_min - (pos_x_piv_max - pos_x_piv_min) / 2
        # shifted_data[:, 0] += posx / 2
        closest_index_sph, shifted_height= closest_point_distance(shifted_data, sph_arr, height)
        percent, diff = subtract_plt(shifted_data, sph_arr, 4, 7, closest_index_sph)
        # plot_graph(data[:, 0, 0], data[:, 1, 0], data[:, 4, 0])
        # plot_graph(shifted_height[:, 0, 0], shifted_height[:,1,0], shifted_height[:,4,0])
        # plot_2_together(data, sph_arr, closest_index_sph)
        outliers_idx = quartile_range(percent)
        closest_pts = np.copy(sph_arr[closest_index_sph, :])
        # plot_2_together(shifted_height, closest_pts, outliers_idx, str(heights_array[idx]), str(y_shift_array[idy]), "outliers")
        # plot_graph(data[~outliers_idx,0,0], data[~outliers_idx,1,0], percent[~outliers_idx], str(heights_array[idx]), str(y_shift_array[idy]), "percent")
        # plot_histogram(percent, heights_array[idx], "histogram")
        average_percent = np.average(percent)
        # print(average_percent)
        average_percent_arr[idx, idy] = average_percent

    return average_percent_arr

def main() -> None:
    path = os.getcwd()
    # set_number = input("Which height is required (Hx)??: ")
    set_number = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # 12 fucking arrays...dont have to count all the time
    heights_array = [.105, .115, .125, .150, .160, .170, .200, .210, .220, .245, .255, .265]
    y_shift_array = np.linspace(1,0.7,10)
    average_percent_arr = np.zeros((len(heights_array), len(y_shift_array)))
    average_percent_arr = [pool.apply(par, args=(path, set_number, heights_array, y_shift_array, average_percent_arr, idy)) for idy in range(len(y_shift_array))]
    pool.close()
    # average_percent_y = par(path, set_number, heights_array, y_shift_array, average_percent_arr, idy)
    # for idy in range(len(y_shift_array)):
        # plot_2_together(shifted_height, sph_arr, None, heights_array[idx], y_shift_array[idy], "position")

    np.savetxt(f"./figs2/percent.csv",  average_percent_arr, delimiter=',')
    # np.savetxt(f"./figs2/percent.csv",  average_percent_arr, header = ([y_shift_array]),delimiter=',')
        # plot_graph(data[:,0,0], data[:,1,0], diff, heights_array[i], "difference")
        # print(data.shape)


if __name__ == "__main__":
    main()