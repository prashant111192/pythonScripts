
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import statistics as stats
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
from typing import TYPE_CHECKING, Tuple
from numpy.lib.function_base import rot90
from scipy.spatial import KDTree, distance, cKDTree

if TYPE_CHECKING:
    from numpy import ndarray


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
    # remove outer boundary
    arr = arr[~(arr[:, 7] == 0)]
    # picked 0_x,1_y,2_z, 3_idp 4_vx, 5vy, 6_vz
    arr = vel(np.delete(arr, 7, 1))
    return arr


def loadAndVel(name_file: str) -> "ndarray":
    """TODO."""
    arr = np.loadtxt(name_file, dtype=float, delimiter=",", skiprows=3, usecols=(0, 1, 2, 3, 6))
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
    closest_index = distance.cdist(piv_arr[:, :3], sph_arr[:, :3], metric="euclidean").argmin(1)
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


def subtract_plt(piv_arr: "ndarray", sph_arr: "ndarray", vel_piv_index: int, vel_sph_index: int, closest_index_sph: "ndarray", vel) -> Tuple["ndarray", "ndarray"]:
    """TODO."""
    sub_array = abs((piv_arr[:, vel_piv_index]*0.006037) - vel[:])
    # sub_array = abs(piv_arr[:, vel_piv_index] - sph_arr[closest_index_sph, vel_sph_index])
    # sub_array = (piv_arr[:, vel_piv_index] - sph_arr[closest_index_sph, vel_sph_index])
    percent_arr = (sub_array * 100) / (piv_arr[:, vel_piv_index]*0.006037)
    vel_mean = stats.mean(sub_array)
    vel_stdev = stats.stdev(sub_array)
    # return percent_arr, sub_array, vel_stdev, vel_mean
    return percent_arr, sub_array 


def plot_histogram(c_: "ndarray", name: str, title: str) -> None:
    "TODO."
    # fig, (ax1, ax2) = plt.subplots(2)
    plt.boxplot(c_, showfliers=False)
    # ax2.boxplot(c_)
    # plt.hist(c_,)
    # plt.show()
    plt.title(f"{title}_height_{str(name)}")
    plt.ylim(0,200)
    plt.minorticks_on()
    plt.grid(b=True, which='major', linewidth='0.5')
    plt.grid(b=True, which='minor', linestyle=':', linewidth='0.5')
    # plt.subgrid()
    plt.savefig(f"./figs3/{title}_{str(name)}.png")
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

def plot_yz(piv_arr: "ndarray", sph_arr: "ndarray", name: str, height: str, title: str) -> None:
    """Plot yz Positions."""
    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = plt.axes(projection='3d')
    # ax = Axes3D(fig)
    ax.scatter(sph_arr[:,0], sph_arr[:,1], sph_arr[:,2], s = 0.01)
    ax.scatter(piv_arr[:,0], piv_arr[:,1],piv_arr[:,2], s = 0.1)
    # plt.gca().set_aspect("equal")
    plt.grid()
    # plt.savefig(f"./figs2/{title}{name}{height}.png")
    plt.show()
    plt.clf()


def plot_2_together(piv_arr: "ndarray", sph_arr: "ndarray", closest: "ndarray", name: str, height: str, title: str) -> None:
    """TODO."""
    # plt.scatter(sph_arr[:,0], sph_arr[:,1],marker = "." )
    # plt.scatter(sph_arr[closest,0], sph_arr[closest,1], marker= ",")
    # plt.scatter(piv_arr[:,0], piv_arr[:,1])
    fig, ax = plt.subplots(ncols=2)
    # ax.scatter(sph_arr[closest, 0], sph_arr[closest, 1], sph_arr[closest, 2], s=3)
    tanh_vector = np.vectorize(math.tanh)
    # # ax.scatter(sph_arr[:, 0], sph_arr[:, 1], sph_arr[:, 2], s=0.1)
    if closest is None:
        temp = tanh_vector(sph_arr[:,1]/sph_arr[:,0])
        im = ax[0].scatter(sph_arr[:, 0], sph_arr[:, 1], c= temp, s=10)
        # ax[0].ylim(-0.115,0.005)
        ax[0].set_ylim(ymin=-0.115,ymax=0.005)
        ax[0].set_xlim(xmin=-0.06,xmax=0.06)
        ax[0].set_aspect("equal")
        # fig[0].colorbar()
    else:
        temp = tanh_vector(sph_arr[closest,1]/sph_arr[closest,0])
        im = ax[0].scatter(sph_arr[closest, 0], sph_arr[closest, 1], c =temp, s=10)
        # ax[0].axis(xmin=-0.115,xmax=0.005)
        # ax[0].ylim(-0.115,0.005)
        ax[0].set_ylim(ymin=-0.115,ymax=0.005)
        ax[0].set_xlim(xmin=-0.06,xmax=0.06)
        ax[0].set_aspect("equal")
        # fig[0].colorbar()
    # # print(piv_arr[:, 2, 0])
    # # ax.scatter(piv_arr[:, 0, 0], piv_arr[:, 1, 0], piv_arr[:, 2, 0], s=1)
    temp = tanh_vector(piv_arr[:,1]/piv_arr[:,0])
    im  =ax[1].scatter(piv_arr[:, 0], piv_arr[:, 1], c=temp, s=5)
    ax[1].set_aspect("equal")
    # plt.clim(-1,1)
    plt.colorbar(im)
    #ax.scatter(piv_arr[:, 0, 0], piv_arr[:, 1, 0], np.ones(len(piv_arr[:, 1, 0])))
    plt.savefig(f"./figs2/{title})_{name}_{height}.png")
    plt.clf()
    # plt.savefig("./fig.png")
    # plt.show()


def box_plot(data, nO_y_shifts, xValues, title):
    # fig = plt.figure()
    # figure, ax = fig.add_axes([0,0,1,1])
    fig, ax = plt.subplots()
    # bp = ax.boxplot(data)
    ax.boxplot(data,  showfliers=False)
    plt.grid(axis='y')
    plt.xticks(np.arange(1,nO_y_shifts+1), (xValues))
    # plt.xticks(np.arange(1,nO_y_shifts+1), f"{xValues:.2f}", rotation=90)
    plt.title(title)
    plt.xlabel("Y Shifts")
    plt.ylabel("percentage error %")
    plt.savefig(f"./figs2/{title}.png")
    # plt.show()


def quartile_range(percent: "ndarray") -> "ndarray":
    "TODO."
    q1 = np.quantile(percent, 0.25)
    q3 = np.quantile(percent, 0.75)
    IQR = q3 - q1
    outliers_idx = (percent < (q1 - 1.5 * IQR)) | (percent > (q3 + 1.5 * IQR))
    return outliers_idx

def cKDTree_method(arr,h):
    tree =cKDTree(arr[:,0:2])
    nn_dist,index= tree.query(arr[:,0:2],k=10, distance_upper_bound=(2*h), workers=12)
    # nn_dist, index = dists[0][:,1]
    # tree = BallTree(arr1[:,0:2], leaf_size=50)  
    # nn_dist, index = tree.query_radius(arr1[:,0:2], r=2*h)
    return (nn_dist, index)

def vel_kernel(sph_arr, closest, index):

    vel = np.zeros(len(closest))
    # vel = np.average(sph_arr[index[closest],7])
    # print(index[closest[1]])
    # print(index.shape)
    for i in range(len(closest)):
        vel[i] = np.average(sph_arr[[index[closest[i]]],7])
        # print(vel.shape)
    return vel


def main_2(path, set_number, sph_y0, heights_array, y_shift_array, idx,sph_arr0, sph_arr1, sph_arr2, sph_arr3, sph_arr4, sph_arr5, sph_arr6, sph_arr7, index0, index1, index2, index3, index4, index5, index6, index7, h):
    average_percent_temp = np.zeros(8)
    # for idx in range(12):
    path_file = f"{path}/H{set_number[idx]}/"
    number_files = find_number_files(path_file)
    # Final array is a 3d array with (points, data, time steps). The data is as follows; x,y,u,v,vel magnitude, vel degree
    data = read_csv(number_files, set_number[idx], path_file)
    #Averaged timesteps
    data = np.mean(data,axis=2)
    #second Timestep
    # data = data[:,:,1]

    pos_x_piv_min = min(data[:, 0])
    pos_x_piv_max = max(data[:, 0])
    pos_y_piv_min = min(data[:, 1])
    # sph_arr, low = read_csv_2(path)
    # posy = min(sph_arr[:, 1])
    # print(posy)
    # posx = min(sph_arr[:, 0])
    height = sph_y0 + heights_array[idx]
    shifted_data = np.copy(data)
    shifted_data[:, 1] -= pos_y_piv_min - y_shift_array[idx]
    # print(pos_y_piv_min)
    # print(y_shift_array[idx])
    shifted_data[:, 0] = shifted_data[:, 0] - pos_x_piv_min - (pos_x_piv_max - pos_x_piv_min) / 2
    # shifted_data = shifted_data[]
    # shifted_data[:, 0] += posx / 2

    percent = np.zeros((len(data),8))
    for sph_id in range(8):
        if sph_id==0:
            sph_arr = sph_arr0
            # index = index0
            _, index = cKDTree_method(sph_arr0, h[sph_id])
        elif sph_id ==1:
            sph_arr = sph_arr1
            # index = index1
            _, index = cKDTree_method(sph_arr1, h[sph_id])
        elif sph_id ==2:
            sph_arr = sph_arr2
            # index = index2
            _, index = cKDTree_method(sph_arr2, h[sph_id])
        elif sph_id ==3:
            sph_arr = sph_arr3
            # index = index3
            _, index = cKDTree_method(sph_arr3, h[sph_id])
        elif sph_id ==4:
            sph_arr = sph_arr4
            # index = index4
            _, index = cKDTree_method(sph_arr4, h[sph_id])
        elif sph_id ==5:
            sph_arr = sph_arr5
            # index = index5
            _, index = cKDTree_method(sph_arr5, h[sph_id])
        elif sph_id ==6:
            sph_arr = sph_arr6
            # index = index6
            _, index = cKDTree_method(sph_arr6, h[sph_id])
        elif sph_id ==7:
            sph_arr = sph_arr7
            # index = index7
            _, index = cKDTree_method(sph_arr7, h[sph_id])
        closest_index_sph, shifted_height= closest_point_distance(shifted_data, sph_arr, height)
        
        vel_ave = np.zeros(len(closest_index_sph))
        vel_ave = vel_kernel(sph_arr, closest_index_sph, index)
        # print(len(vel_ave)-len(closest_index_sph))
        percent[:,sph_id], diff = subtract_plt(shifted_data, sph_arr, 4, 7, closest_index_sph, vel_ave)
        # plot_graph(data[:, 0, 0], data[:, 1, 0], data[:, 4, 0])
        # plot_graph(shifted_height[:, 0, 0], shifted_height[:,1,0], shifted_height[:,4,0])
        # plot_2_together(data, sph_arr, None, str(heights_array[idx]), str(y_shift_array[idx]), "check")
        # outliers_idx = quartile_range(percent)
        outliers_idx = quartile_range(percent[:,sph_id])
        # print(sum(outliers_idx))
        closest_pts = np.copy(sph_arr[closest_index_sph, :])
        # print(closest_pts.shape)
        # plot_2_together(shifted_height, closest_pts, None, str(heights_array[idx]), str(y_shift_array[idx]), "angle")
        # plot_2_together(shifted_height, sph_arr, None, str(heights_array[idx]), str(y_shift_array[idx]), "check")
        plot_2_together(shifted_height, closest_pts, outliers_idx, str(heights_array[idx]), str(sph_id), "outliers")
        # plot_graph(data[~outliers_idx,0,0], data[~outliers_idx,1,0], percent[~outliers_idx], str(heights_array[idx]), str(y_shift_array[idx]), "percent")
    
    # plot_histogram(percent, heights_array[idx], f"sph_histogram")

    # plot_yz(shifted_height, sph_arr, str(heights_array[idx]), str(y_shift_array[idx]), "2d_yz")
    average_percent = np.average(percent)
    average_percent_no_outliers = np.average(percent[~outliers_idx])
    average_stdev_no_outliers = np.average(percent[~outliers_idx])
    # print(average_percent.shape)
    # print(average_percent_no_outliers.shape)
    average_percent_temp = ([idx, heights_array[idx], average_percent, average_percent_no_outliers, average_stdev_no_outliers])
    # average_percent_temp[idx] = average_percent
    # average_percent_no_outliers_temp[idx] = average_percent_no_outliers
    # average_stdev_no_outliers_temp[idx] = average_stdev_no_outliers
    # print(average_percent_no_outliers_temp)

    # print (average_percent_temp.shape)
    # average_percent_return = average_percent_temp.reshape(1,len(average_percent_temp))
    # print (average_percent_return.shape)
    
    # average_percent_temp[len(heights_array):(len(heights_array)*2)] = average_percent_no_outliers_temp
    # average_percent_temp[len(heights_array)*2:] = average_stdev_no_outliers_temp

    return average_percent_temp