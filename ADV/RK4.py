import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors as NN

def sumConc(Conc, particle_coords, vol):
    sumTemp = 0
    for i in range(len(Conc)):
        if(particle_coords[i,0]<0.1 or particle_coords[i,0]>0.9):
            continue
        sumTemp = sumTemp + Conc[i]*vol
    return sumTemp

def kernel(r,h):
    q = r/h
    if(q<2):
        qq = 1-q/2
        qr = qq*qq*qq*qq *(q+q+1)*(7/(4*np.pi*h*h))
        return qr
    else:
        return 0

def dkernel(r,h, dim):
    q = r/h
    if (dim==2):
        if(q<2):
            qq = 1-q/2
            qr = qq*qq*qq *(-5*q)*(7/(4*np.pi*h*h*h))
            return qr
        else:
            return 0
    else: 
        if (dim==1):
            if(q<2):
                qq = 1-q/2
                qr = qq*qq *q *(-15)/(8*h*h)
                return qr
            else:
                return 0

def compute(args):
    i, particle_coords, vel_x, vel_y, kernelSize, dt, Conc, density, mass, NN_idx, dim = args
    ConcTemp = Conc[i]
    # print("i: ", i)
    for k in (NN_idx[i]):
        if(particle_coords[i,0]<0.1 or particle_coords[i,0]>0.9):
            continue
        dr = particle_coords[i] - particle_coords[k]
        r = np.linalg.norm(dr)
        dker = dkernel(r,kernelSize, dim)
        # dker = 1
        if(dker==0):
            continue
        absdr = np.abs(dr)/(np.linalg.norm(dr)**2 + 1e-20)
        adv_v_ij = np.dot(dr, np.array([vel_x, vel_y])*absdr)
        adv_v_ji = np.dot(dr, np.array([vel_x, vel_y])*absdr)
        # adv = dker * dt * adv_v_ij 
        adv = dker *  adv_v_ij *  mass/density
        if(adv>0):
            # Particle i gives 
            ConcTemp = (adv * ConcTemp)
            # print("Gives, ker: ", dker, "adv: ", adv, "Conc: ", ConcTemp, "NN_idx: ", k)
        # adv = dker * dt * adv_v_ji 
        adv = dker * adv_v_ji *  mass/density
        if(adv<0):
            # particle i gets
            ConcTemp =(adv * Conc[k])
            # print("Gets, ker: ", dker, "adv: ", adv, "Conc: ", ConcTemp, "NN_idx: ", k)
    return ConcTemp


def main():
    # set dimensions
    dim = 1
    # Set the particle spacing and box size
    particle_spacing = 0.01
    # length, width
    if(dim==2):
        box_size = (1, 0.2)
    else:
        if(dim==1):
            box_size = (1, particle_spacing)

    # Calculate the number of particles in each dimension
    num_particles_x = int(box_size[0] / particle_spacing)
    num_particles_y = int(box_size[1] / particle_spacing)

    # Create a meshgrid of particle coordinates
    x_coords = np.linspace(0, box_size[0], num_particles_x)
    y_coords = np.linspace(0, box_size[1], num_particles_y)
    particle_coords = np.meshgrid(x_coords, y_coords)
    # find the middle point in y_coords
    tempy = len(y_coords)
    tempy = int(tempy/2)

    # Reshape the particle coordinates into a 2D array
    particle_coords = np.array(particle_coords).reshape(2, -1).T

    num_particles = len(particle_coords)
    # Create the particle velocities (assume all particles have the same velocity)
    vel_x = 0.2
    vel_y = 0
    vol = (box_size[0] * box_size[1])/num_particles
    density = 1000
    mass = density * vol 


    print("Number of particles: ", num_particles)
    # Create the particle masses
    Conc = np.zeros(num_particles)

    for i in range(num_particles):
        if 0.2 < particle_coords[i, 0] < 0.3:
            Conc[i] = 1

    dt = 0.001
    t = np.arange(0, 0.1, dt)
    kernelSize = particle_spacing * 2

    # NEAREST NEIGHBORS
    kdt = NN(radius=kernelSize, algorithm='kd_tree').fit(particle_coords)
    NN_idx = kdt.radius_neighbors(particle_coords)[1]

    # Get the number of available cores
    num_cores = mp.cpu_count()
    # Calculate the number of cores to use
    num_processes = max(num_cores - 2, 1)
    ## for line plot
    line_plot = False
    Flow = False
    if(Flow):
        plt.clf()
        plt.scatter(particle_coords[:,0], particle_coords[:,1], c=Conc)
        plt.title("Particle grid")
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.colorbar()
        plt.clim(0, 1)
        plt.grid()
        plt.draw()
        plt.pause(1.01)

    ConcTemp = Conc
    ConcTemp1 = Conc
    ConcTemp2 = Conc
    ConcTemp3 = Conc
    ConcTemp4 = Conc
    for s in t:
        # print("Time: ", s,"***************************************************")
        Conc_old = Conc
        sumTempOld = sumConc(Conc, particle_coords, vol)
        # with mp.Pool(1) as pool:
        with mp.Pool(num_processes) as pool:
            ranges = [(i, particle_coords, vel_x, vel_y, kernelSize, dt, Conc, density, mass, NN_idx, dim) for i in range(num_particles)]
            ConcTemp1 = pool.map(compute, ranges)

        with mp.Pool(num_processes) as pool:
            Conc_k2 = Conc +(dt/2)*np.array(ConcTemp1)
            ranges = [(i, particle_coords, vel_x, vel_y, kernelSize, dt, Conc_k2, density, mass, NN_idx, dim) for i in range(num_particles)]
            ConcTemp2 = pool.map(compute, ranges)

        with mp.Pool(num_processes) as pool:
            Conc_k3 = Conc +(dt/2)*np.array(ConcTemp2)
            ranges = [(i, particle_coords, vel_x, vel_y, kernelSize, dt, Conc_k3, density, mass, NN_idx, dim) for i in range(num_particles)]
            ConcTemp3 = pool.map(compute, ranges)

        with mp.Pool(num_processes) as pool:
            Conc_k4 = Conc +(dt)*np.array(ConcTemp3)
            ranges = [(i, particle_coords, vel_x, vel_y, kernelSize, dt, Conc_k4, density, mass, NN_idx, dim) for i in range(num_particles)]
            ConcTemp4 = pool.map(compute, ranges)

        Conc = Conc + (dt/6)*(np.array(ConcTemp1) + 2*np.array(ConcTemp2) + 2*np.array(ConcTemp3) + np.array(ConcTemp4))
        # Conc= np.array(ConcTemp)
        sumTempNew = sumConc(Conc, particle_coords, vol)
        Conc = (Conc*sumTempOld)/sumTempNew

        ConcPlot = Conc 
        temp_time = s*100 / dt 
        if(temp_time%1000==0):
        # if(temp_time%10000==0):
            middle_particles = []
            middle_particles = np.where(particle_coords[:, 1] == y_coords[tempy])[0]

            # Extract the concentration values for these particles
            middle_conc = ConcPlot[middle_particles]

            # Extract the x coordinates for these particles
            middle_x = particle_coords[middle_particles, 0]
            # print("Current time: {:.2f} s, Total Conc = {:.4f}".format(s, np.sum(Conc*vol)))
            # Make the FLOW plot
            if(Flow):
                plt.clf()
                plt.scatter(particle_coords[:,0], particle_coords[:,1], c=ConcPlot)
                plt.title("Particle grid, Time: {:.4f} s, Total Conc = {:.4f}".format(s+dt, np.sum(Conc*vol)))
                plt.xlabel("x position")
                plt.ylabel("y position")
                plt.colorbar()
                # plt.clim(0, 1)
                plt.grid()
                plt.draw()
                plt.pause(1.01)

            # Make the LINE plot
            if(line_plot):
                plt.clf()
                plt.plot(middle_x, middle_conc)
                plt.xlabel('Position along x-axis')
                plt.ylabel('Concentration')
                plt.title(label='Time: {:.4f} s, Total Conc = {:.4f}'.format(s+dt, np.sum(Conc*vol)))
                plt.ylim(0, 1.5)
                plt.grid(which='both')
                plt.draw()
                plt.pause(0.01)

    pos_max_idx = np.argmax(middle_conc)
    print("Position of maximum concentration: {:.15f}".format(middle_x[pos_max_idx]))

    plt.clf()
    plt.plot(middle_x, middle_conc, "x")
    plt.xlabel('Position along x-axis')
    plt.ylabel('Concentration')
    plt.title(label='Time: {:.2f} s, Total Conc = {:.4f}'.format(s, np.sum(Conc*vol)))
    plt.grid(which='both')
    plt.ylim(0, 1.5)
    plt.show()


if __name__ == '__main__':
    main()