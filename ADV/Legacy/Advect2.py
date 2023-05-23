import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors as NN

def dkernel(r,h):
    q = r/h
    if(q<2):
        qq = 1-q/2
        qr = qq*qq*qq *(-5*q)*(7/(4*np.pi*h*h*h))

        return qr
    else:
        return 0

def worker(args):
    i, particle_coords, vel_x, vel_y, kernelSize, dt, Conc, density, mass, NN_idx = args
    ConcTemp = Conc
    for k in (NN_idx[i]):
        if(particle_coords[i,0]<0.1 or particle_coords[i,0]>0.9):
            continue
        drx = particle_coords[i,0]-particle_coords[k,0]
        dry = particle_coords[i,1]-particle_coords[k,1]
        r = np.sqrt((drx)**2+(dry)**2)
        dker = dkernel(r,kernelSize)
        if(dker==0):
            continue
        absdrx = np.abs(drx)/(np.abs(drx)**2 + 1e-20)
        absdry = np.abs(dry)/(np.abs(dry)**2 + 1e-20)

        adv_v_ij = drx*vel_x*absdrx + dry*vel_y*absdry
        adv_v_ji = drx*vel_x*absdrx + dry*vel_y*absdry
        adv = dker * dt * adv_v_ij *  mass/density
        if(adv>0):
            # Particle i gives 
            ConcTemp[i] = Conc[i] - (adv * Conc[i])
        adv = dker * dt * adv_v_ji *  mass/density
        if(adv<0):
            # particle I gets
            ConcTemp[i] = Conc[i] - (adv * Conc[k])
    return ConcTemp[i]

def main():
    # Set the particle spacing and box size
    particle_spacing = 0.01
    # length, width
    box_size = (1, 0.3)

    # Calculate the number of particles in each dimension
    num_particles_x = int(box_size[0] / particle_spacing)
    num_particles_y = int(box_size[1] / particle_spacing)

    # Create a meshgrid of particle coordinates
    x_coords = np.linspace(0, box_size[0], num_particles_x)
    y_coords = np.linspace(0, box_size[1], num_particles_y)
    particle_coords = np.meshgrid(x_coords, y_coords)

    # Reshape the particle coordinates into a 2D array
    particle_coords = np.array(particle_coords).reshape(2, -1).T

    # Create the particle velocities (assume all particles have the same velocity)
    vel_x = 1
    vel_y = 0
    mass = 1
    density = 1000

    num_particles = len(particle_coords)
    # Create the particle masses
    Conc = np.zeros(num_particles)


    for i in range(num_particles):
        if(particle_coords[i,0]<0.3 and particle_coords[i,0]>0.2):
            Conc[i] = 1

    dt = 0.001
    t = np.arange(0,1,dt)
    kernelSize = particle_spacing*2
    
    # NEAREST NEIGHBORS
    kdt = NN(radius=kernelSize, algorithm='kd_tree').fit(particle_coords)
    distances,NN_idx = kdt.radius_neighbors(particle_coords)

    # Get the number of available cores
    num_cores = mp.cpu_count()
    # Calculate the number of cores to use
    num_processes = max(num_cores - 4, 1)
    print(num_particles)
    for s in t:
        with mp.Pool(num_processes) as pool:
            ranges = [(i, particle_coords, vel_x, vel_y, kernelSize, dt, Conc, density, mass, NN_idx) for i in range(num_particles)]
            Conc = pool.map(worker, ranges)
        
            middle_particles = []
            for i in range(num_particles):
                # middle_particles = np.logical_and(particle_coords[:, 1] > (box_size[1]/2 - particle_spacing), particle_coords[:, 1] < (box_size[1]/2 + particle_spacing))
                # middle_particles = np.logical_and(middle_particles, particle_coords[:, 0] > (box_size[0]/2 - particle_spacing))
                # middle_particles = np.logical_and(middle_particles, particle_coords[:, 0] < (box_size[0]/2 + particle_spacing))
                # middle_particles = np.where(middle_particles)[0]
                if abs(particle_coords[i, 1] - box_size[1]/2) < particle_spacing/2:
                    middle_particles.append(i)

            # Extract the concentration values for these particles
            middle_conc = Conc[middle_particles]

            # Extract the x coordinates for these particles
            middle_x = particle_coords[middle_particles, 0]
            print(middle_conc)

            # Make the plot
            plt.clf()
            plt.plot(middle_x, middle_conc)
            plt.xlabel('Position along x-axis')
            plt.ylabel('Concentration')
            plt.draw()
            plt.pause(0.01)


if __name__ == '__main__':
    main()
