import numpy as np
import matplotlib.pyplot as plt
import pool

def dkernel(r,h):
    q = r/h
    if(q<2):
        qq = 1-q/2
        qr = qq*qq*qq *(-5*q)*(7/(4*np.pi*h*h*h))

        return qr
    else:
        return 0



def main():
    # Define the size of the grid
    grid_size = 10 # for a 10x10 grid
    num_particles = 100 # number of particles in the grid

    # Calculate the grid spacing
    grid_spacing = 1.0 / (grid_size - 1)

    # Create the particle positions
    x, y = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), indexing='xy')

    # Create the particle velocities (assume all particles have the same velocity)
    vel_x = 1
    vel_y = 0
    mass = 1
    density = 1000

    # Create the particle masses
    Conc = np.zeros(num_particles)
    Conc[25] = 1
    ConcTemp = Conc

    # Reshape the positions and masses into 1D arrays
    x = x.reshape(num_particles)
    y = y.reshape(num_particles)

    dt = 0.1
    t = np.arange(0,100,dt)
    kernelSize = grid_spacing*2
    
    for s in t:
        for i in range(len(x)):
            # use pool to parallelize the next loop
            for k in range(len(x)):
                drx = x[i]-x[k]
                dry = y[i]-y[k]
                absdrx = np.abs(drx)/(np.abs(drx)**2 + 1e-20)
                absdry = np.abs(dry)/(np.abs(dry)**2 + 1e-20)

                adv_v_ij = drx*vel_x*absdrx + dry*vel_y*absdry
                adv_v_ji = drx*vel_x*absdrx + dry*vel_y*absdry
                r = np.sqrt((drx)**2+(dry)**2)
                dker = dkernel(r,kernelSize)
                adv = dker * dt * adv_v_ij *  mass/density
                if(adv>0):
                    ConcTemp[i] = Conc[i] - adv
                adv = dker * dt * adv_v_ji *  mass/density
                if(adv<0):
                    ConcTemp[i] = Conc[i] - adv
        Conc=ConcTemp
            
                    

    # Plot the particle positions
    plt.scatter(x, y, c=Conc)
    # plt.scatter(x, y, c=masses*1000, alpha=0.5)
    plt.title("Particle grid")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()
