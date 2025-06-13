import h5py
from gpu_utils import cp, xp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from wfnreader import WFNReader  # Ensure wfnreader is correctly implemented or installed
from scipy.ndimage import zoom
# This script selects ISDF sampling points via a weighted k-means algorithm.
# The density-driven clustering will remain relevant once the self-consistency
# loop is introduced, since new charge densities will require recomputing these
# centroids.

def weighted_kmeans_cupy(
    avec,
    rho_cp,
    N_k=10,
    t=20,
    max_steps=200,
    tolerance=5e-3 # slight problem in that 
):
    print("Starting weighted k-means clustering")
    """
    Perform weighted k-means clustering using CuPy with periodic boundary conditions (PBC) in 3D.

    Parameters:
    - avec (cp.ndarray): Lattice vectors (3x3 CuPy array where each row is a lattice vector in Cartesian coordinates).
    - rho (cp.ndarray): Charge density array (3D CuPy array with shape corresponding to the grid size).
    - N_k (int): Number of clusters.
    - t (int): Multiplicative factor for initial centroid candidates (default=20).
    - max_steps (int): Maximum number of iterations (default=2000).
    - tolerance (float): Convergence tolerance based on centroid movement (default=1e-2).

    Returns:
    - centroids_indices (cp.ndarray): Indices of the final centroids in the grid.
    - centroids (cp.ndarray): Coordinates of the final centroids in real space.
    - centroid_z_history (cp.ndarray): Array recording the z-components of centroids at each step.
    - steps_taken (int): Number of steps taken until convergence or reaching max_steps.
    """
    # Define grid sizes based on rho shape
    grid_size_x, grid_size_y, grid_size_z = rho_cp.shape

    # Create synthetic Gaussian data
    #x = cp.linspace(0, 1, grid_size_x)
    #y = cp.linspace(0, 1, grid_size_y)
    #z = cp.linspace(0, 1, grid_size_z)
    #X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    
    # Create a 3D meshgrid of x, y, z values
    X, Y, Z = cp.meshgrid(
        cp.linspace(0,1,grid_size_x),
        cp.linspace(0,1,grid_size_y),
        cp.linspace(0,1,grid_size_z),
        indexing='ij'
    )
    frac_positions = cp.stack((X, Y, Z), axis=-1)  # Shape: (grid_x, grid_y, grid_z, 3)
    positions = frac_positions.reshape(-1, 3)  # Shape: (num_points, 3)
    avec_inv = cp.linalg.inv(avec)

    # Create a Gaussian centered at (0.5, 0.5, 0.5)
    #sigma = 0.1  # Width of Gaussian
    #rho = rho_cp**2
    #rho = cp.exp(-((X-0.5)**2 + (Y-0.5)**2 + (Z-0)**2)/(2*sigma**2)) + cp.exp(-((X-0.5)**2 + (Y-0.5)**2 + (Z-1.)**2)/(2*sigma**2))


    # Replace the random initialization with k-means++
    # Initialize array to store centroids
    centroids_frac = cp.zeros((N_k, 3), dtype=cp.float32)
    
    # Choose first centroid randomly with probability proportional to density
    probs = rho.ravel() / rho.sum()
    first_idx = cp.random.choice(len(positions), size=1, p=probs)[0]
    centroids_frac[0] = positions[first_idx]
    
    print("17. Starting k-means++ initialization loop...")
    
    # Pre-allocate all arrays at maximum size
    delta_frac = cp.zeros((positions.shape[0], N_k, 3), dtype=cp.float32)
    delta_cartesian = cp.zeros_like(delta_frac)
    min_dist_sq = cp.zeros(positions.shape[0], dtype=cp.float32)
    probs = cp.zeros_like(min_dist_sq)
    rho_flat = rho_cp.ravel()
    
    batch_size = 5  # Number of centroids to select per iteration
    
    for k in range(1, N_k, batch_size):
        print(f"{k}", end=' ', flush=True)
        # Calculate distances to existing centroids
        curr_k = min(k + batch_size, N_k)  # Don't exceed N_k
        
        # Use only the portion we need with existing centroids
        delta_frac[:, :k, :] = positions[:, cp.newaxis, :] - centroids_frac[:k, cp.newaxis, :].transpose(1, 0, 2)
        delta_frac[:, :k, :] = delta_frac[:, :k, :] - cp.round(delta_frac[:, :k, :])
        delta_cartesian[:, :k, :] = cp.matmul(delta_frac[:, :k, :], avec)
        min_dist_sq[:] = cp.min(cp.sum(delta_cartesian[:, :k, :]**2, axis=2), axis=1)
        
        # Select batch_size new centroids
        for b in range(k, curr_k):
            probs[:] = min_dist_sq * rho_flat
            probs[:] = probs / probs.sum()
            next_idx = cp.random.choice(len(positions), size=1, p=probs)[0]
            centroids_frac[b] = positions[next_idx]
            
            # Update min_dist_sq with the new centroid if not the last one in batch
            if b < curr_k - 1:
                delta_frac[:, b:b+1, :] = positions[:, cp.newaxis, :] - centroids_frac[b:b+1, cp.newaxis, :].transpose(1, 0, 2)
                delta_frac[:, b:b+1, :] = delta_frac[:, b:b+1, :] - cp.round(delta_frac[:, b:b+1, :])
                delta_cartesian[:, b:b+1, :] = cp.matmul(delta_frac[:, b:b+1, :], avec)
                new_dist_sq = cp.sum(delta_cartesian[:, b:b+1, :]**2, axis=2)
                min_dist_sq[:] = cp.minimum(min_dist_sq, new_dist_sq[:, 0])

    # Initialize array to record z-components of centroids at each step
    centroid_z_history = cp.zeros((N_k, max_steps), dtype=cp.float32)

    # Initialize variable to track steps taken
    steps_taken = max_steps

    # Open the movement log file
    with open('max_centroid_movement.txt', 'w') as movement_file:
        movement_file.write("Step, Max_Movement\n")  # Header

        for step in range(max_steps):
            # Convert positions and centroids to Cartesian coordinates first
            positions_cart = cp.matmul(positions, avec)  # Shape: (P, 3)
            centroids_cart = cp.matmul(centroids_frac, avec)  # Shape: (K, 3)

            # Compute distance vectors in Cartesian coordinates
            delta_cart = positions_cart[:, cp.newaxis, :] - centroids_cart[cp.newaxis, :, :]  # Shape: (P, K, 3)

            # Convert to fractional coordinates for PBC
            delta_frac = cp.matmul(delta_cart, avec_inv)
            
            # Apply minimal image convention in fractional coordinates
            delta_frac = delta_frac - cp.round(delta_frac)
            
            # Convert back to Cartesian for final distances
            delta_cart = cp.matmul(delta_frac, avec)
            
            # Compute Euclidean distances
            distances = cp.linalg.norm(delta_cart, axis=2)  # Shape: (P, K)

            # Assign each point to the nearest centroid
            labels = cp.argmin(distances, axis=1)  # Shape: (P,)

            # Create a mask for each centroid
            mask = cp.equal(labels[:, cp.newaxis], cp.arange(N_k))  # Shape: (P, K)

            # Reshape rho to (P, 1) for broadcasting
            rho_flat = rho.ravel()[:, cp.newaxis]  # Shape: (P, 1)

            # Apply mask to rho to get weights for each centroid
            masked_rho = mask * rho_flat  # Shape: (P, K)

            # Multiply each delta_cartesian by the masked_rho
            weighted_positions = masked_rho[:, :, cp.newaxis] * delta_cart  # Shape: (P, K, 3)

            # Sum weighted positions for each centroid, convert to fractional coordinates
            sum_weighted_frac = weighted_positions.sum(axis=0)   # Shape: (K, 3)

            # Sum weights for each centroid
            sum_weights = masked_rho.sum(axis=0)  # Shape: (K,)

            # Avoid division by zero and do not reinitialize centroids with zero weight
            valid = sum_weights > 0
            new_centroids_frac = centroids_frac.copy()
            new_centroids_frac[valid] = centroids_frac[valid] + cp.matmul(sum_weighted_frac[valid] / sum_weights[valid, cp.newaxis], avec_inv)
            # Wrap fractional centroids to [0, 1)

            # Calculate centroid movement
            centroid_movement = cp.linalg.norm(new_centroids_frac - centroids_frac, axis=1)
            max_movement = cp.max(centroid_movement).get()  # Convert to CPU for logging
            movement_file.write(f"{step}, {max_movement}\n")  # Log the maximum movement
            
            new_centroids_frac = new_centroids_frac % 1.0

            # Record the z-components of centroids
            centroid_z_history[:, step] = new_centroids_frac[:, 2]

            # Check for convergence
            if cp.all(centroid_movement < tolerance):
                print(f"Converged in {step} steps.")
                steps_taken = step
                centroids_frac = new_centroids_frac
                break

            # Print every 10th step
            if step % 10 == 0:
                print(f"Step {step}")

            # Update centroids for next iteration
            centroids_frac = new_centroids_frac

        else:
            print(f"Reached max steps ({max_steps}) without full convergence.")

    # Convert final centroid fractional coordinates to Cartesian coordinates
    #centroids = cp.matmul(centroids_frac, avec)  # Shape: (K, 3)

    return labels, centroids_frac, centroid_z_history, steps_taken

if __name__ == "__main__":

    # Read 'avec' from file using WFNReader
    wfn = WFNReader("WFN.h5")  # Ensure 'WFN.h5' exists and is correctly formatted
    #wfn.avec = np.diag([1.,1.,1.])
    wfn.avec = wfn.avec
    avec = cp.asarray(wfn.avec, dtype=cp.float32)  # Shape: (3, 3)


    # First read the original density
    with h5py.File("charge_density.h5", "r") as h5f:
        rho_np = h5f['charge_density'][()]

    # Pad each dimension to even size if needed
    # shape = rho.shape
    # pad_width = [(0, shape[i] % 2) for i in range(3)]  # Calculate padding for each dimension
    # if any(p[1] for p in pad_width):  # If any padding is needed
    #     # Pad using edge values
    #     rho = cp.pad(rho, pad_width, mode='edge')
    #     print(f"Padded array from {shape} to {rho.shape}")

    # Now reshape the padded array
    # shape = rho.shape
    # new_shape = (shape[0]//2, 2, shape[1]//2, 2, shape[2]//2, 2)
    # rho = rho.reshape(new_shape).sum(axis=(1,3,5))


    # use "zoom factors" to interpolate density onto a coarser/faster grid
    # since the procedure is not very resolution-sensitive
    zoom_factors = (15/25, 15/25, 60/100)  # (0.6, 0.6, 0.6)
    #zoom_factors = (1,1,1)

    # Interpolate using spline interpolation
    # order=3 is cubic spline interpolation
    rho_interp = zoom(rho_np, zoom_factors, order=3)
    
    rho = cp.array(rho_interp, dtype=cp.float32)
    rho_np = rho.get()

    print(f"Original shape: {rho.shape}")  # (25, 25, 100)
    print(f"New shape: {rho_interp.shape}")  # (15, 15, 60)

    labels, centroids, centroid_z_history, steps_taken = weighted_kmeans_cupy(avec, rho, N_k=600)
    
    # Save fractional coordinates to file
    np.savetxt('centroids_frac.txt', 
               centroids.get(),  # Convert from cupy to numpy array
               header='x y z',   # Column headers
               fmt='%.6f',       # Format as float with 6 decimal places
               delimiter=' ',     # Space-separated
               comments='# ')    # Use # for header comment
    
    #rho = rho.get()
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Turn off grid and panes
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    ax.set_xlim(-1,3)
    ax.set_ylim(-1,3)
    ax.set_zlim(0,4)

    # Plot density points where rho > threshold
    threshold = 0.05 * np.amax(rho_np) #* 10 # remove 10 if charge density fixed
    print(f"Density threshold: {threshold}")
    print(f"Max density: {np.amax(rho_np)}")
    print(f"Min density: {np.amin(rho_np)}")
    
    X, Y, Z = np.meshgrid(
        np.linspace(0,1,rho_np.shape[0]),
        np.linspace(0,1,rho_np.shape[1]),
        np.linspace(0,1,rho_np.shape[2]),
        indexing='ij'
    )
    
    # Get points above threshold
    density_mask = rho_np > threshold
    print(f"Number of points above threshold: {np.sum(density_mask)}")
    
    density_points = np.stack([X[density_mask], Y[density_mask], Z[density_mask]], axis=1) @ wfn.avec
    density_values = rho_np[density_mask]
    print(f"Shape of density_points: {density_points.shape}")
    print(f"Range of density values: {density_values.min()} to {density_values.max()}")
    
    # Plot density points with color intensity based on density values
    scatter = ax.scatter(density_points[:, 0], 
                        density_points[:, 1], 
                        density_points[:, 2], 
                        c=np.log(np.abs(density_values)-0.9*threshold),  # Color by density
                        cmap='viridis',    # Choose a colormap
                        alpha=0.05,         # Much more opaque
                        s=20,              # Larger markers
                        marker='s',        # Square markers ('s' for square)
                        label='Density',
                        zorder=1)
    plt.colorbar(scatter, label='Charge Density')

    # # Get the masks for first two Voronoi cells
    # mask0 = (labels == 0)
    # mask1 = (labels == 1)
    
    # # Plot points in first Voronoi cell
    # voronoi0_points = positions[mask0].get()  # Convert to numpy for plotting
    # ax.scatter(voronoi0_points[:, 0], 
    #           voronoi0_points[:, 1], 
    #           voronoi0_points[:, 2],
    #           c='blue',
    #           alpha=0.5,
    #           s=1,
    #           label='Voronoi Cell 0')

    # # Plot points in second Voronoi cell
    # voronoi1_points = positions[mask1].get()  # Convert to numpy for plotting
    # ax.scatter(voronoi1_points[:, 0], 
    #           voronoi1_points[:, 1], 
    #           voronoi1_points[:, 2],
    #           c='green',
    #           alpha=0.5,
    #           s=1,
    #           label='Voronoi Cell 1')
    
    # Plot centroids
    centroids_np = centroids.get() @ wfn.avec
    ax.scatter(centroids_np[:, 0], 
              centroids_np[:, 1], 
              centroids_np[:, 2], 
              c='red', s=100, marker='*', label='Centroids',zorder=2)

    # Define vertices of unit cube in fractional coordinates
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ])
    
    # Transform to Cartesian coordinates
    vertices_cart = vertices @ wfn.avec
    
    # Define edges as pairs of vertex indices
    edges = [
        # Bottom face
        (0,1), (1,3), (3,2), (2,0),
        # Top face
        (4,5), (5,7), (7,6), (6,4),
        # Vertical edges
        (0,4), (1,5), (2,6), (3,7)
    ]
    
    # Plot each edge
    for start, end in edges:
        ax.plot([vertices_cart[start,0], vertices_cart[end,0]],
                [vertices_cart[start,1], vertices_cart[end,1]],
                [vertices_cart[start,2], vertices_cart[end,2]],
                'k-', linewidth=1)  # 'k-' means black solid line

    # Customize the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.title('Charge Density and Centroids')
    plt.show()
