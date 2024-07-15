import argparse
import itertools
import json
import math
import os
import sys
import pandas as pd

import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

from .. import utils

np.random.seed(0)

def create_distribution(dist_type, num_samples, a, b, label=False):
    if dist_type in ["grid", "gridr"]:
        samples, mus = make_gaussian_mixture(dist_type, num_samples)
    elif dist_type == "ring":
        samples, mus = make_gaussian_mixture(dist_type, num_samples, num_components=8)
    elif dist_type == "2rings":
        samples = make_two_rings(num_samples)
        mus = np.array([])  # No specific centers for 2rings
    
    if label:
        # Add the new column based on the boundary line
        labels = np.where(samples[:, 1] > a * samples[:, 0] + b, 1, 0)
        return np.column_stack((samples, labels)), mus

    return samples, mus

def make_gaussian_mixture(dist_type, num_samples, num_components=25, s=0.05, n_dim=2):
    """ Generate from Gaussian mixture models arranged in grid or ring
    """
    sigmas = np.zeros((n_dim,n_dim))
    np.fill_diagonal(sigmas, s)
    samples = np.empty([num_samples,n_dim])
    bsize = int(np.round(num_samples/num_components))

    if dist_type == "grid":
        mus = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                        range(-4, 5, 2))],dtype=np.float32)
    elif dist_type == "gridr":
        mus = np.array([np.array([i, j]) + (np.random.rand(2) - 0.5)
                        for i, j in itertools.product(range(-4, 5, 2),
                            range(-4, 5, 2))],dtype=np.float32)
    elif dist_type == "ring":
        mus = np.array([[-1,0],[1,0],[0,-1],[0,1],[-math.sqrt(1/2),-math.sqrt(1/2)],[math.sqrt(1/2),math.sqrt(1/2)],[-math.sqrt(1/2),math.sqrt(1/2)],[math.sqrt(1/2),-math.sqrt(1/2)]])

    for i in range(num_components):
        if (i+1)*bsize >= num_samples:
            samples[i*bsize:num_samples,:] = np.random.multivariate_normal(mus[i],sigmas,size=num_samples-i*bsize)
        else:
            samples[i*bsize:(i+1)*bsize,:] = np.random.multivariate_normal(mus[i],sigmas,size=bsize)
    return samples, mus

def make_two_rings(num_samples):
    samples, _ = make_circles(num_samples, shuffle=True, noise=None, random_state=None, factor=0.6)
    return samples

def visualize_data(samples, mus, dist, a, b, label=False):
    plt.figure(figsize=(12, 10))
    
    # Plot the data points
    if label:
        scatter = plt.scatter(samples[:, 0], samples[:, 1], c=samples[:, 2], cmap='cividis', alpha=0.6)
        # Add a color bar
        plt.colorbar(scatter)

        # Plot the boundary line
        x_range = np.array([samples[:, 0].min(), samples[:, 0].max()])
        plt.plot(x_range, a * x_range + b, 'g--', label=f'Boundary: y = {a}x + {b}')
    else:
        scatter = plt.scatter(samples[:, 0], samples[:, 1], c='blue', alpha=0.6, label='Train data')



    # Plot the 'mus' values
    if len(mus) > 0:
        plt.scatter(mus[:, 0], mus[:, 1], color='red', marker='x', s=100, label='Gaussian centers')

    plt.title(f'Visualization of {dist.capitalize()} Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"data/simulated/{dist}_visualization.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate simulated Data for a distribution')
    parser.add_argument('distribution', type=str, help='specify type of distributions to sample from')
    parser.add_argument('--sample', type=int, default=10000,
                    help='maximum samples in the simulated data.')
    parser.add_argument('--a', type=float, default=1.5,
                    help='slope of the boundary line y = ax + b')
    parser.add_argument('--b', type=float, default=.8,
                    help='y-intercept of the boundary line y = ax + b')
    args = parser.parse_args()
    dist = args.distribution
    num_sample = args.sample
    a = args.a
    b = args.b
    label = False
    samples, mus = create_distribution(dist, num_sample*2, a, b, label=label)
    np.random.shuffle(samples)

    output_dir = "data/simulated"
    try:
        os.mkdir(output_dir)
    except:
        pass

    # Store Meta Files
    meta = []
    for i in range(3):  # Now we have 3 columns
        if i < 2:
            meta.append({
                "name": f"feature_{i}",
                "type": "continuous",
                "min": float(np.min(samples[:,i])) - 1,
                "max": float(np.max(samples[:,i])) + 1
            })
        elif label:
            meta.append({
                "name": "label",
                "type": "categorical",
                "size": 2,
                "i2s": ["0", "1"]
            })

    # Store simulated data
    with open("{}/{}.json".format(output_dir, dist), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, dist), train=samples[:len(samples)//2], test=samples[len(samples)//2:])

    if label:
        df = pd.DataFrame(samples, columns=[meta[0]['name'], meta[1]['name'], meta[2]['name']])
    else:
        df = pd.DataFrame(samples, columns=[meta[0]['name'], meta[1]['name']])
    df.to_csv(f"{output_dir}/{dist}.csv", index=False)

    utils.verify("{}/{}.npz".format(output_dir, dist),
        "{}/{}.json".format(output_dir, dist))

    # Visualize the data
    visualize_data(samples, mus, dist, a, b)

    print(f"Array shape: {samples.shape}")
    print(f"First few rows:\n{samples[:5]}")
    print(f"Last few rows:\n{samples[-5:]}")
    print(f"Visualization saved as {dist}_visualization.png in {output_dir}")


# (summer_research) PS D:\SummerResearch\SDGym-research> python -m synthetic_data_benchmark.sdata_maker.bivariate "gridr"