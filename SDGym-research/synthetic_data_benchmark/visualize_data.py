import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import itertools

def load_data(dist):
    data = np.load(f"data/simulated/{dist}.npz")
    with open(f"data/simulated/{dist}.json", 'r') as f:
        meta = json.load(f)
    return data, meta

def plot_data(data, meta, dist, a, b, label=False):
    plt.figure(figsize=(12, 10))
    
    if label:
        # Plot the data points
        scatter = plt.scatter(data['train'][:, 0], data['train'][:, 1], 
                            c=data['train'][:, 2], cmap='coolwarm', alpha=0.6)
        
        # Add a color bar
        plt.colorbar(scatter)

        # Plot the boundary line
        x_range = np.array([data['train'][:, 0].min(), data['train'][:, 0].max()])
        plt.plot(x_range, a * x_range + b, 'g--', label=f'Boundary: y = {a}x + {b}')

    else:
        # Plot the data points
        plt.scatter(data['train'][:, 0], data['train'][:, 1], c='blue', alpha=0.6, label='Train data')
        plt.scatter(data['test'][:, 0], data['test'][:, 1], c='green', alpha=0.6, label='Test data')

    # Plot the 'mus' values
    if dist in ["grid", "gridr"]:
        mus = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2), range(-4, 5, 2))], dtype=np.float32)
        if dist == "gridr":
            mus += (np.random.rand(25, 2) - 0.5)
    elif dist == "ring":
        mus = np.array([[-1,0],[1,0],[0,-1],[0,1],[-np.sqrt(1/2),-np.sqrt(1/2)],
                        [np.sqrt(1/2),np.sqrt(1/2)],[-np.sqrt(1/2),np.sqrt(1/2)],
                        [np.sqrt(1/2),-np.sqrt(1/2)]])
    else:
        mus = np.array([])  # Empty array for other distributions

    plt.scatter(mus[:, 0], mus[:, 1], color='red', marker='x', s=100, label='Gaussian centers')

    plt.title(f'Visualization of {dist.capitalize()} Distribution')
    plt.xlabel(meta[0]['name'])
    plt.ylabel(meta[1]['name'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"data/simulated/{dist}_visualization.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize simulated data')
    parser.add_argument('distribution', type=str, help='specify type of distribution to visualize')
    parser.add_argument('--a', type=float, default=0, help='slope of the boundary line y = ax + b')
    parser.add_argument('--b', type=float, default=0, help='y-intercept of the boundary line y = ax + b')
    
    args = parser.parse_args()
    
    data, meta = load_data(args.distribution)
    plot_data(data, meta, args.distribution, args.a, args.b)