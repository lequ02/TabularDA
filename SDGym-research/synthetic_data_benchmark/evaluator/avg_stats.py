import json
import os
from collections import defaultdict

def calculate_means(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each JSON file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            print(f"Reading {filename}")
            with open(os.path.join(input_dir, filename), 'r') as f:
                data = json.load(f)
            
            # Calculate the mean for each dataset
            means = defaultdict(list)
            for item in data:
                dataset = item['dataset']
                performance = item['performance'][0]  # assuming there's always one performance item
                syn_likelihood = performance['syn_likelihood']
                test_likelihood = performance['test_likelihood']
                means[dataset].append((syn_likelihood + test_likelihood) / 2)
            
            # Calculate the final mean for each dataset
            for dataset in means:
                means[dataset] = sum(means[dataset]) / len(means[dataset])
            
            # Write the means to a new JSON file in the output directory
            output_filename = filename.rsplit('.', 1)[0] + '_avg.json'
            with open(os.path.join(output_dir, output_filename), 'w') as f:
                json.dump(means, f, indent=4)

# Call the function with the input and output directories
calculate_means('./output/__result__/', './output/__result_avg/')  # replace 'output_directory' with your output directory


# (summer_research) PS D:\SummerResearch\SDGym-research> python .\synthetic_data_benchmark\evaluator\avg_stats.py