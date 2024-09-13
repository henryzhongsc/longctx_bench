import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import json
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='output/needle/')
args = parser.parse_args()

FILE_PATTERNS = ['output_config', '2x', '4x', '6x', '8x', '10x10x3']
APPROX_TOKENS = {
    'mamba_chat': '28K',
    'mamba': '28K',
    'rwkv': '27K',
    'recurrentgemma_2b': '27K',
    'recurrentgemma_9b': '27K',
    'llama': '27K',
    'longchat': '31K',
    'mistral': '30K'
}

def generate_plot(file_path):
    # Open and read the JSON file
    with open(file_path, 'r') as f:
        total_file = json.load(f)


    file = total_file['eval_results']['processed_results']
    plot_path = os.path.join('diagrams/', file_path.replace('.json', '.pdf'))

    background_len_wise_results = file['background_len_wise_results']
    overall_results = file['overall_results']

    keys_list = list(file.keys())
    keys_to_delete = keys_list[-2:]
    for key in keys_to_delete:
        del file[key]

    data = []
    for context_length in file.keys():
        for depth_lvl, results in file[context_length].items():
            if int(context_length) > 1000:
                ctx_length = str(int(context_length) // 1000) + 'K'
            else:
                ctx_length = str(np.round(int(context_length) / 1000, decimals=1)) + 'K'
            data.append({
                "Context Length": ctx_length,
                'Ctx_Length_Value': int(context_length),
                "Document Depth": np.round(float(depth_lvl), decimals=2),
                "Exact Match": results['exact_match']
            })


    df = pd.DataFrame(data)
    df = df.sort_values(by=['Ctx_Length_Value', "Document Depth"])

    print("------------")
    pivot_table = pd.pivot_table(df, values='Exact Match', index=['Document Depth', 'Ctx_Length_Value'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Ctx_Length_Value", values="Exact Match") # This will turn into a proper pivot
    print(pivot_table.iloc[:5, :5])

    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    data = sorted(data, key=lambda x : x['Ctx_Length_Value'])
    NoWords = sorted(list(set([i['Context Length'] for i in data])), key=lambda x : float(x[:-1]))
    NoDepths = sorted(list(set([i['Document Depth'] for i in data])))
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    ax = sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cbar_kws={'shrink': 0.8},
        cbar=False,
        cmap=cmap,
        vmin=0,
        vmax=1
    )

    plt.xlabel('Word Count', fontsize=30)  # X-axis label
    plt.ylabel('Depth', fontsize=30)  # Y-axis label
    approx_max_tokens = 0
    for k, v in APPROX_TOKENS.items():
        if k in plot_path:
            approx_max_tokens = f'20K words ≈ {v} tokens'
            break
    plt.suptitle(approx_max_tokens, fontsize=20, x = '0.665', ha='right')
    plt.xticks([i + 0.5 for i in range(len(NoWords))], NoWords, rotation=45, fontsize=30)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=30)  # Ensures the y-axis labels are horizontal
    plt.gca().set_aspect('equal')
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Draw white lines to separate the cells
    for i in range(len(NoWords) - 1):
        plt.axvline(i + 1, color='white', linewidth=2) 

    for i in range(len(NoDepths) - 1):
        plt.axhline(i + 1, color='white', linewidth=2)  # Horizontal lines

    # Show the plot
    if 'output_config.pdf' in plot_path:
        plot_path = plot_path.replace('/output_config.pdf', '.pdf')
    if not os.path.exists('/'.join(plot_path.split('/')[:-1])):
        os.makedirs('/'.join(plot_path.split('/')[:-1]))
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)

def list_json_files(root_dir):
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            has_pattern = False
            for pattern in FILE_PATTERNS:
                if pattern in filename:
                    has_pattern = True
            if has_pattern and '.json' in filename:
                relative_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
                json_files.append(relative_path)
    return json_files

def generate_all_plots(dir):
    files = list_json_files(dir)
    for file in files:
        generate_plot(file_path=file)

generate_all_plots(args.output_dir)