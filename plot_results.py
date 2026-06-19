import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_benchmark_results(csv_path="benchmark_results.csv", output_dir="plots"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # We will create a figure for each experiment
    experiments = df['experiment'].unique()

    for exp in experiments:
        exp_df = df[df['experiment'] == exp]

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Benchmark Results: {exp}', fontsize=16)

        # 1. Loss Plot
        axs[0].plot(exp_df['step'], exp_df['ana_loss'], label='Analytical', marker='o')
        axs[0].plot(exp_df['step'], exp_df['base_loss'], label='Baseline', marker='x')
        axs[0].set_title('Loss')
        axs[0].set_xlabel('Step')
        axs[0].set_ylabel('MSE Loss')
        axs[0].legend()
        axs[0].grid(True)

        # 2. Accuracy Plot
        axs[1].plot(exp_df['step'], exp_df['ana_acc'], label='Analytical', marker='o')
        axs[1].plot(exp_df['step'], exp_df['base_acc'], label='Baseline', marker='x')
        axs[1].set_title('Accuracy')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()
        axs[1].grid(True)

        # 3. F1-Score Plot
        axs[2].plot(exp_df['step'], exp_df['ana_f1'], label='Analytical', marker='o')
        axs[2].plot(exp_df['step'], exp_df['base_f1'], label='Baseline', marker='x')
        axs[2].set_title('F1 Score')
        axs[2].set_xlabel('Step')
        axs[2].set_ylabel('F1 Score')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        
        # Save figure
        clean_exp_name = exp.replace(" ", "_").lower()
        save_path = os.path.join(output_dir, f"{clean_exp_name}_results.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved plot for '{exp}' to {save_path}")

if __name__ == "__main__":
    plot_benchmark_results()
