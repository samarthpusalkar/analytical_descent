import csv
import os

class BenchmarkLogger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.records = []
        
    def print_header(self):
        print(f"\n=== {self.experiment_name} ===")
        print(f"{'Step':<5} | {'Ana Loss':<9} | {'Base Loss':<9} | {'Ana Acc':<8} | {'Base Acc':<8} | {'Ana F1':<7} | {'Base F1':<7}")
        print("-" * 75)
        
    def log_step(self, step, ana_metrics, base_metrics):
        """
        Logs and prints a single step's metrics side-by-side.
        metrics is a tuple: (loss, acc, f1)
        """
        a_loss, a_acc, a_f1 = ana_metrics
        b_loss, b_acc, b_f1 = base_metrics
        
        print(f"{step:<5} | {a_loss:<9.4f} | {b_loss:<9.4f} | {a_acc:<8.4f} | {b_acc:<8.4f} | {a_f1:<7.4f} | {b_f1:<7.4f}")
        
        # Save record for CSV
        self.records.append({
            'experiment': self.experiment_name,
            'step': step,
            'ana_loss': a_loss,
            'base_loss': b_loss,
            'ana_acc': a_acc,
            'base_acc': b_acc,
            'ana_f1': a_f1,
            'base_f1': b_f1
        })
        
    def save_to_csv(self, filepath="benchmark_results.csv", append=True):
        mode = 'a' if append and os.path.exists(filepath) else 'w'
        write_header = mode == 'w'
        
        with open(filepath, mode, newline='') as f:
            fieldnames = ['experiment', 'step', 'ana_loss', 'base_loss', 'ana_acc', 'base_acc', 'ana_f1', 'base_f1']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
                
            for record in self.records:
                writer.writerow(record)
