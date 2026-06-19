import csv
import os

class BenchmarkLogger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.records = []
        
    def print_header(self):
        print(f"\n=== {self.experiment_name} ===")
        print(f"{'Step':<4} | {'Ana TrLoss':<10} | {'Bas TrLoss':<10} | {'Ana TeLoss':<10} | {'Bas TeLoss':<10} | {'Ana TrAcc':<9} | {'Bas TrAcc':<9} | {'Ana TeAcc':<9} | {'Bas TeAcc':<9}")
        print("-" * 105)
        
    def log_step(self, step, ana_metrics, base_metrics, ana_test_metrics=None, base_test_metrics=None):
        """
        Logs and prints a single step's metrics side-by-side.
        metrics is a tuple: (loss, acc, f1)
        """
        a_loss, a_acc, a_f1 = ana_metrics
        b_loss, b_acc, b_f1 = base_metrics
        
        at_loss, at_acc, at_f1 = ana_test_metrics if ana_test_metrics else (0,0,0)
        bt_loss, bt_acc, bt_f1 = base_test_metrics if base_test_metrics else (0,0,0)
        
        print(f"{step:<4} | {a_loss:<10.4f} | {b_loss:<10.4f} | {at_loss:<10.4f} | {bt_loss:<10.4f} | {a_acc:<9.4f} | {b_acc:<9.4f} | {at_acc:<9.4f} | {bt_acc:<9.4f}")
        
        # Save record for CSV
        self.records.append({
            'experiment': self.experiment_name,
            'step': step,
            'ana_loss': a_loss,
            'base_loss': b_loss,
            'ana_acc': a_acc,
            'base_acc': b_acc,
            'ana_f1': a_f1,
            'base_f1': b_f1,
            'ana_test_loss': at_loss,
            'base_test_loss': bt_loss,
            'ana_test_acc': at_acc,
            'base_test_acc': bt_acc,
            'ana_test_f1': at_f1,
            'base_test_f1': bt_f1
        })
        
    def save_to_csv(self, filepath="benchmark_results.csv", append=True):
        mode = 'a' if append and os.path.exists(filepath) else 'w'
        write_header = mode == 'w'
        
        with open(filepath, mode, newline='') as f:
            fieldnames = ['experiment', 'step', 'ana_loss', 'base_loss', 'ana_acc', 'base_acc', 'ana_f1', 'base_f1',
                          'ana_test_loss', 'base_test_loss', 'ana_test_acc', 'base_test_acc', 'ana_test_f1', 'base_test_f1']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
                
            for record in self.records:
                writer.writerow(record)
