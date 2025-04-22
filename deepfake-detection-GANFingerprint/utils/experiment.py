import os
import json
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_name, base_dir="experiments"):
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.abspath(base_dir) #Absolute paths for Experiment tracker to activate when running from Jupyter notebook
        self.experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
        
        # Create experiment directory
        try:
            os.makedirs(self.experiment_dir, exist_ok=True)
            print(f"Created experiment directory: {self.experiment_dir}")
        except Exception as e:
            print(f"Error creating experiment directory: {e}")
        
        self.config_file = os.path.join(self.experiment_dir, "config.json")
        self.results_file = os.path.join(self.experiment_dir, "results.json")
        self.log_file = os.path.join(self.experiment_dir, "log.txt")
        
        # Initialize the results structure
        self.results = {
            "epochs": [],
            "summary": {}
        }
        
        # Log initial information
        self.log(f"Experiment tracker initialized at {self.experiment_dir}")
        
    def save_config(self, config):
        """Save configuration to JSON file"""
        try:
            # Convert config object to dictionary
            if hasattr(config, '__dict__'):
                config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
            else:
                config_dict = config
                
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=4)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config: {e}")
            
    def log(self, message):
        """Log a message to the log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
            print(message)
        except Exception as e:
            print(f"Error writing to log file: {e}")
        
    def save_results(self, results):
        """Save results to JSON file"""
        try:
            # Determine if this is an epoch result or summary result
            if 'epoch' in results:
                # This is an epoch result, add to epochs list
                self.results["epochs"].append(results)
            else:
                # This is a summary result, update the summary dict
                self.results["summary"].update(results)
            
            # Save the entire results structure
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=4)
        except Exception as e:
            print(f"Error saving results: {e}")

# Test the ExperimentTracker if run directly
if __name__ == "__main__":
    print("Testing ExperimentTracker...")
    test_dir = "test_experiments"
    print(f"Creating test directory: {test_dir}")
    os.makedirs(test_dir, exist_ok=True)
    
    tracker = ExperimentTracker("test_experiment", base_dir=test_dir)
    tracker.log("This is a test message")
    tracker.save_config({"test_param": 123})
    tracker.save_results({"epoch": 1, "accuracy": 0.95})
    tracker.save_results({"final_accuracy": 0.97})
    
    print(f"Test complete. Check {tracker.experiment_dir} for files.")