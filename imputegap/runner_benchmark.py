from imputegap.recovery.benchmark import Benchmark

my_algorithms = ["MeanImpute", "SoftImpute", "DeepMVI"]
# my_algorithms = ["DeepMVI"]

my_opt = ["default_params"]

# my_datasets = ["eeg-alcohol"]
# my_datasets = ["airq","chlorine"]
my_datasets = ["gas-drift", "appliances"]


# my_patterns = ["mcar"]
my_patterns = ["mcar"]

# range = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
range = [0.05, 0.1]

# my_metrics = ["*"]
my_metrics = ["RMSE", "RUNTIME", "RUNTIME_LOG"]

# launch the evaluation
bench = Benchmark()
bench.eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range, metrics=my_metrics, optimizers=my_opt)