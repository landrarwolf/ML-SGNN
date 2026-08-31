# Experiment configuration

The training entry point reads one INI file per dataset and label rate. The default naming convention is `<labels-per-class><dataset>.ini`, such as `20citeseer.ini`.

The original experiment configuration files are not included in the current public release. A complete file must define the following keys:

```ini
[Model_Setup]
epochs =
lr =
weight_decay =
k =
nhid1 =
nhid2 =
dropout =
beta =
theta =
no_cuda =
no_seed =
seed =

[Data_Setting]
n =
fdim =
class_num =
structgraph_path =
featuregraph_path_1 =
featuregraph_path_2 =
featuregraph_path_3 =
ppmi_path =
feature_path =
label_path =
test_path =
train_path =
```

Paths may be absolute or relative to the directory from which `main.py` is executed. To use a nonstandard filename or location, pass it with `python main.py --config /path/to/config.ini`.
