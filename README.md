# AdversarialNeuralNormalization

This repo includes code to reproduce the result from Chapter 5 of the thesis [Investigating the Role of Biological Constraints in Adversarial Robustness via Modeling and Representational Geometry](https://dspace.mit.edu/handle/1721.1/139227). Specifically, the repo has the following components:

- Bash file `normalize_pipeline.sh` to set up the parameters for the experiments. Following parameters are included:
  -  `MODEL_NAME`: can be `standard`, `neural_noise` (inject Poisson noise after the first conv layer), `gaussian` (inject Gaussian noise after the first conv layer).
  -  `SEED`: set up random seed for reproducibility
  -  `MODES`: specify the pipeline mode. Can be `train` (train the model), `val` (validate the model performance), `extract` (extrat the model internal representation)
  -  `LEARNING_RATES`: set up the learning rate of the optimizer
  -  `WEIGHT_DECAYS`: set up weight decay of the optimizer
  -  `NORMALIZES`: specify the normalize mode `nn` (no normalization), `bn` (batch normalization), `in` (instance normalization), `gn` (group normalization), `lrnc` (Channel-wise Local Response Normalization), `lrns` (Spatial-wise Local Response Normalization), `lrnb` (Both channel-spatial-wise Response Normalization).
  -  `EPS`: The epsilon magnitude of the adversarial attacks used to train or evaluate the model.

- Bash file `normalize.sbatch` and `normalize.sh` to run experiments and submit jobs on the MIT OpenMind platform.
- Python file `mnist_normalize.py` includes code to train and evaluate adversarial performance of LeNet-similar model on MNIST.
- Python file `cifar_normalize.py` includes code to train and evaluate adversarial performance of ResNet-18 model on CIFAR-10. `cifar_layer_norm.py` includes code of the ResNet-18 model architecture.
