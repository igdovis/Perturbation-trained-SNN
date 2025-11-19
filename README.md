# Perturbation-trained-SNN


`experiment.py` is a script to run from the command line, with options to choose from the Randman dataset and SHD. It measures how similar is the base weight perturbation algorithm vs surrogate gradient descent, using multiple measurements and focusing on cosine similarity. Here, we use orthogonal noise sampling to perturb the parameters of a feedforward spiking neural network one by one, to approximate the true gradient. 

