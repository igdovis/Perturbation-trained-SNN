import stork
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_randman(plot = True):
    dim_manifold = 1
    nb_classes = 10
    nb_inputs = 20
    nb_time_steps = 100
    step_frac = 0.5
    nb_samples = 1000
    nb_spikes = 1
    alpha = 1
    randmanseed = 42
    dt = 2e-3

    duration = nb_time_steps * dt

    data, labels = stork.datasets.make_tempo_randman(
        dim_manifold=dim_manifold,
        nb_classes=nb_classes,
        nb_units=nb_inputs,
        nb_steps=nb_time_steps,
        step_frac=step_frac,
        nb_samples=nb_samples,
        nb_spikes=nb_spikes,
        alpha=alpha,
        seed=randmanseed,
    )

    ds_kwargs = dict(nb_steps=nb_time_steps, nb_units=nb_inputs, time_scale=1.0)

    # Split into train, test and validation set
    datasets = [
        stork.datasets.RasDataset(ds, **ds_kwargs)
        for ds in stork.datasets.split_dataset(
            data, labels, splits=[0.8, 0.1, 0.1], shuffle=False
        )
    ]
    ds_train, ds_valid, ds_test = datasets
    if plot:
        fig, ax = plt.subplots(1, 5, figsize=(10, 1.5), dpi=150)
        for i in range(5):
            ax[i].imshow(np.transpose(ds_train[i][0]), cmap="binary", aspect="auto")
            ax[i].invert_yaxis()
            ax[i].set_xlabel("time step")
            ax[i].set_ylabel("Neuron Idx.")
            ax[i].set_title("label: " + str(ds_train[i][1]))

        plt.tight_layout()
        sns.despine()
        plt.show()
    return ds_train, ds_valid, ds_test