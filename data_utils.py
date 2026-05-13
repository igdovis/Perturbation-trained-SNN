import os
import torch
import stork.datasets as datasets

import tonic
import tonic.transforms as ttf
from torchvision import transforms as tvt
import numpy as np
from pathlib import Path

def generate_randman(
    dim_manifold=1,
    nb_classes=10,
    nb_inputs=20,
    nb_time_steps=100,
    step_frac=0.5,
    nb_samples=1000,
    nb_spikes=1,
    alpha=1,
    randmanseed=42,
    dt=2e-3,
    plot=True,
):
    duration = nb_time_steps * dt

    data, labels = datasets.make_tempo_randman(
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
    datasets_split = datasets.split_dataset(
        data, labels, splits=[0.8, 0.1, 0.1], shuffle=False
    )
    datasets_ras = [
        datasets.RasDataset(ds, **ds_kwargs)
        for ds in datasets_split
    ]
    return datasets_ras

def load_shd_dataset(args):
    """Load Spiking Heidelberg Digits dataset via tonic and return [train, valid, test], input_dim, output_dim.
    Each sample: x shape (T, 700), y in [0..19].
    """
    print("Loading SHD dataset via tonic .. . . . ..")


    data_root = Path(__file__).resolve().parent / args.data_dir
    data_root = data_root.resolve()
    save_to = str(data_root / "shd")  # or just str(data_root) depending on your structure
    print("SHD save_to resolved:", save_to)


    # (W, H, P)
    sensor_size = tonic.datasets.SHD.sensor_size
    input_dim = sensor_size[0] * sensor_size[1] * sensor_size[2] 
    output_dim = 20  # SHD has 20 classes

    event_transform = ttf.Compose([
        ttf.ToFrame(sensor_size=sensor_size, n_time_bins=args.num_steps),
        tvt.Lambda(
            lambda frames: torch.from_numpy(frames).float()
                            .view(frames.shape[0], -1)  # (T, input_dim)
        ),
    ])
    print("input dim:", input_dim, "output dim:", output_dim)

    train_ds = tonic.datasets.SHD(
        save_to=save_to,
        train=True,
        transform=event_transform,
        target_transform=None,
    )
    test_full = tonic.datasets.SHD(
        save_to=save_to,
        train=False,
        transform=event_transform,
        target_transform=None,
    )

    # Split test into validation + test 
    test_size = len(test_full)
    valid_size = test_size // 2
    test_size  = test_size - valid_size

    valid_ds, test_ds = torch.utils.data.random_split(
        test_full,
        [valid_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    return [train_ds, valid_ds, test_ds], input_dim, output_dim

def get_dataset(args):
    """Get dataset based on args"""
    if args.dataset == 'randman':
        datasets_list = generate_randman()
        input_dim = 20  
        output_dim = 10  
        return datasets_list, input_dim, output_dim
    elif args.dataset == 'shd':
        return load_shd_dataset(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
