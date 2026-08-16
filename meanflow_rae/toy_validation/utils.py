import os

from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card


def save_model_card(
    args,
    repo_id: str,
    repo_folder: str = None,
):
    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. \n

## Pipeline usage

You can use the pipeline like so:

```python
import torch
import os
import sys
import  numpy as np

import torch_xla.core.xla_model as xm
from time import time
from typing import Tuple
from diffusers import StableDiffusionPipeline

def main(args):
    device = xm.xla_device()
    model_path = <output_dir>
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    prompt = ["A naruto with green eyes and red legs."]
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("naruto.png")

if __name__ == '__main__':
    main()
```

## Training info

These are the key hyperparameters used during training:

* Steps: {args.max_train_steps}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))
