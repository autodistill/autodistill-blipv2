import os
import platform
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from autodistill.classification import ClassificationBaseModel
from autodistill.detection import CaptionOntology
from PIL import Image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if device is not arm / Apple Silicon, install LAVIS from pip
# else install from source


@dataclass
class BLIPv2(ClassificationBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology

        if platform.processor() != "arm":
            subprocess.run("pip install salesforce-lavis", shell=True)
        elif not os.path.exists(f"{HOME}/.cache/autodistill/LAVIS"):
            installation_instructions = [
                "cd ~/.cache/autodistill/ && git clone https://github.com/salesforce/LAVIS",
                "cd ~/.cache/autodistill/LAVIS && pip install -r requirements.txt",
                "cd ~/.cache/autodistill/LAVIS && python setup.py build develop --user",
            ]
            for command in installation_instructions:
                subprocess.run(command, shell=True)

        if platform.processor() == "arm":
            sys.path.append(f"{HOME}/.cache/autodistill/LAVIS")

        from lavis.models import load_model_and_preprocess
        from lavis.processors.blip_processors import BlipCaptionProcessor

        model, vis_processors, _ = load_model_and_preprocess(
            "blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=DEVICE
        )

        self.model = model
        self.vis_processors = vis_processors
        self.tokenizer = BlipCaptionProcessor

    def predict(self, input: str) -> sv.Classifications:
        image = Image.open(input).convert("RGB")

        image = self.vis_processors["eval"](image).unsqueeze(0).to(DEVICE)

        classes = self.ontology.classes()

        cls_prompt = "is this image a " + " or a ".join(classes) + "?"

        sample = {"image": image, "text_input": cls_prompt}

        # generate model's response
        result = self.model.generate(**sample)

        # decode response
        class_id = classes.index(result)

        if class_id == -1:
            raise sv.Classifications()

        return sv.Classifications(
            class_id=np.array([class_id]),
            confidence=np.array(1),
        )
