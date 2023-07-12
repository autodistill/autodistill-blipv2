<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png?4"
      >
    </a>
  </p>
</div>

# Autodistill BLIPv2 Module

This repository contains the code supporting the BLIPv2 base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[BLIPv2](https://github.com/salesforce/LAVIS), developed by Salesforce, is a computer vision model that supports visual question answering and zero-shot classification. Autodistill supports classifying images using BLIPv2.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [BLIPv2 Autodistill documentation](https://autodistill.github.io/autodistill/base_models/blipv2/).

## Installation

To use BLIPv2 with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-blipv2
```

## Quickstart

```python
from autodistill_blip import BLIPv2

# define an ontology to map class names to our BLIPv2 prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = BLIPv2(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpeg")
```


## License

This project is licensed under a [3-Clause BSD license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!