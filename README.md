# Jasper: An End-to-End Convolutional Neural Acoustic Model
  
PyTorch implementation of [Jasper: An End-to-End Convolutional Neural Acoustic Model (Jason Li et al., 2019)](https://arxiv.org/pdf/1904.03288.pdf).
  
<img src="https://media.arxiv-vanity.com/render-output/3770675/JasperVerticalDR_3.png" height=500>
  
Jasper (Just Another SPEech Recognizer) is a end-to-end convolutional neural acoustic model. Jasper uses only 1D convolutions, batch normalization, ReLU, dropout, and residual connections, but has shown powerful performance. This repository contains only model code, but you can train with jasper with [this repository](https://github.com/sooftware/KoSpeech).   
I appreciate any kind of feedback or contribution.  
  
## Usage  
  
```python
import torch
from jasper import Jasper

BATCH_SIZE, SEQ_LENGTH, DIM = 3, 14321, 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(BATCH_SIZE, SEQ_LENGTH, DIM).to(device)  # BxTxD
input_lengths = torch.LongTensor([SEQ_LENGTH, SEQ_LENGTH - 10, SEQ_LENGTH - 20]).to(device)

# Jasper 10x3 Model Test
model = Jasper(num_classes=10, version='10x5', device=device).to(device)
output, output_lengths = model(inputs, input_lengths)

# Jasper 5x3 Model Test
model = Jasper(num_classes=10, version='5x3', device=device).to(device)
output, output_lengths = model(inputs, input_lengths)
```
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/Jasper-pytorch/issues) on Github or   
contacts sh951011@gmail.com please.
  
We appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
### Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
  
## Reference
- [Jasper: An End-to-End Convolutional Neural Acoustic Model (Jason Li et al., 2019)](https://arxiv.org/pdf/1904.03288.pdf)
- [NVIDIA/DeepLearningExample](https://github.com/NVIDIA/DeepLearningExamples)
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com
