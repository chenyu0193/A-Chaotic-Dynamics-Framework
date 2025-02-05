# A Chaotic Dynamics Framework Inspired by Dorsal Stream for Event Signal Processing
![fk](https://github.com/user-attachments/assets/cabecd49-6038-4056-b7b4-a8f816b89b63)
##  Highlights
• We propose an event stream processing framework inspired by the brain’s dorsal visual pathway. We introduce the spatial-temporal information encoding mechanism of the brain’s dorsal pathway, also known as the ”where” pathway, into the event stream data processing framework, effectively establishing a high-order mapping fromevent streams to event frames.
• This framework utilize CCNN to encode constant polarity eventsequences as periodic signals and varying-polarity event sequences as chaotic signals, effectively achieving robust event representation. Combined with traditional deep neural network, the frame work successfully performs in object classification for event cameras.
• The proposed framework is evaluated on multiple datasets, achieving the state-of-the-art accuracy on specific benchmarks. It also demonstrates competi tive performance across a variety of datasets. The results demonstrate the framework’s strong generaliza tion across different data structures.

## Installation
• Python 3.9+ and CUDA12.0
• PyTorch 2.4.1+ and corresponding torchvision

## Quick Start
with data:
   python datasets/N-CAR.py

with train:
 python demo/train.py

with test:
 python demo/test.py
