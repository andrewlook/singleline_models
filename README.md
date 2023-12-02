# Sketch RNN

[PyTorch](https://pytorch.org) implementation of the SketchRNN paper, [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477).

Sketch RNN learns to reconstruct stroke-based drawings, by predicting a series of strokes. It uses a sequence-to-sequence LSTM model, with gaussian mixture heads to produce a sequence of stroke coordinates.

![seq2seq model](https://camo.githubusercontent.com/a8fc717aec062f15a231e5f52adbf67f5894a7135516c3e222398e3500a0dc2b/68747470733a2f2f63646e2e7261776769742e636f6d2f74656e736f72666c6f772f6d6167656e74612f6d61696e2f6d6167656e74612f6d6f64656c732f736b657463685f726e6e2f6173736574732f736b657463685f726e6e5f736368656d617469632e737667)

### Datasets

- `data/quickdraw/`: Sample data from [Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- `data/look/`: Custom dataset of single-line drawings by @andrewlook

All data is stored in stroke-3 format, meaning a list with three columns:

- `delta_x`
- `delta_y`
- `lift_pen` (if `1`, "lift the pen" and start a new stroke; otherwise `0`)

![stroke-3 turtle](https://camo.githubusercontent.com/28ac7d05adf47e55b331a38074643f9aeff58a46f3c81058193e62971bdd6675/68747470733a2f2f63646e2e7261776769742e636f6d2f74656e736f72666c6f772f6d6167656e74612f6d61696e2f6d6167656e74612f6d6f64656c732f736b657463685f726e6e2f6173736574732f646174615f666f726d61742e737667)

### Acknowledgements

- [PyTorch Sketch RNN](https://github.com/alexis-jacq/Pytorch-Sketch-RNN) project by [Alexis David Jacq](https://github.com/alexis-jacq)
- [Annotated Sketch RNN in PyTorch](https://nn.labml.ai/sketch_rnn/index.html) by [LabML](https://nn.labml.ai/)
- [Tensorflow SketchRNN](https://github.com/magenta/magenta/blob/main/magenta/models/sketch_rnn/README.md) by [Magenta Team](https://magenta.tensorflow.org/) and [David Ha](https://github.com/hardmaru)
- [sketch-rnn-datasets](https://github.com/hardmaru/sketch-rnn-datasets) by [David Ha](https://github.com/hardmaru)
