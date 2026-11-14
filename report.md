# RNN experiment report · Which signal came first?

The RNN, LSTM and GRU each classified 256/256 sequences correctly on the standard test. When the number of neutral tokens after the second signal increased from 2–10 to 18–26, the RNN and GRU still scored 256/256 and the LSTM made one error. The experiment tests whether the final model state retains enough information to recover the order of two signals.

## Task and data

Each sequence contains exactly one `red` and one `blue` token. The label is the first signal. The following pair comes from the test set. Every neutral token stays in the same position.

| Input sequence | Label |
| --- | --- |
| `tap hum red ping hum ping blue hum tap tap` | Red first |
| `tap hum blue ping hum ping red hum tap tap` | Blue first |

The two sequences have identical token counts and opposite labels. A deterministic classifier using only token counts therefore gives the same prediction to both and gets exactly one right. This yields the 50% baseline.

| Dataset | Sequences | Pairs | Tokens after the second signal |
| --- | ---: | ---: | ---: |
| Training | 768 | 384 | 2–10 |
| Validation | 192 | 96 | 2–10 |
| Standard test | 256 | 128 | 2–10 |
| Long-tail test | 256 | 128 | 18–26 |

The generator places 0–2 neutral tokens before the first signal and 1–3 between the signals. It keeps each pair in one split and prevents duplicate sequences across splits. The long-tail test increases the delay between the last signal and the model’s prediction.

## Models and training

Each token maps to a learned eight-dimensional embedding. Sequences in a batch are padded to the same length. The recurrent layer skips padding because `pack_padded_sequence` receives the true lengths. A single recurrent layer with 16 hidden units reads the full sequence. A linear layer then maps the final hidden state to two label scores.

- **RNN** updates its hidden state using a tanh transformation of the current input and previous state.
- **LSTM** carries an additional cell state. Forget and input gates control how much memory it retains and writes. The output gate controls how much of that memory reaches the hidden state.
- **GRU** carries one hidden state. The reset gate affects the candidate state, and the update gate controls the mix of the candidate and previous state.

All three models ran on CPU with seed 27, a batch size of 48 and 10 epochs. Adam optimized cross-entropy with a learning rate of 0.008 and a gradient-norm limit of 1.0. The notebook measured validation loss after each epoch. All three models reached their lowest validation loss at epoch 10, and those weights were used on both test sets.

![Validation cross-entropy for the RNN, LSTM and GRU over 10 epochs](output/validation_loss.png)

## Results and interpretation

| Model | Trainable parameters | Standard test | Long-tail test |
| --- | ---: | ---: | ---: |
| Token-count baseline | 0 | 128/256 (50%) | 128/256 (50%) |
| RNN | 506 | 256/256 (100%) | 256/256 (100%) |
| LSTM | 1,754 | 256/256 (100%) | 255/256 (99.6%) |
| GRU | 1,338 | 256/256 (100%) | 256/256 (100%) |

![Standard and long-tail test accuracy compared with the token-count baseline](output/test_accuracy.png)

The sequence the LSTM missed began with `ping tap blue ping red`, followed by 22 neutral tokens. The correct label was “blue first” and the LSTM predicted “red first.” Because each color occurs exactly once, a model can infer the answer from either the first color or the last. Accuracy measures whether the model can recover order. It leaves open which cue each model learned to use.

These results come from one training run on synthetic data. At the same hidden width, the LSTM and GRU have more parameters than the RNN. A difference of one prediction on the long-tail test is too small to rank these architectures. The notebook also illustrates a bidirectional GRU and an encoder–decoder with untrained forward passes.

## Reproduce the run and find the figures

1. Open the [experiment notebook](Architectures-based-on-RNNs.ipynb) in Jupyter or VS Code. The recorded run used Python 3.10.11, PyTorch 2.14.1+cpu and Matplotlib 3.10.9.
2. Select a kernel with `torch`, `matplotlib` and `ipykernel`, then choose **Restart Kernel and Run All**. The notebook generates its own data.
3. Both figures appear below their plotting cells. The `save_result_figure` function also creates the `output` folder inside `RNN-Presentation` and saves the two PNGs used in this report. Rerunning a plotting cell updates its file under the same name. [Validation loss](output/validation_loss.png) · [Test accuracy](output/test_accuracy.png)
