# Spike Encoding Models for Auditory Analysis

This repository provides a collection of spike encoding models, auditory datasets, encoded spike data, trained models, and associated training and evaluation scripts for spike neural networks (SNNs). The models and datasets provided here can be used to explore different encoding strategies and evaluate the robustness of auditory perception under various noise conditions.

## Repository Structure

- **Encoding_model**: Contains four spike encoding models: 
  - Bioinspired Auditory Neuron (BAN)
  - Izhikevich (IZH)
  - Leaky Integrate-and-Fire (LIF)
  - Poisson 

  Additionally, a parameter configuration file for the BAN model is provided.

- **snndatabase**: This folder contains the following datasets for spike neural network research:
  - **TIDIGITS**: A dataset consisting of spoken digits.
  - **RWCP**: Real-world recordings for auditory perception.
  - **NOISEX-92**: A collection of noise datasets.

- **After_encoding_data**: Contains the encoded data using the BAN, IZH, LIF and Poisson spike encoding model applied to the **TIDIGITS** and **RWCP** datasets, respectively, with time steps set to 10.

- **snnmodel**: Includes trained SNN models with time steps of 10 for all four spike encoding models: **BAN**, **IZH**, **LIF**, and **Poisson**.

- **Model_training.py**: Python script for training models. You can specify parameters for spectrogram type, network type, and encoding model.
  - **Usage Example**:
    ```
    python Model_training.py --spec cqt --net snn --encoding BAN
    ```
  - **Parameters**:
    - `--spec`: Spectrogram type (cqt, mel, stft)
    - `--net`: Network type (snn, cnn, rnn)
    - `--encoding`: Encoding model (BAN, IZH, LIF, POISSON)

- **Evaluation_SNR.py**: Python script to evaluate SNR accuracy for different SNN models under different noise conditions.
  - **Usage Example**:
    ```
    python Evaluation_SNR.py --spec cqt --net snn --encoding BAN --snr SNR
    ```
  - **Parameters**:
    - `--spec`: Spectrogram type (cqt, mel, stft)
    - `--net`: Network type (snn, cnn, rnn)
    - `--encoding`: Encoding model (BAN, IZH, LIF, POISSON)
    - `--snr`: SNR

## Quick Start Guide

1. **Clone the repository**
    ```
    git clone https://github.com/yourusername/spike-encoding-auditory.git
    cd spike-encoding-auditory
    ```

2. **Dataset Preparation**
   - The **snndatabase** folder already contains pre-prepared datasets. Ensure these are in place before running encoding or training.

3. **Model Training**
   - Train models using `Model_training.py` by providing the desired configuration. For example:
     ```
     python Model_training.py --spec cqt --net snn --encoding BAN
     ```

4. **Evaluate SNR Performance**
   - Use `Evaluation_SNR.py` to assess how well a given model performs at different SNR levels. For instance:
     ```
     python Evaluation_SNR.py --spec cqt --net snn --encoding BAN --snr SNR
     ```

## Requirements
- Python 3.7 or later
- Required libraries:
  - PyTorch
  - NumPy
  - SciPy
  - Matplotlib


