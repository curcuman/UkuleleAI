# UkuleleAI

- the recording folder contains the code that just sends the buffers of recording, with a character delimiter
- The inference folder contains the code and the model for making real time predictions
- The notebook contains the best version I got in trying with lab5 that got 90 percent
- The python files:
  - serial_client_i2s_pcm.py contains the orginal code from the lab
  - Receiving_splitting_pcm.py receives and creates the dataset in pcm
  - Receiving_splitting_wav.py receives and creates the dataset in wav (currently used)
  
-notice that in the confuson matrix, the chords that are similar are more confused


# UkuleleAI - Audio Project

This project involves deploying a neural network on the Nucleo-L476RG board, equipped with an STM32L476RGT6 microcontroller, and a custom audio capture shield. The shield consists of a TLV320ADC3101ADC and an AOM-5024L-HD-R microphone, as well as a 3.5” stereo input jack. Embedded-AI architectures are becoming increasingly important, enabling AI algorithms to be implemented on low-power devices like microcontrollers for tasks such as image and speech recognition, enhancing privacy, reducing latency, and improving efficiency.

## Goal

The objective is to develop a highly accurate neural network capable of classifying Ukulele chords and embedding it onto the Nucleo-L476RG board. The network is trained to classify seven chords: A, B, C, D, E, F, and G.

## Workflow
![Workflow_Diagram](https://github.com/curcuman/UkuleleAI/assets/93979835/386e15c5-91bc-4f92-b5df-61476ce1158a)

- Familiarization with training a neural network through lab experiences.
- Creation of a dataset by recording Ukulele chord samples using the shield's microphone.
- Training of a Convolutional Neural Network (CNN) model using the dataset.
- Processing architecture setup on the board using ADC, DMA, and I2S protocols.
- Live testing and evaluation of the model's performance.

## CNN Model

The model architecture is inspired by a publication on Neural Networks for Raw Waveforms. It includes:
- Increasing number of filters to capture complex patterns.
- Initial MaxPooling to reduce memory usage.
- Average Pooling to extract features.
- Adjustments in learning rate for optimization.
![image](https://github.com/curcuman/UkuleleAI/assets/93979835/d316379c-3e68-42da-ab29-abaeb4873b17)

## Processing Architecture

The shield captures audio data and sends it to the microcontroller for processing using ADC and DMA. I2S communication protocol is used to transmit and receive digital audio data. Neural network inference is performed on the audio data, and the predicted chord is outputted.

## Results and Performance Analysis
![image](https://github.com/curcuman/UkuleleAI/assets/93979835/2c7ff644-b205-4b45-83c8-be3ebb1445dc)

- Training accuracy: 93%
- Testing accuracy: 95%
- RAM footprint: 36,552 bytes
- Latency of inference: 295 milliseconds
- Estimated live prediction accuracy: 85%

The model's power consumption analysis shows an average of 62 mW in active mode and negligible in low-power mode, resulting in an overall average power consumption.
![image](https://github.com/curcuman/UkuleleAI/assets/93979835/76d9db55-f027-4acf-b544-38371fd12d7a)
