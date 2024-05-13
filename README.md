# GlitchHunter Glitch vs Gravitational Wave classifier 

Gravitational waves (GWs) are ripples in spacetime that travel at the speed of light, emanating from some of the most cataclysmic events in the universe. Predicted by Einstein's Theory of General Relativity, their detection not only confirms a fundamental aspect of our understanding of gravity but also opens a new window into observing cosmic phenomena. When massive objects like black holes or neutron stars spiral into each other and merge, they emit GWs, carrying information about their properties and the nature of gravity itself. However, the detection of these elusive waves is fraught with challenges. The signals are incredibly weak when they reach Earth, and the data collected by GW detectors, such as LIGO, are often contaminated with noise transients or "glitches" that can mimic the true GW signals. Differentiating between these glitches and real GW signals is crucial for astrophysical research and requires sophisticated analytical techniques.

This project build GlitchHunter, a glitch predictor  in the GW dettor data.  GlitchHunter uses Convolutional Neural Network (CNN) based deep learning model to detect the presence of glitches in GW wave.  CNN is a state of the art deep learning techniques that is very suitable for image recognition tasks and and working with massive data set, which is a requirement for the data coming out if GW detectors data stream. I used a novel approach of transoforming GW detector data using Singular Value Decomposition (SVD)vectors and using those to train and evaluae the CNN model.  The GlitchHunter is designed to process images representing time evolution graphs of GW signals, which are transformed into a format suitable for machine learning.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Instructions on how to install and set up your project.

## Usage

Guidelines on how to use your project, including code examples or screenshots.

## Contributing

Information on how others can contribute to your project.

## License

Specify the license under which your project is distributed.
