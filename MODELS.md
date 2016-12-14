# neural_walker.jl

Implementation of neural walker

# model01.jl

* Encoder - Decoder (1 Layer)
* Spatial
  * relu . conv

# model02.jl

* Encoder - Decoder (2 Layers)
* Dropout
  * After embedding
  * Between lstm layers
  * Between final lstm layer and softmax (decoder)
* Spatial
  * relu . conv

# model03.jl
* Encoder - Decoder (2 Layers)
* Dropout
  * After embedding
  * Between lstm layers

* Spatial
  * relu . conv

# model04.jl

* Encoder - Decoder (2 Layers)
* Dropout
  *  After embedding
  *  Between lstm layers
  *  Between final lstm layer and softmax (decoder)
* Spatial
  1.  sigm . conv
  2.  sigm . conv

# model05.jl

* Encoder - Decoder (2 Layers)
* Dropout
  * After embedding
  * Between lstm layers
  * Between final lstm layer and softmax (decoder)
* Spatial
  1.  relu . conv
  2.  sigm . conv

# model06.jl

* Encoder - Decoder (2 Layers)
  * Decoder takes the final hidden layer of the encoder in each step
* Dropout
  * After embedding
  * Between lstm layers
  * Between final lstm layer and softmax (decoder)
* Spatial
  1. relu . conv
  2. sigm . conv

# model07.jl

* Encoder - Decoder (2 Layers)
  * Decoder takes the final hidden layer of the encoder in each step
* Rich Softmax
  * Takes the hidden and the perceptual input at time t
* Dropout
  * After embedding
  * Between lstm layers
  * Between final lstm layer and softmax (decoder)
* Spatial
  1. relu . conv
  2. sigm . conv

# model08.jl

* Encoder - Decoder (2 Layers)
  * Decoder takes the final hidden layer of the encoder in each step
* Dropout
  * After embedding
  * Between lstm layers
  * Between final lstm layer and softmax (decoder)
* Spatial
  1. relu . conv
  2. relu . conv

# model09.jl

* Encoder - Decoder
  * Decoder takes the final hidden layer of the encoder in each step
* Rich Softmax
  * Takes the hidden and the perceptual input at time t
* Dropout
  * After embedding
  * Between final lstm layer and softmax (decoder)
* Spatial
  1. relu . conv
  2. relu . conv
  3. sigm . conv

# model10.jl

* Encoder - Decoder
  * Decoder takes the final hidden layer of the encoder in each step
* Rich Softmax
  * Takes the hidden and the perceptual input at time t
* Dropout
  * After embedding
  * Between final lstm layer and softmax (decoder)
* Spatial
  1. relu . conv
  2. relu . conv
  3. relu . conv
  4. sigm . conv

# model11.jl

* Encoder - Decoder
  * Decoder takes the final hidden layer of the encoder in each step
* Attention
  * Based on encoder hiddens
* Rich Softmax
  * Takes the hidden and the perceptual input at time t
* Dropout
  * After embedding
  * Between final lstm layer and softmax (decoder)
* Spatial
  1. relu . conv
  2. relu . conv
  3. sigm . conv

