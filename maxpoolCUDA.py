import cupy as np

class MaxPool2:
  # A Max Pooling layer using a pool size of 2.
  def __init__(self):
    # Initialize last_input attribute to store the input during forward pass.
    self.last_input = None

  def iterate_regions(self, image):
    '''
    Generates non-overlapping 2x2 image regions to pool over.
    - image is a 2d numpy array
    '''
    h, w, _ = image.shape
    new_h = h // 2
    new_w = w // 2

    for i in range(new_h):
      for j in range(new_w):
        im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
    '''
    h, w, num_filters = input.shape
    output = np.zeros((h // 2, w // 2, num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.amax(im_region, axis=(0, 1))

    return output
  
  def forward_batch(self, inputs):
    # inputs shape: (N, H, W, F)
    N, H, W, F = inputs.shape
    new_h, new_w = H // 2, W // 2
    reshaped = inputs.reshape(N, new_h, 2, new_w, 2, F)
    output = reshaped.max(axis=(2, 4))
    self.last_input_batch = inputs  # save for backprop if needed
    return output

  def backprop(self, d_L_d_out):
    # Vectorized backprop for 2x2 maxpool layer
    # Assume self.last_input was saved during forward pass.
    h, w, f = self.last_input.shape
    new_h, new_w = h // 2, w // 2
    # reshape input to (new_h, 2, new_w, 2, f)
    reshaped = self.last_input.reshape(new_h, 2, new_w, 2, f)
    # find the maximum per pooling region
    max_vals = reshaped.max(axis=(1, 3), keepdims=True)
    # Create a mask that marks the positions of the maximum values
    mask = (reshaped == max_vals)
    # Expand d_L_d_out into same shape as reshaped
    d_out_expanded = d_L_d_out.reshape(new_h, 1, new_w, 1, f)
    # Distribute the gradients to positions of max values
    d_L_d_input = mask * d_out_expanded
    # Reshape back to the original shape
    return d_L_d_input.reshape(h, w, f)
