The experimentation process for the traffic.py convolutional neural network
consisted of iteratively adding and removing components and layers
to see how they affect the accuracy and loss. Beginning first with a barebones
neural network containing only an initial convolutional layer, a flatten layer,
and a dense output layer.

The first experiment on the CNN was to test different values for the filters. It
was found that the highest accuracy occured when there were multiple convolutional
layers which had different values for filters. The CNN slightly improved with the
addition of an average pooling layer (moreso than max pooling).

At this point, the accuracy had plateued at approximately 95% and was performing
poorly on the test set. Adding an additional dense layer with a dropout before 
the output improved the performance on the test set; dropout layers on convolutional
and output layers proved to be insignificant at best and fatal to the network 
at worst.

The final addition which made a significant improvement on the existing CNN was the
addition of a second loop of convolutional and pooling layers. The second loop made
the largest improvement on the total accuracy when it had convolutional layers
with different filter values than the initial loop and a max pooling layer
rather than an average pooling.