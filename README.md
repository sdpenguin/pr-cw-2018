# patternrec-cw2

To start running distance metrics, please use main.py. Its help should contain all you need to undertand how to perform the experiments on the data.

``python main.py --help``

To run the various methods use:

``python main.py method [--rank RANK] [-cmc] [--title TITLE] [method_options]``

Note that there may be other mandatory arguments needed for the various methods used. The options are shown below alongside the method names. The syntax is: [--optional_argument [arg_values]], --mandatory_argument [mandatory_arg_values], mandatory_positional_argument.

Methods attempted:

* Baseline Euclidean Distance: ``euclid``
* Mahalanobis (various methods): ``mahala [--model [mmc, lmnn, nca, lfda, mlkr]]``
* k-means clustering: ``kmeans --clusters [num_clusters] --iterations [max_iterations]``
* kernel methods: ``kernel --kernel [cosine, poly, laplacian, chi2]``
* neural network: ``neuralnet --model [mlp, mlp-conv]``
* transformer methods: ``transformer --transformer [pca, quantile, normalizer, standard_scaler, gauss_rand] [--components n_components]``
* distance methods: ``distance --kernel [cosine, poly, laplacian, chi2]``

You can use ``--rank`` to reduce the required rank measurements, which will reduce computational complexity where possible.

You can optionally plot a graph of the results using the option ``-cmc`` (cumulative match curve). Its title can be specified with ``--title``.