## Recursive-Filtering
This is a C++ class library for applying a recursive filter by filtering matrix of samples and cascading second order sections to realize both computational efficiency and filter's accuracy.

<!-- INSTALLATION -->
## Installation

<!-- USAGE -->
## Usage
* Solve for the round error issue happening at building higher order recursive filter in direct form.

<img src="https://github.com/Haotian-RA/recursive-filtering-2-24/blob/main/figures/round_error_zp_plot.png?raw=true" width="300" /> 
the cascaded form of recursive filter re-match up with the desired poles making the system stable.

* Filter matrix of samples rather than vector or scalar realizing more efficient computations.
<img src="https://github.com/Haotian-RA/recursive-filtering-2-24/blob/main/figures/real_time_filtering.png?raw=true" width="300" /> 
filtering matrix of samples saves almost double time than vector of samples, 4 times than scalar.

<!-- LICENSE -->
## License
See [LICENSE.txt](https://github.com/Haotian-RA/recursive-filtering-code/blob/main/LICENSE) for more information.