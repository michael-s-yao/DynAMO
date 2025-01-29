# Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)
[![CONTACT](https://img.shields.io/badge/contact-obastani%40seas.upenn.edu-blue)](mailto:obastani@seas.upenn.edu)

The goal of offline model-based optimization (MBO) is to propose new designs that maximize a reward function given only an offline dataset. However, an important desiderata is to also propose a *diverse* set of final candidates that capture many optimal and near-optimal design configurations. We propose **D**iversit**y** i**n** **A**dversarial **M**odel-based **O**ptimization (**DynAMO**) as a novel method to introduce design diversity as an explicit objective into any MBO problem. Our key insight is to formulate diversity as a *distribution matching problem* where the distribution of generated designs captures the inherent diversity contained within the offline dataset. Extensive experiments spanning multiple scientific domains show that DynAMO can be used with common optimization methods to significantly improve the diversity of proposed designs while still discovering high-quality candidates.

## Installation

To install and run our code, first clone the `DynAMO` repository.

```
cd ~
git clone https://github.com/michael-s-yao/DynAMO
cd DynAMO
```

Next, run the `setup.sh` Bash script to download relevant datasets, setup the Python environment, install relevant dependencies, and train task-specific forward surrogate models. Our `setup.sh` script runs without any modifications on a Linux machine running Debian 5.10.226-1 with CUDA version 12.4.

```
bash setup.sh
```

You can then run DynAMO using the provided [`main.py`](./main.py) Python script:

```
python main.py -t [TASK_NAME] -o [POLICY_NAME] -f [FORWARD_SURROGATE_MODEL_TRANSFORM_NAME] --seed [SEED]
```

Experimental results are automatically saved by default to the `./results` directory. For more information on script parameters and allowed values, you can run

```
python main.py --help
```

## Contact

Questions and comments are welcome. Suggestions can be submitted through GitHub issues. Contact information is linked below.

[Michael Yao](mailto:michael.yao@pennmedicine.upenn.edu)

[Osbert Bastani](mailto:obastani@seas.upenn.edu)

## Citation

If you found our work helpful for your research, please consider citing our paper:

    @misc{yao2025dynamo,
      title={Diversity By Design: {Leveraging} Distribution Matching for Offline Model-Based Optimization},
      author={Yao, Michael S and Gee, James C and Bastani, Osbert},
      journal={arXiv Preprint},
      year={2024},
    }

## License

This repository is MIT licensed (see [LICENSE](LICENSE)).
