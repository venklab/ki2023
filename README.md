## Scripts Used in Venklab 2023 Kidney International Submission

Scripts for processing and analyzing slides. The trained machine learning model
is for working with 20x Aperio CS images at 0.5 microns per pixel. While whole
slide images at different resolutions are compatible, please note that
different scanners generate different color profiles for the same slide.
Therefor it may be necessary to prepare an additional set of training data to
retrain the model for best results.


## Creating a Conda Environment

It is not easy to duplicate our conda environment by install packages one by
one.  Instead, use the following command to create an identical working
environment:


```
conda create --name myenv --file conda-env-spec-file.txt
```

Some packages are not available in conda and need to be installed using pip:

```
# install torch using pip. conda failed to solve for all versions tried
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu \
  -f https://download.pytorch.org/whl/cpu/torch_stable.html

pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

pip install openslide-python
```

## Build libutimage

It uses few functions from libutimage. Go to
`https://github.com/venklab/libutimage`, clone that repository, build the
shared library, and create a symbolic link from `libutimage/utimage` to
`utimage` in the root directory.


## Segmenting slide

Use the following commands to segment slides:

```
./process-slide-glm.py slide_file --inference --merge --tile \
    --model-weight path_to_trained_glomeruli_model

./process-slide-tub.py slide_file --inference --merge --tile \
    --model-weight path_to_trained_tubule_model
```

`process-slide-glm.py` and `process-slide-tub.py` are similar but differ in
that the latter processes the slide twice when segmenting tubules. In the
second pass, every slide window is shifted a few hundred pixels to improve the
overall tubule segmentation result.


## Analysis

`direct_visible_villin_dist.py` and `direct_glm_villin_dist.py` find and measure immediate neighbors.

Since glomerulus and villin segmentations are stored in separate SQLite
databases, the expected directory structure for slides should be as follows:

```
    -- upper_directory
        --glm
            slide and SQLite files
            ...

        --villin
            slide and SQLite files
            ...

```

