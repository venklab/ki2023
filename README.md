## Scripts used in Venklab 2023 Kidney International submission

Scripts for processing and analyzing slides. The trained machine learning model
is for working on 20x Aperio CS image at 0.5 micro per pixel.  Whole slide image
with different resolution won't be a problem, however, since scanners generate
different colors for the same slide, one may need to prepare additional set of
training data to retrain the model in order to achieve best result.



## Create conda env

It is not easy to duplicate our conda environment by install packages one by
one. Use following command to create an identical working environment:

```
conda create --name myenv --file conda-env-spec-file.txt
```

Few packages are not in conda, they need PIP:

```
# need install torch using pip. conda failed to solve for all versions tried
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu \
  -f https://download.pytorch.org/whl/cpu/torch_stable.html

pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

pip install openslide-python
```

## Build libutimage

It uses few functions from libutimage. Go to
`https://github.com/venklab/libutimage`. Clone that repository and build the
shared library, then create a symbolic link from `libutimage/utimage` to
`utimage` under the root.


## Segmenting slide


```
./process-slide-glm.py slide_file --inference --merge --tile \
    --model-weight path_to_trained_glomeruli_model

./process-slide-tub.py slide_file --inference --merge --tile \
    --model-weight path_to_trained_tubule_model
```

`process-slide-glm.py` and `process-slide-tub.py` are similar. They differ in
that it process the slide twice when segmenting tubules. In the second pass
every slide window is shifted few hundreds of pixels to improve the overall
tubule segmenting result.


## Analyze

`direct_visible_villin_dist.py` and `direct_glm_villin_dist.py` find and
measure immediate neighbors.

Since glomerulus and villin segmentations are stored in separated sqlite db.
Expected directory structure for slides are:

```
    -- upper_directory
        --glm
            slide and sqlite files
            ...

        --villin
            slide and sqlite files
            ...

```



