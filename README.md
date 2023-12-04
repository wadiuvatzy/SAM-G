# Adroit Instruction

To get started, you should download adroit demos, pretrained models, and example logs with the following link to create `vrl3data` folder: https://drive.google.com/drive/folders/14rH_QyigJLDWsacQsrSNV7b0PjXOGWwD?usp=sharing. In addtion, you should change the `local_data_dir` in each `cfg/config.yaml` to the path of `vrl3data`.

## Extra Datasets
Some algorithms will use the [Places](http://places2.csail.mit.edu/download.html) dataset for data augmentation, which can be downloaded by running
```
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
```
After downloading and extracting the data, add your dataset directory to the datasets list in `cfgs_adroit/aug_config.cfg`.

Since we will use a pretrained model of efficientViT for segmentation, we need to download a [l1-checkpoint](https://drive.google.com/file/d/1ji6NcDfZF8b2kkFn9DolnbaOGSqklECe/view) and place it in `efficientvit/checkpoints/`


## Setup
- Create a conda env:
    ```
    conda create -n sam-g-adroit python=3.8
    ```
- run the `install_adroit.sh`:
    ```
    bash install_adroit.sh
    ```

## Config file
To use SAM-G, you need to modify the config files in `src/cfgs_adroit/{ALGORITHM}_config.yaml`.

- use_SAM_g: set to true to use SAM-G, making the RL agent receives segmented observations instead of using original observations.
- original_image: the original image you pick.
- masked_image: the corresponding segmented image of the original image.
- extra_point_list: the list of extra points given to help SAM-G better understand the target object. Can be empty.
- extra_masked_image_list: the list of auxiliary masked images of the original image. You can input a small part of the target object for each masked image. Can be empty.


## Evaluate Config file
The eval config file is located at `DHM/cfg/task`. You can change the config file for different setups.


## Training
```
cd src/
bash train.sh
```


## Evaluation
```
cd src/
model_dir=/path/to/model
bash adroit_eval.sh test_vrl3_color-easy  #{mode}_{agent_name}_{generalization type}
```
You should change the `model_dir` to your own path to load the trained model. The eval config file is located at `DHM/cfg/task`. You can change the config file for different setups. Simply modity `type` and `difficulty` in the `background`

## Checkpoints
We have provided some [checkpoints](https://drive.google.com/drive/folders/1a3d5d8n6cl0fr54rq8T31D_UL4_ynv6v?usp=drive_link) for the tasks together with their handmade masked images in `site_images/`. Use our scripts to evaluate the checkpoints.


## Imitation Learning
To run a imitation learning task, you should make sure that you already have a trained agent acting like an expert to generate demos. Then you use
```
cd src/imitation_learning
bash generation.sh
```
to generate demos under the original setting (no color change or background video). Afterwards, running 
```
bash train_demos.sh
```
to train the agent under the imitation learning setting. Test your trained agent by runing `eval_imitation.py`.


More infomation and details can be found at [mjrl](https://github.com/aravindr93/mjrl), [VRL3](https://github.com/microsoft/VRL3) and [ViGen-Adroit](https://github.com/gemcollector/RL-ViGen/tree/ViGen-adroit). 