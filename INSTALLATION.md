# Installation
The installation is mainly based on [RL-ViGen](https://github.com/gemcollector/RL-ViGen). You can get access to a lot of RL benchmark here while we only adopted SAM-G to DMControl and Adroit. If you want to use other benchmarks, you ran refer to RL-ViGen for more details.

Clone the SAM-G repo:
```
git clone https://github.com/wadiuvatzy/SAM-G
cd SAM-G/
```

Create a conda environment:
```
conda create -n sam-g python=3.8
```
Run the installation script:
```
bash setup/install_rlvigen.sh
```

 - DM-Control:  Our DM-Control also contains [mujoco_menagerie
](https://github.com/deepmind/mujoco_menagerie) as the basic component. We have incorporated all the relevant components associated with DM-Control in the first creating conda step.

 - Adroit: For adroit tasks, you should refer to the branch **SAM-G/Adroit**.


