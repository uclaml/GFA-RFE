# Uncertainty-Aware Reward-Free Exploration with General Function Approximation

This is the codebase for our work **Uncertainty-Aware Reward-Free Exploration with General Function Approximation**.






## Requirements
This environment needs access to a GPU that can run CUDA 10.2 and CUDNN 8. To install all required dependencies, create an anaconda environment by running
```sh
conda env create -f conda_env.yml
```
Once the installation is complete, you can activate your environment using the following command:
```sh
conda activate urlb
```

## Agents
| Agent | Command |  Paper |
|---|---|---|
| dsquare | `agent=dsquare` | Our paper
| ICM | `agent=icm` |  [paper](https://arxiv.org/abs/1705.05363)|
| DIAYN | `agent=diayn` |  [paper](https://arxiv.org/abs/1802.06070)|
| APT(ICM) | `agent=icm_apt` |  [paper](https://arxiv.org/abs/2103.04551)|
| APS | `agent=aps` |  [paper](http://proceedings.mlr.press/v139/liu21b.html)|
| SMM | `agent=smm` |  [paper](https://arxiv.org/abs/1906.05274) |
| RND | `agent=rnd` |  [paper](https://arxiv.org/abs/1810.12894) |
| Disagreement | `agent=disagreement` |  [paper](https://arxiv.org/abs/1906.04161) |

## Available Domains
We support the following domains.
| Domain | Tasks |
|---|---|
| `walker` | `walker_stand`, `walker_walk`, `walker_run`, `walker_flip` |
| `quadruped` | `quadruped_walk`, `quadruped_run`, `quadruped_stand`, `quadruped_jump` |

## Instructions
### Running the program
First add execute permission to `run.sh` script:
```sh
chmod +x run.sh
```
Then the online pretraining and offline finetuning can be executed by using the `run.sh` script:

```sh
./run.sh --domain "walker" --task "walker_walk" --agent "dsquare" --num_pretrain_frames 1000010 --device 0 --seed 0 
```
The supported parameters are
| Parameter | Meaning | Values |
| --- | --- | --- |
| domain | The enviroment where the agent complete tasks | `walker`, `quadruped` |
| task | The tasks for the agent to complete | For `walker`: `walker_stand`, `walker_walk`, `walker_run`, `walker_flip`.For `quadruped`:`quadruped_walk`, `quadruped_run`, `quadruped_stand`, `quadruped_jump` |
| agent | The exploration algorithm for online pretrain | `dsquare`, `icm`, `diayn`, `icm_apt`, `aps`, `smm`, `rnd`, `disagreement` |
| num_pretrain_frames | The total number of frames in the pretraining phase | 100010, 500010, 1000010, or any other integers | 
| device | The cuda device for running the program | 0 to 7 |
| seed | The random seed for running the program | Any integers |


The collected trajectories will be stored under 
```
./online/<date>/<time>_<agent>_<task>
```

### Monitoring
Logs for online pretraining and offline finetuning are stored in the `online` and `offline` folders, respectively. To launch tensorboard run:
```sh
tensorboard --logdir online (or offline)
```
The console output is also available in the form of:
```
| train | F: 6000 | S: 3000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```
a training entry decodes as
```
F  : total number of environment frames
S  : total number of agent steps
E  : total number of episodes
R  : episode return
FPS: training throughput (frames per second)
T  : total training time
```

## Acknowledgements

The codebase was adapted from [URLB](https://github.com/rll-research/url_benchmark).