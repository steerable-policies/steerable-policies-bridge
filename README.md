# Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control

[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](TODO)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)]([TODO](https://huggingface.co/Embodied-CoT/steerable-policy-openvla-7b-bridge))

<!-- [**Getting Started**](#getting-started) | [**Pretrained VLAs**](#pretrained-vlas) | [**Installation**](#installation) | [**Fine-Tuning OpenVLA via LoRA**](#fine-tuning-openvla-via-lora) | [**Fully Fine-Tuning OpenVLA**](#fully-fine-tuning-openvla) |
[**Training VLAs from Scratch**](#training-vlas-from-scratch) | [**Evaluating OpenVLA**](#evaluating-openvla) | [**MiniVLA Details**](#minivla) | [**Action Chunking**](#action-chunking) | [**Multi-Image**](#multi-image) | [**Project Website**](https://openvla.github.io/) -->



We present **Steerable Policies**: vision-language-action models (VLAs) that accept diverse instructions at many levels of abstraction.

This codebase is designed for training Steerable Policies on the Bridge dataset. It is built atop the [OpenVLA](https://openvla.github.io/). **We thus highly recommend reading through that codebase's README as well, as it provides much more extensive details on usage and troubleshooting.**

<!-- ## Getting Started

To get started with loading and running OpenVLA models for inference, we provide a lightweight interface that leverages
HuggingFace `transformers` AutoClasses, with minimal dependencies.

For example, to load `openvla-7b` for zero-shot instruction following in the
[BridgeData V2 environments](https://rail-berkeley.github.io/bridgedata/) with a WidowX robot:

```python
# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

# Grab image input & format prompt
image: Image.Image = get_from_camera(...)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# Execute...
robot.act(action, ...)
```

We also provide an [example script for fine-tuning OpenVLA models for new tasks and
embodiments](./vla-scripts/finetune.py); this script supports different fine-tuning modes -- including (quantized)
low-rank adaptation (LoRA) supported by [HuggingFace's PEFT library](https://huggingface.co/docs/peft/en/index).

For deployment, we provide a lightweight script for [serving OpenVLA models over a REST API](./vla-scripts/deploy.py),
providing an easy way to integrate OpenVLA models into existing robot control stacks,
removing any requirement for powerful on-device compute.

--- -->

## Installation

> **Note**: These installation instructions are shared between OpenVLA and this repo. Please see that codebase for [more extensive details](<IP of machine connected to WidowX>) on installation and versioning.

Use the setup commands below to get started:

```bash
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Clone and install this repo
git clone git@github.com:steerable-policies/steerable-policies-bridge.git
cd steerable-policies-bridge
pip install -e .

# Below is NOT NEEDED for inference!
# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```
Again, we highly recommend checking the OpenVLA installation instructions for more details, especially for troubleshooting.

## Downloads

See our HuggingFace for:
- Weights for the base [Steerable Policy](https://huggingface.co/Embodied-CoT/steerable-policy-openvla-7b-bridge)
<!-- - Weights for the embodied reasoning high-level VLM -->
- [Bridge annotations](https://huggingface.co/datasets/Embodied-CoT/steering_features_bridge), including language language rationales and steering commands.
  
The model weights are needed for [**Inference**](#evaluating-steerable-policies-on-bridge-widowx) while the annotations are needed for [**Training**](#training-steerable-policies-on-bridge).

## Evaluating Steerable Policies on Bridge WidowX

Running inference with Steerable Policies can be done with a server-client setup, meaning the VLA does not need to be on the same machine that is connected to the robot. It is specifically designed to interface with [Manipulator Gym](https://github.com/rail-berkeley/manipulator_gym) -- please check that codebase for more extensive instructions.

### Installation: Robot Machine

These instructions are loosely adapted from [Manipulator Gym](https://github.com/rail-berkeley/manipulator_gym).

On the machine connected to the WidowX, you will need to install:
- [Steerable Gym](https://github.com/steerable-policies/steerable_gym) (our adapted version of Manipulator Gym)
- [AgentLace](https://github.com/youliangtan/agentlace)
- [Trossen's Interbotix and ROS scripts](https://docs.trossenrobotics.com/interbotix_xsarms_docs/ros_interface/ros1/software_setup.html)

The former two packages are Python packages, and can thus be installed with `pip install -e .` in their respective repos. The Trossen scripts depend more heavily on the particulars of the machine connected to the WidowX, so please follow the instructions on that page.

### Installation: Policy Machine
On the machine that will host the policy, you will need a conda environment with both this repo, [Steerable Gym](https://github.com/steerable-policies/steerable_gym), and [AgentLace](https://github.com/youliangtan/agentlace) installed. Follow the instructions in [**Installation**](#installation) to install this repo, then run `pip install -e .` to install Steerable Gym and AgentLace, as was done on the robot machine. You do *not* need to install ROS on this machine.

Additionally, be sure to include a valid `.hf_token` file in this repo (see [OpenVLA's instructions here](https://github.com/openvla/openvla?tab=readme-ov-file#fully-fine-tuning-openvla) for details).

### Running Inference

To run inference, you will need to start the Manipulator Gym server on the robot machine, then host the VLA and run a Steerable Gym script on the policy machine.

**On the robot machine:** First, run the requisite Interbotix ROS processes:
```
source ~/interbotix_ws/devel/setup.bash # Or whichever folder contains the ROS workspace
roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s use_rviz:=false
```
Then, in the conda environment where Manipulator Gym and AgentLace are installed, run:
```
source ~/interbotix_ws/devel/setup.bash # Or whichever folder contains the ROS workspace
cd /PATH/TO/manipulator_gym
python3 manipulator_server.py --widowx --cam_ids 0
```

**On the policy machine:** First host the VLA server. 
After downloading the policy weights [here](https://huggingface.co/Embodied-CoT/steerable-policy-openvla-7b-bridge), activate the conda environment this repo was installed in and `cd` into `steerable_gym/policies`, then run:
```
# Hosts the steerable policy
# You can send it http requests containing the task language and images, to which it will reply with actions.
# By default, it uses 0.0.0.0:8000/act.
# Hosting it on a server is good because you won't have to reload the model every time the eval script crashes / is closed.
python policy_server.py --policy_path </path/to/downloaded/pt/file>
```
Then, in a separate terminal, run an inference script in `steerable_gym/policies`. The following runs the interactive inference script, allowing the human user to issue steering commands:
```
# Runs the inference loop: receives images from the WidowX machine, passes images + input language to the policy server,
# then runs whatever action it returns.
python steerable_eval.py  --ip <IP of machine connected to WidowX> --clip_actions --show_img --path_to_rollouts_dir </path/to/directory/to/save/rollouts>
```
We recommend checking that Python script for examples of how to run inference.

TODO: Add instructions for downloading the repo.


## Training Steerable Policies on Bridge
Steerable Policies are trained with *standard behavioral cloning*, as OpenVLA is. The primary change is with regards to which language labels the VLA is trained on: rather than task-level instructions provided by datasets like Bridge, we instead train on diverse *steering commands*.

### Instructions for Training Steerable Policies
The majority of the training changes in this codebase are thus in `prismatic/vla/datasets/datasets.py`. To run training, you will need to do the following:
- Download the Bridge annotations we extracted, provided at this [HuggingFace repo](https://huggingface.co/datasets/Embodied-CoT/steering_features_bridge).
- In `prismatic/vla/datasets/datasets.py`: change the line `PATH_TO_REASONING_DATA = "</path/to/steering_features_bridge>"` to point to the downloaded repo.
- Run training with the instructions provided by [OpenVLA](https://github.com/openvla/openvla?tab=readme-ov-file#fully-fine-tuning-openvla).

### Instructions for Training High-level Embodied Reasoners
We also provide code for fine-tuning a VLM into a high-level embodied reasoner (as describe in our paper). This can be enabled with the `train_reasoner` configuration option in `vla-scripts/train.py`. This yields a VLM that takes in robot observations and Bridge-like task instructions, then outputs a natural-language *rationale* followed by a steering command for solving the task. Said command can then be executed by the Steerable Policy.

### Further Training Details
Our code implements the following simple changes (see `RLDSBatchTransform` in `prismatic/vla/datasets/datasets.py`):
- For each Bridge datapoint in the batch, determine the unique ID (which specifies which episode the frame is from, as well as which timestep).
- This ID indexes into the dictionaries from the HuggingFace repo, yielding a list of all steering commands corresponding to that particular frame and episode.
- If that list exists, sample a random command from that list, and use that as the user instruction for BC training. Otherwise, use the default language.
  
This logic only affects `RLDSBatchTransform`, changing the input language for each frame from the defaults provided by Bridge. Thus, it makes no assumptions about the VLA architectural details; any options provided by OpenVLA should still be compatible.

For training the embodied reasoner, please see `ReasonerRLDSBatchTransform` instead. To summarize, that one also indexes into our annotation dictionaries, but instead of replacing the task-level language, it changes the next-token prediction text to be a chain-of-thought rationale followed by a steering command. The resulting fine-tuned VLM can thus autoregressively predict both these things when given a task and observation.





## Citation

If you find our code or models useful in your work, please cite [our paper](TODO):

```bibtex
@article{Chen26-steerable-policies,
    title={Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control},
    author={William Chen and Jagdeep Bhatia and Catherine Glossop and Nikhil Mathihalli and Ria Doshi and Andy Tang and Danny Driess and Karl Pertsch and Sergey Levine},
    year={2026}
}
```
