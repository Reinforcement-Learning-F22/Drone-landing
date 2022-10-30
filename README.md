# Drone-landing
Autonomous Drone Landing using Deep Reinforcement Learning Techniques 

## Installation

Install ffmpeg
```
sudo apt install ffmpeg
```
Than
```
conda create -n drones python=3.8
conda activate drones
pip3 install --upgrade pip
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/
pip3 install -e .
cd ../
git clone https://github.com/Reinforcement-Learning-F22/Drone-landing
cd Drone-landing/
pip3 install -e .
```

If there is some problem, you can also intall another version of PyTorch.
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
