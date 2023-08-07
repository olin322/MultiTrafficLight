### reinforcement learning
### development environment installation guide

##### install `stable baselines3` and `mujoco`

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple stable-baselines3[extra] gymnasium[mujoco] numpy overrides
```

##### be careful installing `tensorflow`, it seems unnecessary for reinforcement learning using `stable baselines3`
##### and should use `nvidia-cudnn-cu11==8.5.0.96` to be compatable with numpy
```
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

install `pytorch2.0.1`
most of the packages should be installed when installing `stable baselines3`
```
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```