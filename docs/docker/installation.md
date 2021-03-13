# Installing docker, docker-compose, and the nvidia-container-runtime

## Quick links

* [Installing docker](#installing-docker)
* [Installing the nvidia-container-toolkit](#installing-the-nvidia-container-toolkit)
* [Installing docker-compose](#installing-docker-compose)
* [Using the nvidia-container-runtime with docker-compose](#using-the-nvidia-container-runtime-with-docker-compose)

## Installing docker

Follow the [official docker installation instructions](https://docs.docker.com/get-docker/) to install docker for your OS.
<details>
<summary>Click here to see Ubuntu 16.04+ docker-ce installation commands:</summary>
<pre>
# Install docker-ce in one command. Adds your current user to the docker user group.<br/>
release=$(lsb_release -cs) \
 && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - \
 && sudo add-apt-repository -y "deb [arch=amd64] https://download.docker.com/linux/ubuntu $release stable" \
 && sudo apt install -y docker-ce \
 && sudo usermod -aG docker $USER
</pre>
</details>

## Installing the nvidia-container-toolkit

Follow the [official nvidia-container-toolkit installation instructions](https://github.com/NVIDIA/nvidia-docker#quickstart) to install the nvidia-container-toolkit for your OS.
<details>
<summary>Click here to see Ubuntu 16.04+ nvidia-container-toolkit installation commands:</summary>
<pre>
# Add nvidia-container-toolkit apt package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-docker.list<br/>
# Install the nvidia-container-toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit<br/>
# Restart the docker service to make the nvidia-container-toolkit available
sudo systemctl restart docker
</pre>
</details>

## Installing docker-compose

Follow the [official docker-compose installation instructions](https://docs.docker.com/compose/install/) to install docker-compose v1.28.5+ for your OS.
<details>
<summary>Click here to see Ubuntu 16.04+ docker-compose installation commands:</summary>
<pre>
# Install docker-compose v1.28.5, or select any newer release in https://github.com/docker/compose/releases
DOCKER_COMPOSE_VERSION=1.28.5<br/>
sudo curl \
    -L https://github.com/docker/compose/releases/download/$DOCKER_COMPOSE_VERSION/docker-compose-`uname -s`-`uname -m` \
    -o /usr/local/bin/docker-compose && sudo chmod +x /usr/local/bin/docker-compose
</pre>
</details>

## Using the nvidia-container-runtime with docker-compose before v1.28.5

Prior to docker-compose v1.28.5, using the nvidia-container-runtime with docker-compose [requires](https://github.com/docker/compose/issues/6691) `nvidia-container-runtime` is set as the default docker runtime. To do this, you will need to create or edit the `/etc/docker/daemon.json` file and update the "default-runtime" and "runtimes" settings.
<details>
<summary>Click here to see an example daemon.json:</summary>
<pre>
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
</pre>
</details>

If you created or edited the `/etc/docker/daemon.json` file, restart the docker service so the new settings are applied:

```bash
sudo systemctl restart docker
```

If you're unsure whether the changes you made were successful, you can run a quick test to verify `nvidia-container-runtime` is the default docker runtime.
<details>
<summary>Click here to see a successful test of whether NVIDIA devices are available in a docker container:</summary>
<pre>
docker run --rm -it nvidia/cuda nvidia-smi<br/>
> Fri Jul 31 20:39:59 2020
> +-----------------------------------------------------------------------------+
> | NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
> |-------------------------------+----------------------+----------------------+
> | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
> | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
> |                               |                      |               MIG M. |
> |===============================+======================+======================|
> |   0  Quadro RTX 8000     On   | 00000000:15:00.0  On |                  Off |
> | 33%   46C    P8    35W / 260W |   1453MiB / 48584MiB |      1%      Default |
> |                               |                      |                  N/A |
> +-------------------------------+----------------------+----------------------+
> |   1  Quadro RTX 8000     On   | 00000000:99:00.0 Off |                  Off |
> | 33%   34C    P8    14W / 260W |      6MiB / 48601MiB |      0%      Default |
> |                               |                      |                  N/A |
> +-------------------------------+----------------------+----------------------+
>
> +-----------------------------------------------------------------------------+
> | Processes:                                                                  |
> |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
> |        ID   ID                                                   Usage      |
> |=============================================================================|
> +-----------------------------------------------------------------------------+
</pre>
</details>
