ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG NTERACT_VERSION=0.28.0

USER root

# Manually install jupyter-kernelspec binary (for ijavascript)
RUN bash -c "echo -e '#!/usr/bin/python3\n\
import re\n\
import sys\n\
from jupyter_client.kernelspecapp import KernelSpecApp\n\
if __name__ == \"__main__\":\n\
    sys.argv[0] = re.sub(r\"(-script\\.pyw?|\\.exe)?\$\", \"\", sys.argv[0])\n\
    sys.exit(KernelSpecApp.launch_instance())\n\
' > /usr/bin/jupyter-kernelspec" \
 && cat /usr/bin/jupyter-kernelspec \
 && chmod +x /usr/bin/jupyter-kernelspec \
 && mkdir -p \
    /home/node/.jupyter \
    /home/node/.local/share/jupyter/kernels/javascript \
 # Install ijavascript kernel
 && bash -c "echo -e '{\n\
  \"argv\": [\n\
    \"ijskernel\",\n\
    \"--hide-undefined\",\n\
    \"{connection_file}\",\n\
    \"--protocol=5.0\"\n\
  ],\n\
  \"display_name\": \"Javascript (Node.js)\",\n\
  \"language\": \"javascript\"\n\
}' > /home/node/.local/share/jupyter/kernels/javascript/kernel.json" \
 && cat /home/node/.local/share/jupyter/kernels/javascript/kernel.json \
 # Add nteract settings
 && bash -c "echo -e '{\n\
  \"theme\": \"dark\",\n\
  \"editorType\": \"monaco\",\n\
  \"defaultKernel\": \"javascript\"\n\
}' > /home/node/.jupyter/nteract.json" \
 && cat /home/node/.jupyter/nteract.json \
 && chown -R node:node /home/node \
 && curl \
       -L https://github.com/n-riesco/ijavascript/blob/8637a3e18b89270121f49733d03af0e3e6e0a17a/images/nodejs/js-green-32x32.png \
       -o /home/node/.local/share/jupyter/kernels/javascript/logo-32x32.png \
 && curl \
       -L https://github.com/n-riesco/ijavascript/blob/8637a3e18b89270121f49733d03af0e3e6e0a17a/images/nodejs/js-green-64x64.png \
       -o /home/node/.local/share/jupyter/kernels/javascript/logo-64x64.png \
 # Install nteract/desktop
 && curl \
       -L https://github.com/nteract/nteract/releases/download/v${NTERACT_VERSION}/nteract_${NTERACT_VERSION}_amd64.deb \
       -o /tmp/nteract_${NTERACT_VERSION}_amd64.deb \
 && apt update -y \
 && apt install -y --no-install-recommends \
    libasound2 jupyter-notebook \
    /tmp/nteract_${NTERACT_VERSION}_amd64.deb \
 && apt autoremove -y \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
 # Install ijavascript
 && npm install --global --unsafe-perm --no-audit --no-fund ijavascript \
 && ijsinstall --install=global --spec-path=full

USER node
