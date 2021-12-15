ARG FROM_IMAGE

FROM ${FROM_IMAGE}

ARG TARGETARCH

ADD --chown=rapids:rapids \
    https://raw.githubusercontent.com/n-riesco/ijavascript/8637a3e18b89270121f49733d03af0e3e6e0a17a/images/nodejs/js-green-32x32.png \
    /opt/rapids/.local/share/jupyter/kernels/javascript/logo-32x32.png

ADD --chown=rapids:rapids \
    https://raw.githubusercontent.com/n-riesco/ijavascript/8637a3e18b89270121f49733d03af0e3e6e0a17a/images/nodejs/js-green-64x64.png \
    /opt/rapids/.local/share/jupyter/kernels/javascript/logo-64x64.png

ARG NTERACT_VERSION=0.28.0

ADD --chown=root:root \
    https://github.com/nteract/nteract/releases/download/v${NTERACT_VERSION}/nteract_${NTERACT_VERSION}_${TARGETARCH}.deb \
    /tmp/nteract.deb

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
 && chmod +x /usr/bin/jupyter-kernelspec \
 # Install ijavascript kernel
 && bash -c "echo -e '{\n\
  \"argv\": [\n\
    \"ijskernel\",\n\
    \"--hide-undefined\",\n\
    \"{connection_file}\",\n\
    \"--protocol=5.0\"\n\
  ],\n\
  \"name\": \"javascript\",\n\
  \"language\": \"javascript\",\n\
  \"display_name\": \"Javascript (Node.js)\"\n\
}' > /opt/rapids/.local/share/jupyter/kernels/javascript/kernel.json" \
 && chmod 0644 /opt/rapids/.local/share/jupyter/kernels/javascript/logo-{32x32,64x64}.png \
 # Add nteract settings
 && mkdir -p /opt/rapids/.jupyter \
 && bash -c "echo -e '{\n\
  \"theme\": \"dark\",\n\
  \"editorType\": \"codemirror\",\n\
  \"defaultKernel\": \"javascript\",\n\
  \"codeMirror\": {\n\
    \"mode\": \"text/javascript\",\n\
    \"theme\": \"monokai\",\n\
    \"tabSize\": 2,\n\
    \"matchTags\": true,\n\
    \"undoDepth\": 999999,\n\
    \"inputStyle\": \"contenteditable\",\n\
    \"lineNumbers\": true,\n\
    \"matchBrackets\": true,\n\
    \"indentWithTabs\": false,\n\
    \"cursorBlinkRate\": 500,\n\
    \"lineWiseCopyCut\": false,\n\
    \"autoCloseBrackets\": 4,\n\
    \"selectionsMayTouch\": true,\n\
    \"showCursorWhenSelecting\": true\n\
  }\n\
}' > /opt/rapids/.jupyter/nteract.json" \
 && chown -R rapids:rapids /opt/rapids \
 # Install nteract/desktop
 && apt update \
 && DEBIAN_FRONTEND=noninteractive \
    apt install -y --no-install-recommends \
    python3-minimal libasound2 jupyter-notebook /tmp/nteract.deb \
 \
 # Clean up
 && apt autoremove -y && apt clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/lib/apt/lists/* \
    /var/cache/apt/archives/* \
 # Remove python3 kernelspec
 && jupyter kernelspec remove -f python3 \
 # Install ijavascript
 && npm install --global --unsafe-perm --no-audit --no-fund ijavascript \
 && ijsinstall --install=global --spec-path=full

ENV NTERACT_DESKTOP_DISABLE_AUTO_UPDATE=1

USER rapids

WORKDIR /opt/rapids/node

SHELL ["/bin/bash", "-l"]

CMD ["nteract"]
