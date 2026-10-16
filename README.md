# Qiskit Pipeline — Quantum ML on Kubeflow

**A hybrid quantum–classical neural network (Qiskit + PyTorch) packaged as a Kubeflow Pipeline, and triggerable from a custom Node-RED node.**

![Qiskit](https://img.shields.io/badge/Qiskit-6929C4?logo=qiskit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Kubeflow](https://img.shields.io/badge/Kubeflow_Pipelines-1.8-4285F4?logo=kubeflow&logoColor=white)
![Node-RED](https://img.shields.io/badge/Node--RED-8F0000?logo=nodered&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)

## Overview

This project explores running **quantum machine learning workloads as cloud-native ML pipelines**.
A hybrid QNN is trained to classify MNIST digits (0 vs 1). Each step is a containerized Kubeflow
Pipelines component, and the whole pipeline can be launched from a low-code Node-RED flow
(built on top of [kube-nodered](https://github.com/footprintai/kube-nodered)).

```
Node-RED [qnn] node ──KFP REST API──▶ Kubeflow Pipelines
                                        │
                     ┌──────────────────┼───────────────────┐
                     ▼                  ▼                   ▼
              download_data   ──▶      qnn         ──▶  show_results
         MNIST 0/1 subset          hybrid QNN training    print train/test
         (200 train / 100 test)    & evaluation           accuracy
```

## Model

The QNN layer is a 2-qubit circuit wrapped as a PyTorch module with `TorchConnector`, so it trains
end-to-end with classical layers:

| Stage | Details |
|---|---|
| Classical feature extractor | Conv2d(1→2) → Conv2d(2→16) → Dropout → FC(256→64) → FC(64→2) |
| Quantum layer | `ZZFeatureMap(2)` + `RealAmplitudes(2, reps=1)`, `EstimatorQNN` |
| Output | FC(1→1) → 2-class probabilities, `NLLLoss`, Adam (lr = 1e-3), 10 epochs |

## Project Structure

```
├── pipeline.py              # KFP pipeline definition → compiles to qnnPipeline.yaml
├── download_data/           # component 1: prepare MNIST 0/1 subset
│   ├── download_data.py
│   ├── download_data.yaml   # KFP component spec
│   └── Dockerfile, requirements.txt
├── qnn/                     # component 2: train & evaluate the hybrid QNN
│   ├── qnn.py
│   ├── qnn.yaml             # KFP component spec
│   └── Dockerfile, requirements.txt
└── Qnn_nodered/             # Node-RED integration
    ├── qnn/                 # custom Node-RED node (qnn.html / qnn.js)
    ├── qnn.py               # uploads & runs the pipeline via the KFP REST API, polls the result
    └── qnnPipeline.yaml     # compiled pipeline
```

## Getting Started

### Prerequisites

- A Kubeflow 1.x instance (Dex auth)
- Python 3.8+, `kfp==1.7`
- Docker (only if you rebuild the component images)

### 1. Build the component images (optional)

The component specs reference pre-built images on Docker Hub. To use your own, build and push them,
then update `image:` in `download_data/download_data.yaml` and `qnn/qnn.yaml`.

```bash
docker build -t <your-registry>/qnn-download-data ./download_data
docker build -t <your-registry>/qnn-train ./qnn
docker push <your-registry>/qnn-download-data
docker push <your-registry>/qnn-train
```

### 2. Compile the pipeline

```bash
pip install kfp==1.7
python pipeline.py          # produces qnnPipeline.yaml
```

You can upload `qnnPipeline.yaml` directly in the Kubeflow Pipelines UI, or:

### 3. Run it from Node-RED

1. Set up [kube-nodered](https://github.com/footprintai/kube-nodered) (example `1.connect-kubeflow`).
2. Copy `Qnn_nodered/qnn/` into the Node-RED user directory's `node_modules/`, and
   `Qnn_nodered/qnn.py` + `qnnPipeline.yaml` (under `pipelines/`) into its `py/` folder.
3. Provide Kubeflow credentials through environment variables:

   ```bash
   export KUBEFLOW_HOST=<your-kubeflow-endpoint>
   export KUBEFLOW_USERNAME=<your-username>
   export KUBEFLOW_PASSWORD=<your-password>
   ```

4. Drag the **qnn** node into a flow and trigger it. The train / test accuracy is returned to the debug panel.

## Related

- [Node-RED Kubeflow Pipeline Extension](https://github.com/justin0322/Node-RED-Kubeflow-Pipeline-Extension) — low-code AutoML nodes for Kubeflow
- [Qiskit Machine Learning – Torch Connector tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/05_torch_connector.html)
- [Kubeflow Pipelines Python HTTP client](https://github.com/kubeflow/pipelines/tree/1.8.21/backend/api/python_http_client)
- [node-red-contrib-pythonshell](https://github.com/namgk/node-red-contrib-pythonshell)
