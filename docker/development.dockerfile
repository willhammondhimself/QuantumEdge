# Development Dockerfile with hot-reloading and debugging tools
FROM python:3.10-slim-bullseye

# Install system dependencies including development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Node.js for Jupyter extensions
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies including dev tools
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-dev.txt

# Install Jupyter Lab extensions
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
    jupyter labextension install @jupyterlab/debugger

# Configure Jupyter
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:$PYTHONPATH \
    JUPYTER_ENABLE_LAB=yes

# Expose ports
EXPOSE 8000 8888 5678

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]