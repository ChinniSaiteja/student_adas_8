# =============================================================================
# HSHL ADAS Student Lab — Container
# Base: ROS 2 Humble (no CARLA / display required)
# =============================================================================
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# System dependencies
# (Do NOT modify this block unless you know what you are doing)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ros-humble-rmw-zenoh-cpp \
    ros-humble-sensor-msgs \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-default-plugins \
    python3-pip \
    libjpeg-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Core Python libraries — already available in your code
#
#   numpy          – array maths, matrix operations
#   opencv         – image processing (cv2)
#   scikit-learn   – classical machine learning (SVM, clustering, …)
#   scipy          – signal processing, linear algebra, statistics
#
# These are installed for you. You can import them directly without adding
# anything to requirements.txt.
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    opencv-python-headless \
    scikit-learn \
    scipy

# -----------------------------------------------------------------------------
# Student Python libraries
#
# Add any extra pip packages you need to requirements.txt.
# They are installed here automatically when you rebuild the image.
#
# Example — to add PyTorch:
#   1. Add  torch torchvision  to requirements.txt
#   2. Run  docker compose up --build
#
# If a package needs a system apt package first, add it to the apt-get block
# above and rebuild.
# -----------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
WORKDIR /app
COPY . /app/

# Strip Windows CR (\r) from all shell scripts so they run correctly inside
# this Linux container even if the host checked them out with CRLF endings.
RUN find /app -name "*.sh" -exec sed -i 's/\r$//' {} +

ENV RMW_IMPLEMENTATION=rmw_zenoh_cpp
ENV PYTHONPATH=/app

CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && python3 solution/my_adas.py"]
