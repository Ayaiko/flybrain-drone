# Sourced by start_sim.sh / run.sh / run_forest.sh inside a flybrain-px4 container.
# GZ_PARTITION keeps the Gazebo topics of parallel worker containers apart (gz-transport discovers across the Docker
# bridge network otherwise). Workers are started with FLYBRAIN_WORKER=w1, w2, ...; the original container is "main".
export PATH=$HOME/.local/bin:$PATH PYTHONWARNINGS=ignore PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export GZ_PARTITION=fb_${FLYBRAIN_WORKER:-main}
export LP_NUM_THREADS=${LP_NUM_THREADS:-2}   # software renderer threads per sim; the eye images are tiny, more threads only contend
export GZ_IP=127.0.0.1   # keep gz-transport discovery on loopback: on the shared Docker bridge network, missed multicast heartbeats made
                          # Gazebo drop the eye subscriber mid-flight (camera frames stopped while physics kept running)
