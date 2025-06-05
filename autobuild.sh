#!/usr/bin/env bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python-headless numpy

mkdir -p tools && cd tools
curl -L -o ffmpeg.tgz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg.tgz
mv ffmpeg-* ffmpeg
echo "export PATH=\"$(pwd)/ffmpeg:$PATH\"" >> ~/.bashrc
echo "🔧  Add   export PATH=\"$(pwd)/ffmpeg:\$PATH\"   to your shell rc if not done automatically."
