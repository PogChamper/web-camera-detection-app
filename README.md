# web-camera-detection-app \
for Windows: \
0: change in config.yaml source field to "http://host.docker.internal:56000/mjpeg" \
1 https://github.com/gen2brain/cam2ip - download Windows version \
2 launch cam2ip.exe \
3 install VcXsrv https://sourceforge.net/projects/vcxsrv/ \
4 launch vcxsrv.exe \
5 docker build -t yolo-detection-app . \
6 docker run --rm -it -e DISPLAY=host.docker.internal:0.0 -v /tmp/.X11-unix:/tmp/.X11-unix -p 56000:56000 --network host yolo-detection-app \
\
\
for Linux: \
0 change in config.yaml source field to 0 \
1 docker build -t yolo-detection-app . \
2 xhost +local:docker \
3 docker run --device=/dev/video0:/dev/video0 -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix --network host yolo-detection-app \
