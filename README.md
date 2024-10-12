# web-camera-detection-app
### for Windows:
1. Change in ```config/config.yaml``` **source** field to ```"http://host.docker.internal:56000/mjpeg"``` \
2. Download **cam2ip** Windows version - ```https://github.com/gen2brain/cam2ip``` \
3. Launch ```cam2ip.exe``` \
4. Install **VcXsrv** ```https://sourceforge.net/projects/vcxsrv/``` \
5. Launch ```vcxsrv.exe``` \
6. ```docker build -t yolo-detection-app .``` \
7. ```docker run --rm -it -e DISPLAY=host.docker.internal:0.0 -v /tmp/.X11-unix:/tmp/.X11-unix -p 56000:56000 --network host yolo-detection-app``` \
\

### for Linux:
1. Change in ```config.yaml``` **source** field to ```0``` \
2. ```docker build -t yolo-detection-app .``` \
3. ```xhost +local:docker``` \
4. ```docker run --device=/dev/video0:/dev/video0 -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix --network host yolo-detection-app``` \
