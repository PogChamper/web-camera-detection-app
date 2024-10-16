# web-camera-detection-app
### Windows with Docker:
1. ```git clone https://github.com/PogChamper/web-camera-detection-app.git```
2. ```cd web-camera-detection-app```
3. Change in ```config/config.yaml``` **source** field to ```"http://host.docker.internal:56000/mjpeg"```
4. Download **cam2ip** Windows OpenCV version - ```https://github.com/gen2brain/cam2ip```
5. Launch ```cam2ip.exe```
6. Install **VcXsrv** ```https://sourceforge.net/projects/vcxsrv/```
7. Launch ```vcxsrv.exe```
8. ```docker build -t yolo-detection-app .```
9. ```docker run --rm -it -e DISPLAY=host.docker.internal:0.0 -v /tmp/.X11-unix:/tmp/.X11-unix -p 56000:56000 --network host yolo-detection-app```


### Linux with Docker:
1. ```git clone https://github.com/PogChamper/web-camera-detection-app.git```
2. ```cd web-camera-detection-app```
3. Change in ```config.yaml``` **source** field to ```0```
4. ```docker build -t yolo-detection-app .```
5. ```xhost +local:docker```
6. ```docker run --device=/dev/video0:/dev/video0 -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix --network host yolo-detection-app```

###  Windows locally run without Docker:
1. ```python -m venv yolo-detection-app-venv```
2. ```pip install --upgrade pip```
3. ```.\yolo-detection-app-venv\Scripts\activate.bat``` in **CMD** or  ```source ./yolo-detection-app-venv/Scripts/activate``` in **git bash for Windows**
4. ```git clone https://github.com/PogChamper/web-camera-detection-app.git```
5. ```cd web-camera-detection-app```
6. Change in ```config.yaml``` **source** field to ```0```
7. ```pip install -r requirements.txt```
8. ```python main.py```

###  Linux locally run without Docker:
1. ```python -m venv yolo-detection-app-venv```
2. ```pip install --upgrade pip```
3. ```source yolo-detection-app-venv/bin/activate```
4. ```git clone https://github.com/PogChamper/web-camera-detection-app.git```
5. ```cd web-camera-detection-app```
6. Change in ```config.yaml``` **source** field to ```0```
7. ```pip install -r requirements.txt```
8. ```python main.py```

###  MacOS (only locally run without Docker):
1. ```python -m venv yolo-detection-app-venv```
2. ```pip install --upgrade pip```
3. ```source yolo-detection-app-venv/bin/activate```
4. ```git clone https://github.com/PogChamper/web-camera-detection-app.git```
5. ```cd web-camera-detection-app```
6. Change in ```config.yaml``` **source** field to ```0```
7. ```pip install -r requirements.txt```
8. ```python main.py```