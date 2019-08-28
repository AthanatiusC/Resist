import cv2
import queue
import threading
import system_logging as log
import PySimpleGUI as sg

##GUI INIT
sg.ChangeLookAndFeel('Dark')
layout =[
            [sg.Text('Register Dataset', size=(18,1), font=('Any',18),text_color='#1c86ee' ,justification='left')],
            [sg.Text('NIS/NIK'), sg.In('',size=(40,1), key='NISNIK')],
            [sg.Text('Dataset Resolution'), sg.In('1000',size=(40,1), key='datasetres'),sg.Text('px')],
            [sg.Text('Resolution'), sg.In('1000',size=(40,1), key='Resolution'),sg.Text('px')],
            [sg.Text('Camera ID'), sg.In('0',size=(40,1), key='CAMID')],
            # [sg.Text('Sampling'), sg.Slider(range=(0,5),orientation='h', resolution=.1, default_value=3, size=(15,15), key='Sampling')],
            [sg.Text('Gamma'), sg.Slider(range=(1,5),orientation='h', resolution=.1, default_value=1, size=(15,15), key='Gamma')],
            # [sg.Text('Resolution'), sg.Slider(range=(720,1920),orientation='h', resolution=.1, default_value=1000, size=(15,15), key='Resolution')],
            # [sg.Text('Confidence'), sg.Slider(range=(0,1),orientation='h', resolution=.1, default_value=0.5, size=(15,15), key='Confidence')],
            # [sg.Text('Output:')],
            # [sg.Output(size=(80, 10))],
            [sg.Button("Track"), sg.Cancel()]
        ]
win = sg.Window('Register Faces',default_element_size=(21,1),text_justification='right',auto_size_text=False).Layout(layout)

##INITIALIZE VARIABLE
vs = cv2.VideoCapture(0)
frames = queue.Queue(maxsize=10)
Closed = False

##PREPARING CAMERA
def init():
    cam_list = []
    for i in range(10):
        vs = cv2.VideoCapture(i)
        if vs.isOpened():
            cam_list.append(i)
    for cams in cam_list:
        log.sys("Cam {} online".format(cams))
    return cam_list
        
##CAPTURE CAMERA PUT INTO QUEUE
def store_frame(num):
    vs = cv2.VideoCapture(num)
    fps = vs.get(cv2.CAP_PROP_FPS)
    while True:
        if Closed:
            break
        _,frame = vs.read()
        if not _:
            break
        cv2.putText(frame,"FPS : {}".format(fps),(0,100),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
        frames.put(frame)

##CORE PROCESS
def main():
    while frames.not_empty:
        frame = frames.get()
        cv2.imshow("image",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            Closed = True
            log.warn("Master key pressed!")
            break


if __name__ == "__main__":
    while not Closed:
        event, values = win.Read()
        ##BUTTON PRESS
        if event is None or event =='Cancel':
            exit()
        elif event == 'Track':
            ##GUI VAR TO LOCAL DATA
            NISNIK = values['NISNIK']
            ##CALL INIT
            online_cams = init()
            ##FRAME QUEUE THREAD
            t1 = threading.Thread(target=store_frame,args=online_cams[:1])
            t1.daemon = True
            t1.start()
            ##CALL MAIN
            main()
            vs.release()
            cv2.destroyAllWindows()