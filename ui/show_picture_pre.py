from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
from pathlib import Path
import dlib
import numpy as np
from imdb_data.face_estimate.source_file.model_compil import get_model
# from imdb_data.face_estimate.source_file.model_compli_7 import get_model

class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数
        self.margin = 0.4
        self.detector = dlib.get_frontal_face_detector()
        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头
        self.model = get_model(model_name="ResNet50")
        self.model.load_weights("G:/Face age estimation/imdb_data/face_estimate/source_file/Finetune_with_appa/20_model_sum.hdf5")
        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数
        self.director = ""


    '''程序界面布局'''

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 总布局
        self.__layout_pic = QtWidgets.QHBoxLayout()   # 图片信息布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()  # 按键布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局
        self.button_open_camera = QtWidgets.QPushButton('打开相机')  # 建立用于打开摄像头的按键
        self.button_close = QtWidgets.QPushButton('退出')  # 建立用于退出程序的按键
        self.button_picture = QtWidgets.QPushButton("预测") #建立使用图片进行预测显示
        self.button_open_camera.setMinimumHeight(50)  # 设置按键大小
        self.button_close.setMinimumHeight(50)
        self.button_picture.setMinimumHeight(50)
        self.picture_dir = QtWidgets.QLineEdit("")


        self.button_close.move(10, 100)  # 移动按键
        '''信息显示'''
        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(641, 481)  # 给显示视频的Label设置大小为641x481
        '''图片信息布局'''
        self.__layout_pic.addWidget(self.picture_dir)
        self.__layout_pic.addWidget(self.button_picture)
        '''把按键加入到按键布局中'''
        self.__layout_fun_button.addWidget(self.button_open_camera)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_close)  # 把退出程序的按键放到按键布局中
        self.__layout_fun_button.addLayout(self.__layout_pic)
        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_fun_button)  # 把按键布局加入到总布局中
        self.__layout_main.addWidget(self.label_show_camera)  # 把用于显示视频的Label加入到总布局中
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    '''初始化所有槽函数'''

    def slot_init(self):
        self.button_open_camera.clicked.connect(
            self.button_open_camera_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序
        self.button_picture.clicked.connect(self.pre_picture)
        self.picture_dir.textChanged.connect(self.save_dir)

    '''槽函数之一'''

    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭相机')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.button_open_camera.setText('打开相机')


    def draw_label(self,image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def yield_images_from_dir(self,image_dir):
        image_dir = Path(image_dir)

        for image_path in image_dir.glob("*.*"):
            img = cv2.imread(str(image_path), 1)

            if img is not None:
                h, w, _ = img.shape
                r = 640 / max(w, h)
                yield cv2.resize(img, (int(w * r), int(h * r)))
    '''槽函数之二'''

    def pre_picture(self):
        image_generator = self.yield_images_from_dir(self.director)
        for img in image_generator:
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)
            detected = self.detector(input_img, 1)
            faces = np.empty((len(detected), 224, 224, 3))
            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - self.margin * w), 0)
                    yw1 = max(int(y1 - self.margin * h), 0)
                    xw2 = min(int(x2 + self.margin * w), img_w - 1)
                    yw2 = min(int(y2 + self.margin * h), img_h - 1)
                    cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (224, 224))

            results = self.model.predict(faces)
            pre_ages = []
            ages = np.arange(0, 101).reshape(101, 1)
            for i in range(len(results)):
                # pre_ages.append(results[i])
                age = 0.0
                for v in range(20):
                    predicted_ages = results[i][v * 101:(v + 1) * 101].dot(ages).flatten()
                    age += predicted_ages
                age = age / 20
                pre_ages.append(int(age))
            for i, d in enumerate(detected):
                label = str(int(pre_ages[i]))
                self.draw_label(input_img, (d.left(), d.top()), label)
            show = cv2.resize(input_img, (640, 480))  # 把读到的帧的大小重新设置为 640x480
            # show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                     QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage



    '''槽函数之三'''

    def save_dir(self,text):
        self.director = text






    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取
        img_h, img_w, _ = np.shape(self.image)
        detected = self.detector(self.image, 1)
        faces = np.empty((len(detected), 224, 224, 3))
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - self.margin * w), 0)
                yw1 = max(int(y1 - self.margin * h), 0)
                xw2 = min(int(x2 + self.margin * w), img_w - 1)
                yw2 = min(int(y2 + self.margin * h), img_h - 1)
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(self.image[yw1:yw2 + 1, xw1:xw2 + 1, :], (224, 224))

        results = self.model.predict(faces)
        pre_ages = []
        ages = np.arange(0, 101).reshape(101, 1)
        for i in range(len(results)):
            # pre_ages.append(results[i])
            age = 0.0
            for v in range(20):
                predicted_ages = results[i][v * 101:(v + 1) * 101].dot(ages).flatten()
                age += predicted_ages
            age = age / 20
            pre_ages.append(int(age))
        for i, d in enumerate(detected):
            label = str(int(pre_ages[i]))
            self.draw_label(self.image, (d.left(), d.top()), label)
        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = Ui_MainWindow()  # 实例化Ui_MainWindow
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过
