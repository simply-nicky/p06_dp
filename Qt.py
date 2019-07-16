import pyqtgraph as pg, pyqtgraph.opengl as gl, numpy as np, sys
from PyQt5 import QtCore, QtGui, QtWidgets
from OpenGL.GL import glEnable, glBlendFunc, glBegin, glColor4f, glVertex3f, glEnd, glHint
from OpenGL.GL import GL_LINE_SMOOTH, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_LINE_SMOOTH_HINT, GL_NICEST, GL_LINES
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLViewWidget

def makeApp():
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    return app

class Viewer(QtGui.QMainWindow):
    def __init__(self, data, label, levels, parent=None, size=(640, 480)):
        super(Viewer, self).__init__(parent=parent, size=QtCore.QSize(size[0], size[1]))
        self.setWindowTitle('CBC Viewer')
        self.update_ui(data, label, levels)

        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.show()

    def update_ui(self, data, label, levels):
        self.layout = QtGui.QVBoxLayout()
        _label_widget = QtGui.QLabel(label)
        _label_widget.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(_label_widget)
        _image_view = pg.ImageView()
        _image_view.setPredefinedGradient('thermal')
        _image_view.setImage(img=data, levels=levels)
        self.layout.addWidget(_image_view)

def showData(self, data, label, levels=(0, 100)):
    app = makeApp()
    viewer = Viewer(data=data, label=label, levels=levels)
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        app.exec_()

class Grid(GLGraphicsItem):
    def __init__(self, size=None, color=None, antialias=True, glOptions='translucent'):
        GLGraphicsItem.__init__(self)
        self.setGLOptions(glOptions)
        self.antialias = antialias
        if color is None:
            color = (255, 255, 255, 80)
        self.setColor(color)
        if size is None:
            size = QtGui.QVector3D(1, 1, 0)
        self.setSize(size=size)
        self.setSpacing(0.05, 0.05)

    def setColor(self, *args):
        """
        Set the color of the grid. Arguments are the same as those accepted by
        :func:`glColor <pyqtgraph.glColor>`.
        """
        self.color = pg.glColor(*args)
        self.update()

    def setSize(self, x=None, y=None, size=None):
        if size is not None:
            x = size.x()
            y = size.y()
        self.__size = [x,y]
        self.update()
        
    def size(self):
        return self.__size[:]
        
    def setSpacing(self, x=None, y=None, spacing=None):
        """
        Set the spacing between grid lines.
        Arguments can be x,y,z or spacing=QVector3D().
        """
        if spacing is not None:
            x, y = spacing.x(), spacing.y()
        self.__spacing = [x,y]
        self.update()

    def spacing(self):
        return self.__spacing[:]
        
    def paint(self):
        self.setupGLState()

        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glBegin(GL_LINES)       

        x, y = self.size()
        xs, ys = self.spacing()
        xvals = np.arange(0, x + xs * 0.001, xs)
        yvals = np.arange(0, y + ys * 0.001, ys)
        glColor4f(*self.color)
        for x in xvals:
            glVertex3f(x, yvals[0], 0)
            glVertex3f(x, yvals[-1], 0)
        for y in yvals:
            glVertex3f(xvals[0], y, 0)
            glVertex3f(xvals[-1], y, 0)

        glEnd()

class ScatterViewer(GLViewWidget):
    def __init__(self, title='Scatter Plot', origin=(0.0, 0.0, 0.0), roi=(1.0, 1.0, 1.0), size=(800, 600), parent=None):
        GLViewWidget.__init__(self, parent)
        self.resize(size[0], size[1])
        self.setWindowTitle(title)
        self.origin, self.roi = origin, roi
        self.setCenter(pos=QtGui.QVector3D(origin[0] + roi[0] / 2, origin[1] + roi[1] / 2, origin[2] + roi[2] / 2))
        self.setDistance(max(self.roi) * 2)
        self.setGrid()
        self.sp = gl.GLScatterPlotItem()
        self.sp.setGLOptions('translucent')
        self.addItem(self.sp)
        
    def setGrid(self):
        self.gx = Grid(color=(255, 255, 255, 50))
        self.gx.setSize(self.roi[0], self.roi[2])
        self.gx.rotate(90, 1, 0, 0)
        self.gx.translate(*self.origin)
        self.addItem(self.gx)
        self.gy = Grid(color=(255, 255, 255, 50))
        self.gy.setSize(self.roi[2], self.roi[1])
        self.gy.rotate(90, 0, -1, 0)
        self.gy.translate(*self.origin)
        self.addItem(self.gy)
        self.gz = Grid(color=(255, 255, 255, 50))
        self.gz.setSize(self.roi[0], self.roi[1])
        self.gz.translate(*self.origin)
        self.addItem(self.gz)
        
    def setGridColor(self, color):
        self.gx.setColor(color)
        self.gy.setColor(color)
        self.gz.setColor(color)
        self.update()
        
    def setDistance(self, distance):
        self.opts['distance'] = distance
        self.update()
        
    def setCenter(self, x=None, y=None, z=None, pos=None):
        if pos is not None:
            x, y, z = pos.x(), pos.y(), pos.z()
        self.opts['center'] = QtGui.QVector3D(x, y, z)
        self.update()
        
    def setData(self, **kwds):
        self.sp.setData(**kwds)