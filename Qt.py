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

class Viewer2D(QtGui.QMainWindow):
    def __init__(self, data, label, levels, parent=None, size=(640, 480)):
        QtGui.QMainWindow.__init__(parent=parent, size=QtCore.QSize(size[0], size[1]))
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
    viewer = Viewer2D(data=data, label=label, levels=levels)
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

    def setColor(self, color):
        """
        Set the color of the grid. Arguments are the same as those accepted by
        :func:`glColor <pyqtgraph.glColor>`.
        """
        self.color = pg.glColor(color)
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

    def setGrid(self, x=None, y=None, size=None, ratio=20):
        if size is not None:
            x = size.x()
            y = size.y()
        self.setSize(x, y)
        self.setSpacing(x / ratio, x / ratio)

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

class Viewer3D(GLViewWidget):
    def __init__(self, title='Plot3D', origin=(0.0, 0.0, 0.0), roi=(1.0, 1.0, 1.0), size=(800, 600), parent=None):
        GLViewWidget.__init__(self, parent)
        self.resize(size[0], size[1])
        self.setWindowTitle(title)
        self.origin, self.roi = origin, roi
        self.setCamera()
        self.makeAxisGrid()
        
    def makeAxisGrid(self):
        self.gx = Grid(color=(255, 255, 255, 50))
        self.gx.setGrid(self.roi[0], self.roi[2])
        self.gx.rotate(90, 1, 0, 0)
        self.gx.translate(*self.origin)
        self.addItem(self.gx)
        self.gy = Grid(color=(255, 255, 255, 50))
        self.gy.setGrid(self.roi[2], self.roi[1])
        self.gy.rotate(90, 0, -1, 0)
        self.gy.translate(*self.origin)
        self.addItem(self.gy)
        self.gz = Grid(color=(255, 255, 255, 50))
        self.gz.setGrid(self.roi[0], self.roi[1])
        self.gz.translate(*self.origin)
        self.addItem(self.gz)

    def setAxisGrid(self, origin, roi):
        self.gx.translate(origin[0] - self.origin[0], origin[1] - self.origin[1], origin[2] - self.origin[2])
        self.gx.setGrid(roi[0], roi[2])
        self.gy.translate(origin[0] - self.origin[0], origin[1] - self.origin[1], origin[2] - self.origin[2])
        self.gy.setGrid(roi[2], roi[1])
        self.gz.translate(origin[0] - self.origin[0], origin[1] - self.origin[1], origin[2] - self.origin[2])
        self.gz.setGrid(roi[0], roi[1])

    def setAxisGridColor(self, color):
        self.gx.setColor(color)
        self.gy.setColor(color)
        self.gz.setColor(color)
        self.update()

    def setCamera(self):
        self.opts['center'] = QtGui.QVector3D(self.origin[0] + self.roi[0] / 2, self.origin[1] + self.roi[1] / 2, self.origin[2] + self.roi[2] / 2)
        self.opts['distance'] = max(self.roi) * 2
        self.update()

class ScatterViewer(Viewer3D):
    def __init__(self, title='Scatter Plot', origin=(0.0, 0.0, 0.0), roi=(1.0, 1.0, 1.0), size=(800, 600), parent=None):
        Viewer3D.__init__(self, title, origin, roi, size, parent)
        self.sp = gl.GLScatterPlotItem()
        self.sp.setGLOptions('translucent')
        self.addItem(self.sp)

    def setData(self, pos, color=[1.0 ,1.0, 1.0, 0.5], size=10):
        """
        Update the data displayed by this item. All arguments are optional; 
        for example it is allowed to update spot positions while leaving 
        colors unchanged, etc.
        
        ====================  ==================================================
        **Arguments:**
        pos                   (N,3) array of floats specifying point locations.
        color                 (N,4) array of floats (0.0-1.0) specifying
                              spot colors OR a tuple of floats specifying
                              a single color for all spots.
        size                  (N,) array of floats specifying spot sizes or 
                              a single value to apply to all spots.
        ====================  ==================================================
        """
        kwds = {'pos': pos, 'color': color, 'size': size}
        self.sp.setData(**kwds)
        origin = pos.min(axis=0)
        roi = pos.max(axis=0) - origin
        self.setAxisGrid(origin, roi)
        self.origin, self.roi = origin, roi
        self.setCamera()

class VolumeViewer(Viewer3D):
    def __init__(self, title='Volume Plot', origin=(0.0, 0.0, 0.0), roi=(1.0, 1.0, 1.0), size=(800, 600), parent=None):
        Viewer3D.__init__(self, title, origin, roi, size, parent)
        self.v = gl.GLVolumeItem(data=None)
        self.addItem(self.v)

    def setData(self, data, smooth=True, sliceDensity=1):
        """
        ==============  =======================================================================================
        **Arguments:**
        data            Volume data to be rendered. *Must* be 4D numpy array (x, y, z, RGBA) with dtype=ubyte.
        sliceDensity    Density of slices to render through the volume. A value of 1 means one slice per voxel.
        smooth          (bool) If True, the volume slices are rendered with linear interpolation 
        ==============  =======================================================================================
        """
        self.v.sliceDensity, self.v.smooth = sliceDensity, smooth
        self.v.setData(data)
        roi = data.shape[:-1]
        self.setAxisGrid(self.origin, roi)
        self.roi = roi
        self.setCamera()

def volumedata(data, col=[255, 255, 255]):
    voldata = np.empty(data.shape + (4,), dtype=np.ubyte)
    adata = np.log(data - data.min() + 1)
    voldata[..., 0:3] = col
    voldata[..., 3] = adata * (255 / adata.max())
    return voldata