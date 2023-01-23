import win32con
import win32gui
import win32process
import mss
import mss.tools
import numpy as np


def isRealWindow(hWnd):
    '''Return True iff given window is a real Windows application window.'''
    if not win32gui.IsWindowVisible(hWnd):
        return False
    if win32gui.GetParent(hWnd) != 0:
        return False
    hasNoOwner = win32gui.GetWindow(hWnd, win32con.GW_OWNER) == 0
    lExStyle = win32gui.GetWindowLong(hWnd, win32con.GWL_EXSTYLE)
    if (((lExStyle & win32con.WS_EX_TOOLWINDOW) == 0 and hasNoOwner)
      or ((lExStyle & win32con.WS_EX_APPWINDOW != 0) and not hasNoOwner)):
        if win32gui.GetWindowText(hWnd):
            return True
    return False


def getWindowSizes():
    '''
    Return a list of dict for each real window within the screen boundaries.
    '''
    def callback(hWnd, windows):
        if not isRealWindow(hWnd):
            return
        rect = list(win32gui.GetWindowRect(hWnd))
        name = win32gui.GetWindowText(hWnd)
        ctid, cpid = win32process.GetWindowThreadProcessId(hWnd)
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        if all([r >= 0 for r in rect]):
            windows.append({"name": name, "pid": cpid, "rect": rect, "width": w, "height": h})
    windows = []
    win32gui.EnumWindows(callback, windows)
    return windows