# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import pickle
import sys
import time
import threading
import json

import wx
import wx.xrc

"""
ids
76,77 - left/right
78 - slider slice
79 -  num slice combo box
80 - erase
81 - remove last vertex
82 - contrast slider
83 - save all
84 - save current polygon/contour
85 - contour drawing style
86 - brush drawing style
87 - brush size
88 - view axis
89 - delete brush
90-xxx classes
201 - send to back
202 - undo last polygon
203 - select previous polygon
204 - select next polygon
205 - back to segmentation
206 - change class dropdown
207 - change class button
208 - Previous
209 - Next
210 - Hide segmentation
"""

# global vars
SEG_PATH_IN = ''
SEG_PATH_OUT = ''
FILE_PATH = []
PREV_FILE_PATH = []
IMAGE_WIDTH = 0
IMAGE_HEIGHT = 0
IMAGE_EXTENSION = ''
NUM_SLICES = 0
current_image = []
image_np_all_slices = []
image_np_all_slices_normed = []
image_np_all_slices_segmentation = []
image_np_all_slices_normed_plot = []
image_np_all_slices_segmentation_plot = []
polygons = []
polygons_to_be_deleted = []
current_polygon_to_be_deleted_idx = 0
num_polygons = 0
old_area_list = []
current_operation = []
operation_changed = True

START_BUTTON_ID_CLASSES = 90

FIRST_CLICK = False

HIDE_SEGMENTATION = False

image_previous_segmentation_overlaid = []

try:
    strings = json.load(open("languageConfig.yaml"))
except:
    pass

def _(s):
    try:
        return strings[s]
    except Exception as e:
        return s

try:
    classConfigLines = open("classConfig.yaml").read().splitlines()
except Exception as e:
    print('No/corruped classConfig.yaml file. Correct format: {className} {R} {G} {B} , where R,G,B range from 0..255')
    app = wx.App(None)
    style = wx.OK
    dialog = wx.MessageDialog(None, 'Missing/corrupted classConfig.yaml file. Correct format: {className} {R} {G} {B} , where R,G,B range from 0..255', 'Error', style)
    dialogResult = dialog.ShowModal()
    dialog.destroy()
    sys.exit(1)
    #dialog.SetYesNoCancelLabels(_("Segment image"), _("Edit segmented image"), _("Cancel"))


CLASS_NAMES = []
CLASS_COLORS_RGB = []

for line in classConfigLines:
    x=line.split(' ')
    CLASS_NAMES.append(x[0])
    CLASS_COLORS_RGB.append((int(x[1]), int(x[2]), int(x[3])))


operations_list = ['segment', 'edit']

operations_list_translated = [_('segment'), _('edit')]

BORDER_PADDING = 70

NUM_CLASSES = len(CLASS_NAMES)


# helpers / GUI

def selectOperation():
    # style = wx.YES_NO | wx.ICON_QUESTION | wx.CANCEL | wx.HELP
    style = wx.YES_NO | wx.ICON_QUESTION | wx.CANCEL
    dialog = wx.MessageDialog(None, _('Select operation'), _('Image segmentation'), style)

    # argh, wx 2.9+ :/
    try:
        dialog.SetYesNoCancelLabels(_("Segment image"), _("Edit segmented image"), _("Cancel"))
        dialog.SetHelpLabel(_("Quit"))
    except:
        dialog.setMessage(_("Cancel"))
    dialogResult = dialog.ShowModal()
    result = []
    if dialogResult == wx.ID_YES:
        result = 'segment'
    elif dialogResult == wx.ID_NO:
        result = 'edit'
    else:
        result = 'quit'
    dialog.Destroy()
    return result


def selectFolder():
    style = wx.DD_DIR_MUST_EXIST | wx.DD_CHANGE_DIR
    # dialog = wx.DirDialog(None, "Please choose image directory", style)
    dialog = wx.DirDialog(None, "Please choose image directory", style=style)

    if dialog.ShowModal() == wx.ID_OK:
        result = dialog.GetPath()
    else:
        result = ""
    dialog.Destroy()
    return result


def getSingleFilePath(wildcard):
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, _('Open image'), wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path


def bbox_np(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def drawSegmentationPolys(fromArea=0, current_seg=0, current_class=0, send_to_back=False, limits=(0, 0, 0, 0)):
    polygons_area = [item[0] for item in polygons]
    min_area = min(polygons_area)

    if min_area >= fromArea:
        # print('here!', min_area, fromArea)
        # print(polygons)
        # burn in only last one
        for idx_class in range(NUM_CLASSES):
            image_np_all_slices_segmentation[limits[0]:limits[1], limits[2]:limits[3], idx_class][
                np.where(current_seg[limits[0]:limits[1], limits[2]:limits[3]] > 0)] = 0
        image_np_all_slices_segmentation[limits[0]:limits[1], limits[2]:limits[3], current_class] = np.logical_or(
            image_np_all_slices_segmentation[limits[0]:limits[1], limits[2]:limits[3], current_class],
            current_seg[limits[0]:limits[1], limits[2]:limits[3]])
    else:  # redraw from fromArea
        for poly in polygons:
            if poly[0] > fromArea:
                continue
            bw_poly = cv2.fillPoly(np.zeros((poly[3][1] - poly[3][0], poly[3][3] - poly[3][2])), [poly[2]], 255, 8, 0,
                                   (-poly[3][2], -poly[3][0]))
            for idx_class in range(NUM_CLASSES):
                image_np_all_slices_segmentation[poly[3][0]:poly[3][1], poly[3][2]:poly[3][3], idx_class][
                    np.where(bw_poly > 0)] = 0
            image_np_all_slices_segmentation[:, :, poly[1]][poly[3][0]:poly[3][1],
            poly[3][2]:poly[3][3]] = np.logical_or(
                image_np_all_slices_segmentation[poly[3][0]:poly[3][1], poly[3][2]:poly[3][3], poly[1]], bw_poly)


def redrawSegmentationPolys(polygonss, highlighted=None):
    global image_np_all_slices_segmentation
    highlighted_poly = []
    if highlighted != None:
        highlighted_poly = polygonss[highlighted]
    polygonss = sorted(polygonss, reverse=True, key=lambda x: x[0])
    if highlighted != None:
        highlighted = polygonss.index(highlighted_poly)
    image_np_all_slices_segmentation = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES), dtype=np.bool_)
    for idx_current_poly, poly in enumerate(polygonss):
        if highlighted != None:
            if idx_current_poly == highlighted:
                bw_poly = cv2.fillPoly(np.zeros((poly[3][1] - poly[3][0], poly[3][3] - poly[3][2])), [poly[2]], 255, 8,
                                       0,
                                       (-poly[3][2], -poly[3][0]))
            else:
                bw_poly = cv2.drawContours(np.zeros((poly[3][1] - poly[3][0], poly[3][3] - poly[3][2])), [poly[2]], 0,
                                           255, 5, 8, 0, 0, (-poly[3][2], -poly[3][0]))
        else:
            bw_poly = cv2.fillPoly(np.zeros((poly[3][1] - poly[3][0], poly[3][3] - poly[3][2])), [poly[2]], 255, 8, 0,
                                   (-poly[3][2], -poly[3][0]))
        #for idx_class in range(NUM_CLASSES):
        image_np_all_slices_segmentation[poly[3][0]:poly[3][1], poly[3][2]:poly[3][3], :][np.where(bw_poly > 0)] = 0

        image_np_all_slices_segmentation[:, :, poly[1]][poly[3][0]:poly[3][1], poly[3][2]:poly[3][3]] = np.logical_or(
            image_np_all_slices_segmentation[poly[3][0]:poly[3][1], poly[3][2]:poly[3][3], poly[1]], bw_poly)


def plotAnnotationsOverlay():
    rgb_overlay = plotAnnotations()
    img_out = np.zeros_like(rgb_overlay)
    if not HIDE_SEGMENTATION:
        cv2.addWeighted(rgb_overlay, 0.5, image_np_all_slices_normed, 1, 0, img_out)
    else:
        img_out = image_np_all_slices_normed
    return img_out


def plotAnnotations():
    rgb_overlay = np.zeros_like(image_np_all_slices_normed)
    for idx_class in range(NUM_CLASSES):
        rgb_overlay[image_np_all_slices_segmentation[:, :, idx_class] > 0] = CLASS_COLORS_RGB[idx_class]
    # print(np.count_nonzero(rgb_overlay))
    return rgb_overlay


def plotAnnotationsOverlay2(image_np_all_slices_normed_inside, image_np_all_slices_segmentation_inside):
    rgb_overlay = plotAnnotations2(image_np_all_slices_normed_inside, image_np_all_slices_segmentation_inside)
    img_out = np.zeros_like(rgb_overlay)
    cv2.addWeighted(rgb_overlay, 0.5, image_np_all_slices_normed_inside, 1, 0, img_out)
    return img_out


def plotAnnotations2(image_np_all_slices_normed_inside, image_np_all_slices_segmentation_inside):
    rgb_overlay = np.zeros_like(image_np_all_slices_normed_inside)
    labels_for_drawing = []
    for idx_class in range(NUM_CLASSES):
        rgb_overlay[image_np_all_slices_segmentation_inside[:, :, idx_class] > 0] = CLASS_COLORS_RGB[idx_class][::-1]
    return rgb_overlay

# other helpers

def make_unique(original_list):
    unique_list = []
    map(lambda x: unique_list.append(x) if (x not in unique_list) else False, original_list)
    return unique_list

def removePolygonDuplicates(old_polygons):
    # add : polygons.append([current_area, self.selected_class, approxContourPoints, (rmin, rmax, cmin, cmax)])
    new_polygons = []
    for current_old_polygon in old_polygons:
        is_duplicate = False
        for current_new_polygon in new_polygons:
            if current_new_polygon[0] == current_old_polygon[0] and current_new_polygon[3][0] == current_old_polygon[3][0] and current_new_polygon[3][1] == current_old_polygon[3][1] and current_new_polygon[3][2] == current_old_polygon[3][2] and current_new_polygon[3][3] == current_old_polygon[3][3]:
                is_duplicate = True
                break
        if not is_duplicate:
            new_polygons.append(current_old_polygon)
    new_polygons = sorted(new_polygons, key=lambda x: x[0])
    return new_polygons

###########################################################################
## Python code generated with wxFormBuilder (version Jan 23 2018)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################


class PleaseWaitFrame(wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=wx.EmptyString, pos=wx.DefaultPosition,
                          size=wx.Size(500, 300), style=0 | wx.TAB_TRAVERSAL)

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNHIGHLIGHT))

        bSizer2 = wx.BoxSizer(wx.VERTICAL)

        bSizer2.Add((0, 0), 1, wx.EXPAND, 5)

        self.m_staticText2 = wx.StaticText(self, wx.ID_ANY, u"Așteptați...", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText2.Wrap(-1)

        bSizer2.Add(self.m_staticText2, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        bSizer2.Add((0, 0), 1, wx.EXPAND, 5)

        self.SetSizer(bSizer2)
        self.Layout()

        self.Centre(wx.BOTH)

    def __del__(self):
        pass


###########################################################################
## Class drawing_panel
###########################################################################

class drawing_panel(wx.Panel):
    def __init__(self, parent, original_image, drawing_style, brush_size):
        global operation_changed
        operation_changed = False
        self.parent = parent
        self.brushSize = brush_size
        # self.original_image = original_image.Rotate90(False)
        self.original_image = original_image
        self.resized_bitmap = original_image.ConvertToBitmap()
        self.contourPoints = []
        self.old_area = 0
        wx.Panel.__init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(800, 800),
                          style=wx.TAB_TRAVERSAL)
        self.isDrawing = False
        self.drawingStyle = drawing_style
        self.dc = wx.ClientDC(self)
        self.gc = wx.GraphicsContext.Create(self.dc)
        self.selected_class = 0

        self.Bind(wx.EVT_MOVE, self.OnMove)
        self.Bind(wx.EVT_MOTION, self.OnMouseMove)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnResize)
        self.Bind(wx.EVT_LEFT_DOWN, self.startDrawing)
        self.Bind(wx.EVT_LEFT_UP, self.endDrawing)
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)

    def paint(self, event=None):
        # print("paint")
        dc = wx.PaintDC(self)
        dc.DrawBitmap(self.resized_bitmap, 0, 0, False);

    def OnMove(self, e):
        x, y = e.GetPosition()
        # print("current window position x = ", x, " y= ", y)

    def OnMouseMove(self, e):
        if current_operation != 'edit':
            x, y = e.GetPosition()
            if self.isDrawing and self.drawingStyle == 'contour':
                self.contourPoints.append([x, y])
                if len(self.contourPoints) > 1:
                    for idx, point in enumerate(self.contourPoints):
                        if idx == len(self.contourPoints) - 1:  # last point
                            continue
                        # self.gc.DrawLines(self.contourPoints[-2:-1])
                        self.dc.DrawLine(self.contourPoints[-1][0], self.contourPoints[-1][1],
                                         self.contourPoints[-2][0], self.contourPoints[-2][1])
            elif self.isDrawing and self.drawingStyle == 'brush':
                self.contourPoints.append([x, y])

                # dc.SetBrush(wx.Brush("red", wx.SOLID))
                # dc.SetPen(wx.Pen("red", self.brushSize, style=wx.SOLID))

                if len(self.contourPoints) > 1:
                    self.dc.DrawLine(self.contourPoints[-1][0], self.contourPoints[-1][1],
                                     self.contourPoints[-2][0], self.contourPoints[-2][1])
                    # write seg @ mouse up

            elif self.isDrawing and self.drawingStyle == 'brushDelete':
                self.contourPoints.append([x, y])

                if len(self.contourPoints) > 1:
                    # self.dc.DrawLine(self.contourPoints[-1][0], self.contourPoints[-1][1],
                    #            self.contourPoints[-2][0], self.contourPoints[-2][1])
                    # self.dc = wx.ClientDC(self) #batman
                    self.gc.DrawLines(self.contourPoints)
                    # write seg @ mouse up
                    # current_segmentation = self.getCurrentSegmentation()
                    # print(np.count_nonzero(current_segmentation == [255,255,255]))

                # dc.DrawCircle(x, y, 1)
            # print("current panel mouse position x = ", x, " y= ", y)

    def changeImage(self, new_image):
        self.dc = wx.ClientDC(self)  # batman
        self.gc = wx.GraphicsContext.Create(self.dc)  # batman
        self.original_image = new_image
        # self.original_image = new_image
        self.redrawImage()

    def OnPaint(self, event):
        self.redrawImage()

    def OnResize(self, event):
        self.redrawImage()
        self.dc.SetPen(wx.Pen(wx.Colour(CLASS_COLORS_RGB[self.selected_class]), style=wx.SOLID))
        self.gc.SetBrush(wx.Brush(
            wx.Colour(CLASS_COLORS_RGB[self.selected_class][0], CLASS_COLORS_RGB[self.selected_class][1],
                      CLASS_COLORS_RGB[self.selected_class][2], 125), wx.SOLID))
        new_w, new_h = self.dc.GetSize()
        rescaled_contour_points = []
        for point in self.contourPoints:
            rescaled_contour_points.append([int(point[0] * IMAGE_WIDTH / new_w), int(point[1] * IMAGE_HEIGHT / new_h)])
            self.contourPoints = rescaled_contour_points

    def OnCloseWindow(self, e):
        global operation_changed
        operation_changed = False

    def redrawImage(self):
        self.dc = wx.ClientDC(self)  # batman
        new_w, new_h = self.dc.GetSize()
        if new_w != 0 and new_h != 0:
            self.resized_bitmap = self.original_image.Scale(new_w, new_h).ConvertToBitmap()
            self.dc.DrawBitmap(self.resized_bitmap, 0, 0);

    def startDrawing(self, e):
        if current_operation != 'edit':
            x, y = e.GetPosition()
            self.isDrawing = True
            self.contourPoints.append([x, y])
            if self.isDrawing and self.drawingStyle == 'contour':
                self.dc.SetPen(wx.Pen(wx.Colour(CLASS_COLORS_RGB[self.selected_class]), style=wx.SOLID))
                self.gc.SetBrush(wx.Brush(
                    wx.Colour(CLASS_COLORS_RGB[self.selected_class][0], CLASS_COLORS_RGB[self.selected_class][1],
                              CLASS_COLORS_RGB[self.selected_class][2], 125), wx.SOLID))
                # self.gc.BeginLayer(125)
            elif self.isDrawing and self.drawingStyle == 'brushDelete':
                self.dc.SetBrush(wx.Brush("white", wx.SOLID))
                self.dc.SetPen(wx.Pen("white", self.brushSize, style=wx.SOLID))
            elif self.isDrawing and self.drawingStyle == 'brush':
                self.dc.SetBrush(wx.Brush("red", wx.SOLID))
                self.dc.SetPen(wx.Pen("red", self.brushSize, style=wx.SOLID))

    def endDrawing(self, e=None):
        global FIRST_CLICK
        global polygons_to_be_deleted
        if FIRST_CLICK:
            FIRST_CLICK = False
            return
        if current_operation != 'edit':
            # x, y = e.GetPosition()
            self.isDrawing = False
            # this should change depending on the drawing type but, hell, pre-alpha
            if self.drawingStyle == 'points':
                # draw points
                self.redrawImage()
                dc = wx.ClientDC(self)
                # dc.BeginDrawing()
                dc.SetPen(wx.Pen("green", style=wx.SOLID))
                dc.SetBrush(wx.Brush("green", wx.SOLID))
                for point in self.contourPoints:
                    dc.DrawCircle(point[0], point[1], 3)
                if len(self.contourPoints) > 2:
                    # ch = ConvexHull(np.array(self.contourPoints))
                    # Get the indices of the hull points.
                    # hull_indices = ch.vertices

                    # These are the actual points.
                    # hull_pts = self.contourPoints[hull_indices, :]
                    # draw lines
                    dc.SetPen(wx.Pen(CLASS_COLORS_RGB[self.selected_class], style=wx.SOLID))

                    for idx, current_hull_index in enumerate(hull_indices):
                        if idx == len(hull_indices) - 1:
                            dc.DrawLine(self.contourPoints[hull_indices[idx]][0],
                                        self.contourPoints[hull_indices[idx]][1],
                                        self.contourPoints[hull_indices[0]][0], self.contourPoints[hull_indices[0]][1])
                        else:
                            dc.DrawLine(self.contourPoints[hull_indices[idx]][0],
                                        self.contourPoints[hull_indices[idx]][1],
                                        self.contourPoints[hull_indices[idx + 1]][0],
                                        self.contourPoints[hull_indices[idx + 1]][1])
                    # plt.plot(pts[:, 0], pts[:, 1], 'ko', markersize=10)
                    # plt.fill(hull_pts[:, 0], hull_pts[:, 1], fill=False, edgecolor='b')
            elif self.drawingStyle == 'contour':
                # draw last point
                self.redrawImage()

                # dc = wx.ClientDC(self)
                # dc.SetPen(wx.Pen("red", style=wx.SOLID))
                # dc.DrawLine(self.contourPoints[-1][0], self.contourPoints[-1][1],
                #    self.contourPoints[0][0], self.contourPoints[0][1])
                # fill contour
                # self.dc = wx.ClientDC(self)
                # self.gc = wx.GraphicsContext.Create(self.dc)
                self.dc.SetPen(wx.Pen(CLASS_COLORS_RGB[self.selected_class], style=wx.SOLID))

                self.gc.SetBrush(wx.Brush(
                    wx.Colour(CLASS_COLORS_RGB[self.selected_class][0], CLASS_COLORS_RGB[self.selected_class][1],
                              CLASS_COLORS_RGB[self.selected_class][2], 125), wx.SOLID))
                if len(self.contourPoints) > 1:
                    self.gc.DrawLines(self.contourPoints)
                    self.dc.DrawLines(self.contourPoints)
                # self.gc.DrawLines(self.contourPoints)
                # self.dc.DrawLines(self.contourPoints)
        elif current_operation == 'edit':
            x, y = e.GetPosition()
            print('here', x, y)
            new_w, new_h = self.dc.GetSize()
            x_scaled, y_scaled = (int(x * IMAGE_WIDTH / new_w), int(y * IMAGE_HEIGHT / new_h))
            #if polygons_to_be_deleted == []:
            for current_polygon in polygons:
                if cv2.pointPolygonTest(current_polygon[2], (x_scaled, y_scaled), True) > 0:
                    polygons_to_be_deleted.append(current_polygon)
                    #polygons_to_be_deleted = removePolygonDuplicates(polygons_to_be_deleted)
                    print('appended')
            redrawSegmentationPolys(polygons_to_be_deleted, highlighted=0)
            self.parent.PrepAndChangeImage(plotAnnotationsOverlay())
            if len(polygons_to_be_deleted) > 0:
                self.parent.m_button31b.Enable()
                self.parent.m_button3.Enable()
                self.parent.m_choiceClass.Enable()
                self.parent.m_choiceClass.SetSelection(polygons_to_be_deleted[0][1])
                self.parent.m_button31c.Enable()
                if len(polygons_to_be_deleted) > 1:
                    self.parent.m_button31p.Enable()
                    self.parent.m_button31n.Enable()

        # self.clear()
        # self.parent.PrepAndChangeImage(self.parent.current_slice)

    def deleteLastVertex(self):
        self.contourPoints = self.contourPoints[:-1]
        self.endDrawing()

    def setDrawingStyle(self, drawing_style):
        self.drawingStyle = drawing_style

    def clear(self):
        dc = wx.ClientDC(self)
        dc.Clear()
        self.contourPoints = []
        self.redrawImage()

    def getCurrentSegmentation(self):
        dc = wx.ClientDC(self)
        img = self.saveSnapshot(dc)
        buf = img.GetDataBuffer()  # use img.GetAlphaBuffer() for alpha data
        arr = np.frombuffer(buf, dtype='uint8')
        arr = np.reshape(arr, (dc.Size.height, dc.Size.width, 3))  # hardcoded num channels :/
        image = np.zeros_like(arr)
        image[np.where((arr == [255, 0, 0]).all(axis=2))] = [255, 255, 255]
        image[np.where((arr == [255, 255, 255]).all(axis=2))] = [255, 255, 255]
        image = cv2.rotate(image, 0)
        image_resized = cv2.resize(image, (self.original_image.Height, self.original_image.Width))
        return image_resized

    def savePolygon(self):
        global polygons
        if len(self.contourPoints) == 0:
            return
        # self.dc = wx.ClientDC(self) #batman
        new_w, new_h = self.dc.GetSize()
        rescaled_contour_points = []
        for point in self.contourPoints:
            rescaled_contour_points.append([int(point[0] * IMAGE_WIDTH / new_w), int(point[1] * IMAGE_HEIGHT / new_h)])

        if self.drawingStyle == 'contour':
            print('saved contour')
            send_to_back = self.parent.checkBox1.GetValue()
            print(image_np_all_slices_segmentation[:, :, self.selected_class].shape)
            approxContourPoints = cv2.approxPolyDP(np.asarray(rescaled_contour_points), 1, closed=True)

            bw_poly = cv2.fillPoly(np.zeros_like(image_np_all_slices_normed[:, :, 0]), [approxContourPoints], 255,
                                   8) > 0

            current_area = np.count_nonzero(bw_poly)

            rmin, rmax, cmin, cmax = bbox_np(bw_poly)

            self.old_area = current_area
            old_area_list.append(current_area)

            if send_to_back:
                for idx_class in range(NUM_CLASSES):
                    bw_poly[np.where(image_np_all_slices_segmentation[:, :, idx_class] > 0)] = 0
                # cv2.imwrite('out_bw_poly.png', np.array(bw_poly*255, dtype=np.uint8))
                contours, hierarchy = cv2.findContours(np.array(bw_poly, dtype=np.uint8), cv2.RETR_LIST,
                                                            cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                for cnt in contours:
                    approxContourPoints = cv2.approxPolyDP(np.asarray(cnt), 1, closed=True)
                    print(cnt)
                    print('problem here')
                    break
            # check dealbreaker
            """ 
            if self.selected_class != 4 and self.selected_class != 7:
                avg_p_x = 0
                avg_p_y = 0
                for p in approxContourPoints:
                    avg_p_x+=p[0][0]
                    avg_p_y+=p[0][1]
                avg_p_x /=len(approxContourPoints)
                avg_p_y /=len(approxContourPoints)
                if cv2.pointPolygonTest(approxContourPoints, (avg_p_x, avg_p_y), True) < 0:
                    style = wx.ICON_ERROR | wx.OK
                    dialog = wx.MessageDialog(None, 'Polygon is not convex', 'Error', style)
                    dialogResult = dialog.ShowModal()
                    dialog.Destroy()
                    self.contourPoints = []
                    self.parent.PrepAndChangeImage(plotAnnotationsOverlay())
                    return
            """
            polygons.append([current_area, self.selected_class, approxContourPoints, (rmin, rmax, cmin, cmax)])
            polygons = sorted(polygons, reverse=True, key=lambda x: x[0])
            drawSegmentationPolys(fromArea=current_area, current_seg=bw_poly, current_class=self.selected_class,
                                  limits=(rmin, rmax, cmin, cmax))

            self.parent.PrepAndChangeImage(plotAnnotationsOverlay())
            self.contourPoints = []

        elif self.drawingStyle == 'brush':
            # approximate curve -- closed does NOT work, for whatever reason
            approxContourPoints = cv2.approxPolyDP(np.asarray(self.contourPoints), 1, closed=True)

    def saveSnapshot(self, dcSource):
        # based largely on code posted to wxpython-users by Andrea Gavana 2006-11-08
        size = dcSource.Size

        # Create a Bitmap that will later on hold the screenshot image
        # Note that the Bitmap must have a size big enough to hold the screenshot
        # -1 means using the current default colour depth
        bmp = wx.Bitmap(size.width, size.height)  #

        # Create a memory DC that will be used for actually taking the screenshot
        memDC = wx.MemoryDC()

        # Tell the memory DC to use our Bitmap
        # all drawing action on the memory DC will go to the Bitmap now
        memDC.SelectObject(bmp)

        # Blit (in this case copy) the actual screen on the memory DC
        # and thus the Bitmap
        memDC.Blit(0,  # Copy to this X coordinate
                   0,  # Copy to this Y coordinate
                   size.width,  # Copy this width
                   size.height,  # Copy this height
                   dcSource,  # From where do we copy?
                   0,  # What's the X offset in the original DC?
                   0  # What's the Y offset in the original DC?
                   )

        # Select the Bitmap out of the memory DC by selecting a new
        # uninitialized Bitmap
        memDC.SelectObject(wx.NullBitmap)

        img = bmp.ConvertToImage()
        return img

    def __del__(self):
        pass


###########################################################################
## Class seg_frame
###########################################################################

class seg_frame(wx.Frame):
    def __init__(self, parent, title):
        self._current_slice = 0
        self.num_slices_current_axis = NUM_SLICES
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=title, pos=wx.DefaultPosition,
                          size=wx.Size(1280, 768), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL, name=u"Seg")

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)

        bSizer1 = wx.BoxSizer(wx.VERTICAL)

        bSizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.bSizer2Ext = bSizer2

        self.m_button5 = wx.Button(self, 83, _("Save frame"), wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer2.Add(self.m_button5, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        # if current_operation == 'segment':
        self.m_button6 = wx.Button(self, 84, _("Save current polygon"), wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button6.Hide()
        bSizer2.Add(self.m_button6, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        # if current_operation == 'segment':
        self.m_button31 = wx.Button(self, 81, _("Delete last vertex"), wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button31.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
        self.m_button31.Hide()
        bSizer2.Add(self.m_button31, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.m_button3 = wx.Button(self, 80, _("Delete current polygon"), wx.DefaultPosition, wx.DefaultSize, 0)
        if current_operation == 'edit':
            self.m_button3.Disable()
        self.m_button3.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

        bSizer2.Add(self.m_button3, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        # if current_operation == 'segment':

        self.checkBox1 = wx.CheckBox(self, 201, _("Send to back"))
        self.checkBox1.Hide()
        bSizer2.Add(self.checkBox1, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.checkBox2 = wx.CheckBox(self, 210, _("Hide seg"))
        self.checkBox2.Hide()
        bSizer2.Add(self.checkBox2, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.m_button31x = wx.Button(self, 202, _("Undo polygon"), wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button31x.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
        self.m_button31x.Hide()
        bSizer2.Add(self.m_button31x, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        # edit mode
        # if current_operation == 'edit':
        self.m_button31p = wx.Button(self, 203, _("Previous polygon"), wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button31p.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
        self.m_button31p.Disable()
        self.m_button31p.Hide()

        bSizer2.Add(self.m_button31p, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.m_button31n = wx.Button(self, 204, _("Next polygon"), wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button31n.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
        self.m_button31n.Disable()
        self.m_button31n.Hide()

        bSizer2.Add(self.m_button31n, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.m_choiceClass = wx.Choice(self, 206, choices=CLASS_NAMES);
        self.m_choiceClass.Disable()
        self.m_choiceClass.Hide()

        bSizer2.Add(self.m_choiceClass, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.m_button31c = wx.Button(self, 207, _("Change class"), wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button31c.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
        self.m_button31c.Disable()
        self.m_button31c.Hide()

        bSizer2.Add(self.m_button31c, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.m_button31b = wx.Button(self, 205, _("Back to segmentation"), wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button31b.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
        self.m_button31b.Disable()
        self.m_button31b.Hide()

        bSizer2.Add(self.m_button31b, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.rbox = wx.RadioBox(self, label=_("Mode"), size= wx.DefaultSize, choices=operations_list_translated, majorDimension=1,
                                style=wx.RA_SPECIFY_ROWS)
        self.rbox.SetSelection(operations_list.index(current_operation))
        self.rbox.Bind(wx.EVT_RADIOBOX, self.OnRadioBox)

        bSizer2.Add(self.rbox, 0, wx.ALL | wx.ALIGN_LEFT, 5)

        self.m_button111 = wx.Button(self, 208, _("Previous"), wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer2.Add(self.m_button111, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.m_button112 = wx.Button(self, 209, _("Next"), wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer2.Add(self.m_button112, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        bSizer1.Add(bSizer2, 0, wx.EXPAND, 5)

        self.bSizer81 = wx.BoxSizer(wx.HORIZONTAL)

        # if current_operation == 'segment':
        self.m_staticText121 = wx.StaticText(self, wx.ID_ANY, _("Draw class"), wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText121.Wrap(-1)

        self.bSizer81.Add(self.m_staticText121, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.buttons_classes = []

        for idxx in range(START_BUTTON_ID_CLASSES, START_BUTTON_ID_CLASSES + NUM_CLASSES):
            current_button = wx.Button(self, idxx, CLASS_NAMES[idxx - START_BUTTON_ID_CLASSES], wx.DefaultPosition,
                                       wx.DefaultSize, 0)
            self.buttons_classes.append(current_button)
            self.bSizer81.Add(self.buttons_classes[-1], 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        # if current_operation == 'segment':
        self.bSizer81.ShowItems(show=False)
        bSizer1.Add(self.bSizer81, 0, wx.ALL, 5)

        bSizer3 = wx.BoxSizer(wx.HORIZONTAL)

        bSizer4 = wx.BoxSizer(wx.VERTICAL)

        bSizer4.Add((0, 0), 1, wx.EXPAND, 5)

        # self.m_bpButton1 = wx.BitmapButton(self, 76, wx.Bitmap(u"arrow_left_half.png", wx.BITMAP_TYPE_ANY),
        #                                   wx.DefaultPosition, wx.DefaultSize, wx.BU_AUTODRAW)

        bSizer4.Add((0, 0), 1, wx.EXPAND, 5)

        # self.m_button11 = wx.Button(self, wx.ID_ANY, u"Previous CT", wx.DefaultPosition, wx.DefaultSize, 0)
        # bSizer4.Add(self.m_button11, 0, wx.ALL, 5)

        bSizer3.Add(bSizer4, 0, wx.EXPAND, 5)

        bSizer8 = wx.BoxSizer(wx.VERTICAL)

        image = wx.Image(IMAGE_WIDTH, IMAGE_HEIGHT)  # EmptyImage
        # image.SetData(np_prep_image.tostring())

        #
        self.drawPane = drawing_panel(self, image, 'contour', 20)  # brush size
        # bSizer8.Add(self.drawPane, 1, wx.ALL | wx.EXPAND, 5)
        bSizer8.Add(self.drawPane, 1, wx.EXPAND, 5)
        # if current_operation == 'segment':
        self.buttons_classes[self.drawPane.selected_class].SetBackgroundColour(
            wx.Colour(CLASS_COLORS_RGB[self.drawPane.selected_class][0],
                      CLASS_COLORS_RGB[self.drawPane.selected_class][1],
                      CLASS_COLORS_RGB[self.drawPane.selected_class][2],
                      125))

        bSizer3.Add(bSizer8, 1, wx.EXPAND, 5)

        bSizer41 = wx.BoxSizer(wx.VERTICAL)

        bSizer41.Add((0, 0), 1, wx.EXPAND, 5)

        bSizer41.Add((0, 0), 1, wx.EXPAND, 5)

        # self.m_button111 = wx.Button(self, wx.ID_ANY, u"Next", wx.DefaultPosition, wx.DefaultSize, 0)
        # bSizer41.Add(self.m_button111, 0, wx.ALL | wx.ALIGN_RIGHT, 5)

        bSizer3.Add(bSizer41, 0, wx.EXPAND, 5)

        bSizer7 = wx.BoxSizer(wx.VERTICAL)

        bSizer3.Add(bSizer7, 1, wx.EXPAND, 5)

        bSizer1.Add(bSizer3, 1, wx.EXPAND, 5)

        self.initOperation()

        self.SetSizer(bSizer1)
        self.Layout()

        self.Centre(wx.BOTH)

        self.Bind(wx.EVT_MOVE, self.OnMove)
        self.Bind(wx.EVT_BUTTON, self.OnButtonClicked)
        self.Bind(wx.EVT_SCROLL, self.OnScrollChanged)
        self.Bind(wx.EVT_COMBOBOX, self.OnComboboxChanged)
        self.Bind(wx.EVT_CHAR_HOOK, self.OnKeyDown)
        self.Bind(wx.EVT_CHECKBOX, self.OnCheckBoxChanged)
        # self.m_comboBox11.Bind(wx.EVT_COMBOBOX, self.OnComboboxChanged)
        # this isn't supposed to be here, but otherwise it fails to draw the segmentation for the 1st slice
        self.PrepAndChangeImage(plotAnnotationsOverlay())

    @property
    def current_slice(self):
        return self._current_slice

    @current_slice.setter
    def current_slice(self, value):
        if value < 0:
            self._current_slice = 0
        elif value > self.num_slices_current_axis - 1:
            self._current_slice = self.num_slices_current_axis - 1
        else:
            self._current_slice = value
        self.m_comboBox1.Value = str(self._current_slice + 1)

    def initOperation(self):
        if current_operation == 'segment':
            # show
            self.m_button6.Show()
            self.m_button31.Show()
            self.checkBox1.Show()
            self.checkBox2.Show()
            self.m_button31x.Show()
            self.bSizer81.ShowItems(show=True)
            # hide
            self.m_button31p.Hide()
            self.m_button31n.Hide()
            self.m_button31c.Hide()
            self.m_button31b.Hide()
            self.m_choiceClass.Hide()
            # enable
            self.m_button3.Enable()

            # bug
            #self.rbox.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT))
            self.rbox.Refresh()
            #self.bSizer2Ext.Layout()
        if current_operation == 'edit':
            # hide
            self.m_button6.Hide()
            self.m_button31.Hide()
            self.checkBox1.Hide()
            self.checkBox2.Hide()
            self.m_button31x.Hide()
            self.bSizer81.ShowItems(show=False)
            # show
            self.m_button31p.Show()
            self.m_button31n.Show()
            self.m_button31c.Show()
            self.m_button31b.Show()
            self.m_choiceClass.Show()
            # disable
            self.m_button3.Disable()
            # delete
            self.drawPane.contourPoints = []
            # bug
            self.rbox.Refresh()
            #self.rbox.Layout()

    def OnRadioBox(self, e):
        global current_operation
        global operation_changed
        #operation_changed = True
        current_operation = operations_list[self.rbox.GetSelection()]
        # self.__init__(self, "segmentation")
        self.initOperation()
        # self.mainLogin.Hide()
        self.Layout()
        # self.__init__(None, title="lala")
        # self.Close()

    def OnMove(self, e):
        x, y = e.GetPosition()
        # print("current window position x = ", x, " y= ", y)

    def OnKeyDown(self, e):
        # print(e.GetKeyCode())
        if e.GetKeyCode() == 32:
            # space -- save polygon
            self.drawPane.savePolygon()
        elif e.GetKeyCode() == 8:
            # backspace -- delete current vertex
            self.drawPane.deleteLastVertex()

    def OnButtonClicked(self, e):
        global polygons
        global polygons_to_be_deleted
        global current_polygon_to_be_deleted_idx
        global image_np_all_slices_segmentation
        global old_area_list
        global current_operation
        global operation_changed
        global FILE_PATH
        if e.Id == 76:  # load left slice
            if self.current_slice > 0:
                # self.SaveCurrentSegmentation()
                self.current_slice -= 1
                self.m_slider1.Value = self.current_slice
                self.PrepAndChangeImage(self.current_slice)
        elif e.Id == 77:  # load right slice
            if self.current_slice < self.num_slices_current_axis - 1:
                # self.SaveCurrentSegmentation()
                self.current_slice += 1
                self.m_slider1.Value = self.current_slice
                self.PrepAndChangeImage(self.current_slice)
        elif e.Id == 80:  # proper erase?
            if current_operation == 'edit':
                oal = polygons_to_be_deleted[current_polygon_to_be_deleted_idx]
                polygons_new = []
                for elem in polygons:
                    if elem[0] != oal[0]:
                        polygons_new.append(elem)
                polygons_to_be_deleted = []
                current_polygon_to_be_deleted_idx = 0
                polygons = polygons_new.copy()
                self.drawPane.contourPoints = []
                image_np_all_slices_segmentation = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES), dtype=np.bool_)
                redrawSegmentationPolys(polygons)
                self.PrepAndChangeImage(plotAnnotationsOverlay())
                self.m_button3.Disable()
                self.m_button31b.Disable()
                self.m_button31p.Disable()
                self.m_button31n.Disable()
                self.m_choiceClass.Disable()
                self.m_button31c.Disable()
            else:
                # self.drawPane.clear()
                # self.EraseCurrentSegmentation()
                self.drawPane.contourPoints = []
                self.PrepAndChangeImage(plotAnnotationsOverlay())
                """
                if  self.m_button3.GetForegroundColour().RGB == 0:
                    self.m_button3.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT))
                else:
                    self.m_button3.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
                """
        elif e.Id == 81:  # delete last vertex
            self.drawPane.deleteLastVertex()
        elif e.Id == 83:  # save all
            self.SaveSegmentationToFile()
        elif e.Id == 84:  # save polygon
            self.drawPane.savePolygon()
            # image_np_all_slices_segmentation = np.zeros_like(image_np_all_slices)
        elif e.Id == 85:  # contour
            self.drawPane.setDrawingStyle('contour')
            self.m_button61.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT))
            self.m_button611.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
            self.m_button6111.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

        elif e.Id == 86:  # brush
            self.drawPane.setDrawingStyle('brush')
            self.m_button61.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
            self.m_button611.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT))
            self.m_button6111.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

        elif e.Id == 89:  # brush delete
            self.drawPane.setDrawingStyle('brushDelete')
            self.m_button61.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
            self.m_button611.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
            self.m_button6111.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT))
        elif e.Id >= START_BUTTON_ID_CLASSES and e.Id <= START_BUTTON_ID_CLASSES + NUM_CLASSES:
            for idxx in range(START_BUTTON_ID_CLASSES, START_BUTTON_ID_CLASSES + NUM_CLASSES):
                self.buttons_classes[idxx - START_BUTTON_ID_CLASSES].SetForegroundColour(
                    wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))

                self.buttons_classes[idxx - START_BUTTON_ID_CLASSES].SetBackgroundColour(
                    wx.Colour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)))
            self.drawPane.selected_class = e.Id - START_BUTTON_ID_CLASSES
            self.buttons_classes[e.Id - START_BUTTON_ID_CLASSES].SetBackgroundColour(
                wx.Colour(CLASS_COLORS_RGB[self.drawPane.selected_class][0],
                          CLASS_COLORS_RGB[self.drawPane.selected_class][1],
                          CLASS_COLORS_RGB[self.drawPane.selected_class][2],
                          125))
            self.buttons_classes[e.Id - START_BUTTON_ID_CLASSES].SetForegroundColour(
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT))
        elif e.Id == 202:
            if len(old_area_list) > 0:
                polygons_new = []
                for elem in polygons:
                    if elem[0] != old_area_list[-1]:
                        polygons_new.append(elem)
                old_area_list = old_area_list[:-1]
                polygons = polygons_new.copy()
                self.drawPane.contourPoints = []
                image_np_all_slices_segmentation = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES), dtype=np.bool_)
                redrawSegmentationPolys(polygons)
                self.PrepAndChangeImage(plotAnnotationsOverlay())
        elif e.Id == 203:
            if current_polygon_to_be_deleted_idx - 1 < 0:
                current_polygon_to_be_deleted_idx = len(polygons_to_be_deleted) - 1
            else:
                current_polygon_to_be_deleted_idx -= 1
            redrawSegmentationPolys(polygons_to_be_deleted, highlighted=current_polygon_to_be_deleted_idx)
            self.m_choiceClass.SetSelection(polygons_to_be_deleted[current_polygon_to_be_deleted_idx][1])
            print('pdbd idx', current_polygon_to_be_deleted_idx)
            self.PrepAndChangeImage(plotAnnotationsOverlay())
        elif e.Id == 204:
            if current_polygon_to_be_deleted_idx + 1 > len(polygons_to_be_deleted) - 1:
                current_polygon_to_be_deleted_idx = 0
            else:
                current_polygon_to_be_deleted_idx += 1
            redrawSegmentationPolys(polygons_to_be_deleted, highlighted=current_polygon_to_be_deleted_idx)
            self.m_choiceClass.SetSelection(polygons_to_be_deleted[current_polygon_to_be_deleted_idx][1])
            print('pdbd idx', current_polygon_to_be_deleted_idx)

            self.PrepAndChangeImage(plotAnnotationsOverlay())
        elif e.Id == 205:
            polygons_to_be_deleted = []
            self.m_button3.Disable()
            self.m_button31p.Disable()
            self.m_button31n.Disable()
            self.m_button31b.Disable()
            self.m_choiceClass.Disable()
            self.m_button31c.Disable()
            redrawSegmentationPolys(polygons)
            self.PrepAndChangeImage(plotAnnotationsOverlay())
        elif e.Id == 207:
            self.m_button3.Disable()
            self.m_button31p.Disable()
            self.m_button31n.Disable()
            self.m_button31b.Disable()
            self.m_choiceClass.Disable()
            self.m_button31c.Disable()
            oal = polygons_to_be_deleted[current_polygon_to_be_deleted_idx]
            polygons_new = []
            for elem in polygons:
                if elem[0] != oal[0]:
                    polygons_new.append(elem)
                else:
                    polygons_new.append([elem[0], self.m_choiceClass.GetSelection(), elem[2], elem[3]])
            polygons = polygons_new.copy()
            polygons_to_be_deleted = []
            self.drawPane.contourPoints = []
            image_np_all_slices_segmentation = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES), dtype=np.bool_)
            redrawSegmentationPolys(polygons)
            self.PrepAndChangeImage(plotAnnotationsOverlay())
        elif e.Id == 208: # prev
            file_dir = os.path.dirname(FILE_PATH)
            print(file_dir)
            file_list = sorted(os.listdir(file_dir))
            file_idx = file_list.index(os.path.basename(FILE_PATH))
            new_path = ''
            while file_idx > 0:
                file_idx -= 1
                if '.jpg' in file_list[file_idx]:
                    new_path = file_list[file_idx]
                    break
            if new_path != '':
                FILE_PATH = os.path.join(file_dir, new_path)
                self.Close()
                loadFile()
                operation_changed = True
                #redrawSegmentationPolys(polygons)
                self.PrepAndChangeImage(plotAnnotationsOverlay())
            current_polygon_to_be_deleted_idx = 0
            #current_operation = operations_list[self.rbox.GetSelection()]
            # self.__init__(self, "segmentation")
            # change file path

            #self.initOperation()
            # self.mainLogin.Hide()
            # self.Layout()
            # self.__init__(None, title="lala")
            # self.Close()
        elif e.Id == 209: # next
            file_dir = os.path.dirname(FILE_PATH)
            print(file_dir)
            file_list = sorted(os.listdir(file_dir))
            file_idx = file_list.index(os.path.basename(FILE_PATH))
            new_path = ''
            while file_idx < len(file_list)-1:
                file_idx += 1
                if '.jpg' in file_list[file_idx]:
                    new_path = file_list[file_idx]
                    break
            if new_path != '':
                FILE_PATH = os.path.join(file_dir, new_path)
                self.Close()
                loadFile()
                operation_changed = True
                #redrawSegmentationPolys(polygons)
                self.PrepAndChangeImage(plotAnnotationsOverlay())
            current_polygon_to_be_deleted_idx = 0
    def OnScrollChanged(self, e):
        if e.Id == 78:  # slice scroll
            # self.SaveCurrentSegmentation()
            self.current_slice = e.Position
            self.PrepAndChangeImage(self.current_slice)
        elif e.Id == 82:  # contrast
            # self.SaveCurrentSegmentation()
            self.clahe = cv2.createCLAHE(clipLimit=e.Position, tileGridSize=(8, 8))
            self.PrepAndChangeImage(self.current_slice)

    def OnComboboxChanged(self, e):
        if e.Id == 79:  # num_slice
            # self.SaveCurrentSegmentation()
            new_slice_index = e.Selection - 1
            self.current_slice = new_slice_index
            self.PrepAndChangeImage(self.current_slice)
        elif e.Id == 87:  # brush size
            self.drawPane.brushSize = int(e.String)
        elif e.Id == 88:  # view axis change
            if e.String != self.current_axis:
                self.current_axis = e.String
                self.current_slice = 0
                self.PrepAndChangeImage(self.current_slice)

                self.num_slices_current_axis = NUM_SLICES

                # reset slider + slice selector
                comboBox1Choices = list(map(str, range(1, self.num_slices_current_axis)))
                self.m_comboBox1.Items = comboBox1Choices

                self.m_slider1.Max = self.num_slices_current_axis

    def OnCheckBoxChanged(self, e):
        global HIDE_SEGMENTATION
        if e.Id == 210: # hide segmentation
            if self.checkBox2.GetValue() is True:
                HIDE_SEGMENTATION = True
                self.PrepAndChangeImage(plotAnnotationsOverlay())
            else:
                HIDE_SEGMENTATION = False
                self.PrepAndChangeImage(plotAnnotationsOverlay())


    def PrepAndChangeImage(self, slice):
        # print(image_np_all_slices_normed.shape)
        try:
            c_w, c_h, _ = slice.shape
        except:
            slice = image_np_all_slices_normed
        # np_prep_image = np.array(slice, dtype=np.uint8)
        np_prep_image = slice
        c_w, c_h, _ = np_prep_image.shape
        image = wx.Image(c_h, c_w)
        image.SetData(np_prep_image)
        self.drawPane.changeImage(image)

    def SaveCurrentSegmentation(self):
        current_segmentation = self.drawPane.getCurrentSegmentation()
        image_np_all_slices_segmentation[:, :, self.current_slice] = np.maximum(
            image_np_all_slices_segmentation[:, :, self.current_slice], current_segmentation[:, :, 0])
        self.drawPane.clear()

    def EraseCurrentSegmentation(self):
        image_np_all_slices_segmentation[:, :, self.current_slice] = 0

        self.drawPane.clear()

    def SaveSegmentationToFile(self):
        np.savez_compressed(
            os.path.join(SEG_PATH_OUT, 'seg_' + os.path.basename(FILE_PATH).replace('.' + IMAGE_EXTENSION, '.npz')),
            np.array(image_np_all_slices_segmentation > 0, dtype=np.bool_))
        with open(os.path.join(SEG_PATH_OUT,
                               'seg_' + os.path.basename(FILE_PATH).replace('.' + IMAGE_EXTENSION, '_polys.pkl')),
                  'wb') as fp:
            pickle.dump(polygons, fp)
        if not os.path.exists(os.path.join(SEG_PATH_OUT, 'segmentation_preview')):
            os.makedirs(os.path.join(SEG_PATH_OUT, 'segmentation_preview'))
        cv2.imwrite(os.path.join(SEG_PATH_OUT, 'segmentation_preview',
                                 'seg_rgb_' + os.path.basename(FILE_PATH).replace('.' + IMAGE_EXTENSION, '.png')),
                    plotAnnotationsOverlay()[:, :, ::-1])
        # current_annotations = plotAnnotations()
        # current_annotations = current_annotations[BORDER_PADDING:IMAGE_HEIGHT - BORDER_PADDING,
        #                      BORDER_PADDING:IMAGE_WIDTH - BORDER_PADDING, :]
        # current_annotations = cv2.cvtColor(current_annotations, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(SEG_PATH_OUT, 'seg_' + os.path.basename(FILE_PATH)), current_annotations)

    def __del__(self):
        pass


###########################################################################
## Class display_frame
###########################################################################

class display_frame(wx.Frame):
    def __init__(self, parent, title):
        self._current_slice = 0
        self.num_slices_current_axis = NUM_SLICES

        self.frame = wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=title, pos=wx.DefaultPosition,
                                       size=wx.Size(1280, 768), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL,
                                       name=u"Seg")

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)

        bSizer1 = wx.BoxSizer(wx.VERTICAL)

        self.image = wx.Image(IMAGE_WIDTH, IMAGE_HEIGHT)  # EmptyImage
        self.image.SetData(image_previous_segmentation_overlaid.tostring())

        self.bitmap = self.image.ConvertToBitmap()

        self.m_bitmap1 = wx.StaticBitmap(self, wx.ID_ANY, self.bitmap, wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer1.Add(self.m_bitmap1, 1, wx.ALL | wx.EXPAND, 5)

        self.SetSizer(bSizer1)
        self.Layout()

        self.Centre(wx.BOTH)

        self.Bind(wx.EVT_SIZE, self.onResize)

    def redrawImage(self):
        self.dc = wx.ClientDC(self)  # batman
        new_w, new_h = self.dc.GetSize()
        print("nwh", new_w, new_h)
        if new_w != 0 and new_h != 0:
            self.bitmap = self.image.Scale(new_w, new_h).ConvertToBitmap()
            # self.dc.DrawBitmap(self.resized_bitmap, 0, 0)
            self.m_bitmap1.SetBitmap(self.bitmap)

    def onResize(self, e=None):
        W, H = self.m_bitmap1.Size
        if W > H:
            NewW = W
            NewH = W * H / W
        else:
            NewH = H
            NewW = H * W / H
        img = self.image
        img = img.Scale(NewW, NewH)
        self.m_bitmap1.SetBitmap(wx.Bitmap(img))
        e.Skip()
        self.Refresh()


def plotPreviousSegmentations(path):
    for current_image_path in os.listdir(path):
        if not 'jpg' in current_image_path or 'png' in current_image_path:
            continue
        if os.path.exists(os.path.join(path, 'segmentation_preview',
                                 'seg_rgb_' + os.path.basename(current_image_path).replace('.' + IMAGE_EXTENSION,
                                                                                  '.png'))):
            continue
        image_np_all_slices_normed_inside = []
        image_np_all_slices_segmentation_inside = []
        try:
            image_np_all_slices_normed_inside = cv2.imread(os.path.join(SEG_PATH_OUT, current_image_path))
            image_np_all_slices_segmentation_inside = np.load(
                os.path.join(path, 'seg_' + os.path.basename(current_image_path).replace('.jpg', '.npz')))[
                'arr_0']
            image_np_all_slices_normed_inside = cv2.copyMakeBorder(image_np_all_slices_normed_inside, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING, cv2.BORDER_CONSTANT)
            if not os.path.exists(os.path.join(path, 'segmentation_preview')):
                os.makedirs(os.path.join(path, 'segmentation_preview'))
            cv2.imwrite(os.path.join(path, 'segmentation_preview',
                                     'seg_rgb_' + os.path.basename(current_image_path).replace('.' + IMAGE_EXTENSION,
                                                                                      '.png')),
                        plotAnnotationsOverlay2(image_np_all_slices_normed_inside, image_np_all_slices_segmentation_inside))
        except:
            pass

def loadFile():
    global image_previous_segmentation_overlaid
    global image_np_all_slices_segmentation
    global polygons
    global image_np_all_slices_normed
    global IMAGE_HEIGHT, IMAGE_WIDTH, NUM_SLICES
    global current_image
    global image_np_all_slices
    global PREV_FILE_PATH

    current_image = cv2.imread(FILE_PATH, 1)
    current_image = cv2.copyMakeBorder(current_image, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING,
                                       BORDER_PADDING, cv2.BORDER_CONSTANT)
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

    image_np_all_slices = current_image

    #if image_np_all_slices.shape[0] == 2300:
    #    image_np_all_slices = cv2.resize(image_np_all_slices, None, fx=0.5, fy=0.5)

    IMAGE_HEIGHT, IMAGE_WIDTH, NUM_SLICES = image_np_all_slices.shape
    image_np_all_slices_normed = image_np_all_slices.copy()
    try:

        current_listdir = sorted(os.listdir(os.path.dirname(FILE_PATH)))

        current_basename = os.path.basename(FILE_PATH)

        current_idx = current_listdir.index(current_basename)

        putative_previous_basename = current_listdir[current_idx -1]

        if not (current_basename.split('_')[1] == putative_previous_basename.split('_')[1]):
            raise ValueError('No previous frames found.')


        image_previous_segmentation_overlaid = cv2.imread(os.path.join(SEG_PATH_OUT, 'segmentation_preview',
                                                                       'seg_rgb_' + putative_previous_basename.replace('.' + IMAGE_EXTENSION,
                                                                                              '.png')))
        image_previous_segmentation_overlaid = image_previous_segmentation_overlaid[:, :, ::-1]
        PREV_FILE_PATH = putative_previous_basename.split('.')[0]
    except:
        image_previous_segmentation_overlaid = []
    try:
        # current_image_seg = np.load(os.path.join(os.path.dirname(FILE_PATH), 'seg_'+os.path.basename(FILE_PATH).replace('.png', '.npz')))
        image_np_all_slices_segmentation = \
            np.load(os.path.join(SEG_PATH_OUT,
                                 'seg_' + os.path.basename(FILE_PATH).replace('.' + IMAGE_EXTENSION, '.npz')))[
                'arr_0']
        #if image_np_all_slices_segmentation.shape[0] == 2300:
        #    image_np_all_slices_segmentation = np.array(resize(image_np_all_slices_segmentation, (1150, 2118),preserve_range=True), dtype=np.bool_)
        with open(os.path.join(SEG_PATH_OUT,
                               'seg_' + os.path.basename(FILE_PATH).replace('.' + IMAGE_EXTENSION, '_polys.pkl')),
                  'rb') as fp:
            polygons_old = pickle.load(fp)
            #areas = []
            polygons = removePolygonDuplicates(polygons_old)
            #remove duplicates somehow
            #for p in polygons_old:
                # print(p[0])
                #current_area = int(p[0])
                # if current_area not in areas:
                # polygons.append(p)
                #areas.append(current_area)
                #areas = list(set(areas))
            # for p in polygons:
            #    print(p[0])

    except:
        image_np_all_slices_segmentation = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES), dtype=np.bool_)
        polygons = []


if __name__ == '__main__':
    app = wx.App(False)

    current_operation = 'segment'  # selectOperation()
    image_path_list = []

    if current_operation == 'segment' or current_operation == 'edit':
        FILE_PATH = getSingleFilePath("*.png;*.jpg;*.jpeg;*.JPEG;*.PNG;")
    elif current_operation == 'cancel':
        sys.exit(0)

    IMAGE_EXTENSION = os.path.basename(FILE_PATH).split('.')[-1]

    SEG_PATH_OUT = os.path.dirname(FILE_PATH)

    # plot previous segmentations
    previousSegemntationPlotThread = threading.Thread(target=plotPreviousSegmentations, args=[SEG_PATH_OUT])
    previousSegemntationPlotThread.start()

    loadFile()

    while operation_changed:
        print('new frame instance')
        top = seg_frame(None, title=os.path.basename(FILE_PATH).split('.')[0])
        print('instancing')
        try:
            if image_previous_segmentation_overlaid != []:
                print('bottom found')
                if PREV_FILE_PATH != '':
                    bottom = display_frame(top, title=os.path.basename(PREV_FILE_PATH).split('.')[0])
                else:
                    bottom = display_frame(top, title=_("Previous segmentation"))
                bottom.SetPosition((0, 0))
                bottom.Show()
        except:
            pass
        top.Show()
        print('showing')
        app.MainLoop()
        print('end main loop')
