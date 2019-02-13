# -*- coding: utf-8 -*-
# @Author: Yongqiang Qin
# @Date:   2018-06-07 23:40:25
# @Last Modified by:   Yongqiang Qin
# @Last Modified time: 2018-07-14 13:33:39

from lxml import etree

class GEN_Annotations:
    def __init__(self, folder, filename, path):
        self.root = etree.Element("annotation")

        child1 = etree.SubElement(self.root, "folder")
        child1.text = folder

        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename

        child3 = etree.SubElement(self.root, "path")
        child3.text = path
        # child2.set("database", "The VOC2007 Database")


    def set_size(self,witdh,height,channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "channel")
        channeln.text = str(channel)

    def savefile(self,filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

    def add_pic_attr_xywh(self,label,x,y,w,h):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        
        posen = etree.SubElement(object, "pose")
        posen.text = 'Unspecified'
        truncatedn = etree.SubElement(object, "truncated")
        truncatedn.text = '0'
        difficultn = etree.SubElement(object, "difficult")
        difficultn.text = '0'

        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(x)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(y)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(x+w)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(y+h)

    def add_pic_attr_xyxy(self,label,x1,y1,x2,y2, pose='Unspecified', trunc='0', difficult='0'):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label

        posen = etree.SubElement(object, "pose")
        posen.text = pose

        truncatedn = etree.SubElement(object, "truncated")
        truncatedn.text = trunc
        difficultn = etree.SubElement(object, "difficult")
        difficultn.text = difficult

        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(x1)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(y1)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(x2)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(y2)

