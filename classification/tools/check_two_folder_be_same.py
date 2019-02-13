# encoding=utf-8

import os
import tensorflow as tf

FLAG = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("folderA", "",
                           "folder A")
tf.app.flags.DEFINE_string("folderB", "",
                           "folder B")
tf.app.flags.DEFINE_boolean("skip_empty", False,
                            "ignore empty folder")


def check_two_folder_be_same(folder0, folder1, skip_empty=True):
    # print("Checking: %s <-> %s" % (os.path.split(folder0)[-1] if folder0 else "",
    #                                os.path.split(folder1)[-1] if folder1 else ""))

    # get all sub files in two folder
    folder0_file_names = []
    if folder0:
        folder0_file_names = os.listdir(folder0)
    folder1_file_names = []
    if folder1:
        folder1_file_names = os.listdir(folder1)

    for name in set(folder0_file_names + folder1_file_names):
        sub_path0 = None
        if name in folder0_file_names:
            sub_path0 = os.path.join(folder0, name)
        sub_path1 = None
        if name in folder1_file_names:
            sub_path1 = os.path.join(folder1, name)

        if sub_path0 and sub_path1:
            isdir0 = os.path.isdir(sub_path0)
            isdir1 = os.path.isdir(sub_path1)
            if isdir0 and isdir1:
                check_two_folder_be_same(sub_path0, sub_path1, skip_empty)
            elif (isdir0 and not isdir1) or (isdir0 and not isdir1):
                print("Miss match: one file one folder. (%s <-> %s)" %
                      (sub_path0.replace(os.path.dirname(folder0), ""), sub_path1.replace(os.path.dirname(folder1), "")))
        elif sub_path0 and not sub_path1:
            isdir0 = os.path.isdir(sub_path0)
            if skip_empty and isdir0:
                check_two_folder_be_same(sub_path0, sub_path1, skip_empty)
            else:
                print("Miss match: one folder/file one null. (%s <-> )" % sub_path0.replace(os.path.dirname(folder0), ""))
        elif not sub_path0 and sub_path1:
            isdir1 = os.path.isdir(sub_path1)
            if skip_empty and isdir1:
                check_two_folder_be_same(sub_path0, sub_path1, skip_empty)
            else:
                print("Miss match: one folder/file one null. ( <-> %s)" % sub_path1.replace(os.path.dirname(folder1), ""))


def main(_):
    check_two_folder_be_same(FLAG.folderA, FLAG.folderB,
                             FLAG.skip_empty)
    print("Check done!")


if __name__ == '__main__':
    tf.app.run()
