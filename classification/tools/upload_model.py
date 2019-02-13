#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-10-29 上午11:33
# @Author  : Jh Zhao
# @Site    : 
# @File    : upload_model.py
# @Software: PyCharm Community Edition

"""
upload model to some where

require fabric3

"""

import os
import argparse

from fabric.api import *

TARED_MODEL_NAME = "model.tar.gz"
TARED_MODEL_TEMP_DIR = os.path.join(os.path.expanduser("~"), "tared_models")


def tar_model(model_dir):
    tar_model_cmd = "cd {folder} && tar -czf {target} ./* --exclude *frozen_inference_graph.pb".format(folder=model_dir,
                                                                                                       target=TARED_MODEL_NAME)
    local(tar_model_cmd)
    # local("[[ ! -e '{fld}' ]] && mkdir '{fld}'".format(fld=TARED_MODEL_TEMP_DIR))
    if not os.path.exists(TARED_MODEL_TEMP_DIR):
        os.makedirs(TARED_MODEL_TEMP_DIR)
    move_model_cmd = "mv {target} {obj}".format(target=os.path.join(model_dir, TARED_MODEL_NAME),
                                                obj=os.path.join(TARED_MODEL_TEMP_DIR, TARED_MODEL_NAME))
    local(move_model_cmd)


@parallel(5)
def make_dirs(dirs):
    # names = dirs.split("/")
    # st = 1 if names[0] == "" else 0
    # for i in range(st, len(names)):
    #     fld = "/".join(names[:i + 1]) + "/"
    #     run("[[ ! -e '{fld}' ]] && mkdir '{fld}'".format(fld=fld))
    run(""" python -c "import os; os.makedirs('{fld}') if not os.path.exists('{fld}') else '' " """.format(fld=dirs))


@parallel(5)
def remove_dirs(dirs):
    run(""" python -c "import os, shutil; shutil.rmtree('{fld}') if os.path.exists('{fld}') else '' " """.format(
        fld=dirs))


@parallel(5)
def upload_model_to_server(tared_model_path, remote_model_path):
    remote_dir = os.path.dirname(remote_model_path)
    # remove_dirs(remote_dir)
    with settings(warn_only=True):
        run(""" rm -r '{fld}' """.format(fld=remote_dir))
    temp_dir = remote_dir.rstrip("/") + "_"
    temp_path = os.path.join(temp_dir, os.path.basename(remote_model_path))
    make_dirs(temp_dir)
    put(tared_model_path, temp_path)
    untar_cmd = "cd {fld} && tar -xzf {target} && rm {target}".format(fld=temp_dir, target=TARED_MODEL_NAME)
    run(untar_cmd)
    run("mv {fld0} {fld1}".format(fld0=temp_dir, fld1=remote_dir))
    run("ls -al {fld}".format(fld=remote_dir))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="model directory")
    parser.add_argument("--upload_dir", required=True, help="model upload directory")
    parser.add_argument("--user_at_host", required=True, help="remote host and user name")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    user, host = args.user_at_host.split("@")
    env["host_string"] = host
    env["user"] = user
    env["key_filename"] = os.path.expanduser("~/.ssh/id_rsa.pub")

    tar_model(args.model_dir)
    path = os.path.join(TARED_MODEL_TEMP_DIR, TARED_MODEL_NAME)
    upload_model_to_server(path, os.path.join(args.upload_dir, TARED_MODEL_NAME))
