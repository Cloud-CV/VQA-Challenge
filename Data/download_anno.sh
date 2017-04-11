#! /bin/sh
#
# download_anno.sh
# Copyright (C) 2016 prithv1 <prithv1@vt.edu>
#
# Distributed under terms of the MIT license.
wget https://filebox.ece.vt.edu/~prithv1/eval_dummy_split.tar.gz && tar -xvzf eval_dummy_split.tar.gz && rm -rf eval_dummy_split.tar.gz && cp ../Results/OpenEnded_mscoco_train2014_fake_results.json VQA_jsons/
