import numpy as np
import os
from manopth.manolayer import ManoLayer
import torch 
import pickle
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from utils import viewJointswObj
import cv2

path_root = r"F:\zbh_codes\annotation_tools\Annotations_Tools\sample_imgs"
joints2d = np.load(os.path.join(path_root,"00000023_kpts_2d_glob_r.npy"))
print(joints2d)
image = cv2.imread(r"F:\zbh_codes\annotation_tools\Annotations_Tools\sample_imgs\00000023.jpg")
image = cv2.flip(image, 1)
for i in range(len(joints2d)):
    kpoints = joints2d[i]
    image = cv2.circle(image, (int(kpoints[0]*150), int(kpoints[1]*150)), 3, (255, 0, 0), -1)
cv2.imshow("joints2d", image)
cv2.waitKey(-1)
mano_param = np.load(os.path.join(path_root,"00000023_mano_r.npy"))
print(mano_param.shape)
mano_right = ManoLayer(
        mano_root='models/MANO_layer/manopth/mano/models', use_pca=True, ncomps=45, flat_hand_mean=True, side='right')
trans_param = mano_param[:3]
pose_param = mano_param[3:]

verts, joints = mano_right(torch.tensor([pose_param]), torch.zeros(1), torch.tensor([trans_param]))
face_R = pickle.load(open(r'models\MANO_layer\manopth\mano\models\MANO_RIGHT.pkl',"rb"),encoding="latin1")['f']

verts_R = np.array(verts[0]).astype(np.float32)
joints= np.array(joints[0]).astype(np.float32)
print(verts_R)

joints3d = np.load(os.path.join(path_root,"00000023_kpts_3d_glob_r.npy"))
print(joints3d)
viewJointswObj([joints3d],[verts_R,face_R])
