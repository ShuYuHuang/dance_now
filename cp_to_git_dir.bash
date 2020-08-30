cp ./utils/loaders.py ~/AT091_18_Orig_Character/Api/utils/
<<<<<<< HEAD
cp ./utils/integration.py ~/AT091_18_Orig_Character/Api/utils/
cp ./utils/resnet_model.py ~/AT091_18_Orig_Character/Api/utils/
cp ./utils/BASNet.py ~/AT091_18_Orig_Character/Api/utils/

cp ./inference_openpose.py ~/AT091_18_Orig_Character/Api/
cp ./try_preload.py ~/AT091_18_Orig_Character/Api/
=======
cp ./inference_openpose.py ~/AT091_18_Orig_Character/Api/

mkdir ~/AT091_18_Orig_Character/Api/model_openpose
cp ./model_face/netGface_struct.pth ~/AT091_18_Orig_Character/Api/model_openpose/
cp ./model_face/netGface_run370.pt ~/AT091_18_Orig_Character/Api/model_openpose/
cp ./model_face/netDface_struct.pth ~/AT091_18_Orig_Character/Api/model_openpose/
cp ./model_face/netDface_run370.pt ~/AT091_18_Orig_Character/Api/model_openpose/

cp ./model_body/netGbody_struct.pth ~/AT091_18_Orig_Character/Api/model_openpose/
cp ./model_body/netGbody_run460.pt ~/AT091_18_Orig_Character/Api/model_openpose/
cp ./model_body/netDbody_struct.pth ~/AT091_18_Orig_Character/Api/model_openpose/
cp ./model_body/netDbody_run460.pt ~/AT091_18_Orig_Character/Api/model_openpose/


mkdir -p ~/AT091_18_Orig_Character/Api/data/source/keypoints
cp ./data/anime/train_label/img_1_0_keypoints.json ~/AT091_18_Orig_Character/Api/data/source/keypoints/
>>>>>>> 72dfc472d73fe993ea067735b47f10c67b2b0a65
