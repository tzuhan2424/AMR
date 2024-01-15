"""
viz the naive cam, spotlight cam, compensation map, mixmap
"""
#%%
from help.camHelper import compareDifferentCam, compareGTandCamROI

#%%
# 'result/cam_merge' 'result/cam_spotlight' 'result_previous/cams' 'result_previous/cams_spotlight'
# compareDifferentCam('result/cam_merge', 'result/cam_spotlight','result_previous/cams', 'result_previous/cams_spotlight')


    
#%%

    #now we have use selective search to get the ROI
compareGTandCamROI(cam ='result/cam_merge', isShowPic=True, isSaveROI = False)
    #%%
# compareGTandCamROI(cam ='result/cams_compensation', isShowPic=True)
#     #%%
# compareGTandCamROI(cam ='result/cams_spotlight', isShowPic=True)





    


