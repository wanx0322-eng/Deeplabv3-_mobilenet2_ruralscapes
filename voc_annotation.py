import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   想要增加测试集修改trainval_percent 
#   修改train_percent用于改变验证集的比例 9:1
#   
#   当前该库将测试集当作验证集使用，不单独划分测试集
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.7
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOCdevkit_path      = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    
    #-------------------------------------------------------------------#
    #   只有【索引图】标签才能参与划分。
    #   RGB / RGBA 三通道标签进入训练集后会在 one-hot 处直接崩溃，
    #   所以这里先把它们剔除并报出来，用 tools/fix_rgb_masks.py 转换后再划分。
    #-------------------------------------------------------------------#
    temp_seg = os.listdir(segfilepath)
    total_seg, bad_seg = [], []
    for seg in temp_seg:
        if not seg.endswith(".png"):
            continue
        with Image.open(os.path.join(segfilepath, seg)) as im:
            if im.mode in ("P", "L"):
                total_seg.append(seg)
            else:
                bad_seg.append((seg, im.mode))

    if bad_seg:
        print("\n[Warning] 以下 %d 个标签不是灰度图/八位彩图，已跳过，不会进入划分：" % len(bad_seg))
        for seg, mode in bad_seg[:10]:
            print("    %s  (mode=%s)" % (seg, mode))
        if len(bad_seg) > 10:
            print("    ... 其余 %d 个略" % (len(bad_seg) - 10))
        print("[Warning] 请先运行  python tools/fix_rgb_masks.py  转换后再重新划分。\n")

    if not total_seg:
        raise ValueError("SegmentationClass 中没有可用的索引图标签。")

    total_seg.sort()          # 固定顺序，保证同一 seed 下划分可复现
    num     = len(total_seg)
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    classes_nums        = np.zeros([256], np.int32)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。"%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            #   这里原本写成 "...没有占位符的字符串" % (name, shape)，
            #   会抛 TypeError —— 恰好在检测到坏标签时崩溃。
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。"%(name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。")
            continue

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("SegmentationClass 中的标签必须是 png 索引图（像素值 = 类别下标）。")
    print("本项目的 JPEGImages 使用 .png 原图（dataloader / get_miou 均按 .png 读取）。")
    print("如果格式有误，参考:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")