# import os
# import numpy as np
# import tifffile as tiff
# from PIL import Image

# topo_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/dsm_pngs/'
# out_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/dsm_vis/'
# os.makedirs(out_dir, exist_ok=True)

# for fname in os.listdir(topo_dir):
#     if not fname.endswith('2603.tif'):
#         continue

#     path = os.path.join(topo_dir, fname)
#     data = tiff.imread(path)

#     # 检查并确保数据为2个波段
#     if data.ndim == 3:
#         if data.shape[0] == 2:
#             band0, band1 = data[0], data[1]
#         elif data.shape[-1] == 2:
#             band0, band1 = data[..., 0], data[..., 1]
#         else:
#             print(f"⚠️ 非2波段数据跳过：{fname} shape={data.shape}")
#             continue
#     else:
#         print(f"⚠️ 非多波段数据跳过：{fname} shape={data.shape}")
#         continue

#     # 归一化到 0–255（避免溢出）
#     def normalize_and_convert(band):
#         band = band.astype(np.float32)
#         band -= band.min()
#         band /= (band.max() + 1e-8)
#         return (band * 255).astype(np.uint8)

#     band0_png = normalize_and_convert(band0)
#     band1_png = normalize_and_convert(band1)

#     base = os.path.splitext(fname)[0]
#     Image.fromarray(band0_png).save(os.path.join(out_dir, f"{base}_band0.png"))
#     Image.fromarray(band1_png).save(os.path.join(out_dir, f"{base}_band1.png"))
#     print(f"✅ 已保存：{base}_band0.png 和 {base}_band1.png")


# import os
# import numpy as np
# import tifffile as tiff
# from PIL import Image

# # 标签映射表：IGBP → Hunan Land Cover 类别
# igbp2hunan = np.array([255, 0, 1, 2, 1, 3, 4, 6, 6, 5, 6, 7, 255], dtype=np.uint8)

# def convert_label(raw_label):
#     raw_label[raw_label == 255] = 12  # 把无效值设为 indexable 值
#     return igbp2hunan[raw_label]

# # 设置路径
# base_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/test/'
# s2_dir = os.path.join(base_dir, 's2')
# topo_dir = os.path.join(base_dir, 'topo')
# lc_dir = os.path.join(base_dir, 'lc')
# output_dir1 = os.path.join(base_dir, 's2_outputs')
# os.makedirs(output_dir1, exist_ok=True)
# output_dir2 = os.path.join(base_dir, 'topo_outputs')
# os.makedirs(output_dir2, exist_ok=True)
# output_dir3 = os.path.join(base_dir, 'lc_outputs')
# os.makedirs(output_dir3, exist_ok=True)

# # 批量处理
# s2_files = sorted([f for f in os.listdir(s2_dir) if f.endswith('.tif')])

# for fname in s2_files:
#     basename = os.path.splitext(fname)[0]
#     basename = basename.replace('s2_','')
#     s2fname = 's2_' + basename + '.tif'
#     topofname = 'topo_' + basename + '.tif'
#     lcfname = 'lc_' + basename + '.tif'

#     s2_path = os.path.join(s2_dir, s2fname)
#     topo_path = os.path.join(topo_dir, topofname)
#     lc_path = os.path.join(lc_dir, lcfname)

#     if not os.path.exists(topo_path) or not os.path.exists(lc_path):
#         print(f"跳过 {basename}：topo 或 lc 文件不存在")
#         continue

#     try:
#         s2_img = tiff.imread(s2_path)
#         topo_img = tiff.imread(topo_path)
#         lc_img = tiff.imread(lc_path)

#         rgb = s2_img[:, :, [3, 2, 1]].astype(np.float32)
#         rgb = 255 * (rgb / np.max(rgb))
#         rgb_img = Image.fromarray(rgb.astype(np.uint8))
#         rgb_img.save(os.path.join(output_dir1, f'{basename}_s2_rgb.png'))

#         topo_norm = 255 * (topo_img - np.min(topo_img)) / (np.max(topo_img) - np.min(topo_img))
#         topo_uint8 = topo_norm.astype(np.uint8)
#         Image.fromarray(topo_uint8).save(os.path.join(output_dir2, f'{basename}_dsm.png'))

#         # 标签映射
#         mapped_label = convert_label(lc_img)
#         Image.fromarray(mapped_label).save(os.path.join(output_dir3, f'{basename}_label.png'))

#         print(f"✅ 已处理: {basename}")

#     except Exception as e:
#         print(f"❌ 处理 {basename} 时出错：{e}")


# import os

# s2_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/train/lc'
# s2_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/train/lc'

# for fname in os.listdir(s2_dir):
#     if fname.startswith('lc_') and fname.endswith('.tif'):
#         new_name = fname.replace('lc_', '', 1)
#         src = os.path.join(s2_dir, fname)
#         dst = os.path.join(s2_dir, new_name)
#         os.rename(src, dst)
#         print(f"重命名：{fname} → {new_name}")
        
# s2_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/val/lc'
# s2_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/val/lc'

# for fname in os.listdir(s2_dir):
#     if fname.startswith('lc_') and fname.endswith('.tif'):
#         new_name = fname.replace('lc_', '', 1)
#         src = os.path.join(s2_dir, fname)
#         dst = os.path.join(s2_dir, new_name)
#         os.rename(src, dst)
#         print(f"重命名：{fname} → {new_name}")

# s2_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/test/lc'
# s2_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/test/lc'

# for fname in os.listdir(s2_dir):
#     if fname.startswith('lc_') and fname.endswith('.tif'):
#         new_name = fname.replace('lc_', '', 1)
#         src = os.path.join(s2_dir, fname)
#         dst = os.path.join(s2_dir, new_name)
#         os.rename(src, dst)
#         print(f"重命名：{fname} → {new_name}")

# import os
# import re

# # 设置目标目录路径
# target_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/test/s2/'

# # 匹配纯数字文件名（形如 1212.tif）
# pattern = re.compile(r'^(\d+)\.tif$')

# # 存储数字结果
# numbers = []

# for fname in os.listdir(target_dir):
#     match = pattern.match(fname)
#     if match:
#         numbers.append(int(match.group(1)))

# print(len(numbers))
# print("提取到的数字编号：", numbers)


# import os
# import numpy as np
# import tifffile as tiff

# lc_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/train/lc'
# expected_labels = set(range(0, 7))  # 或自定义：{0,1,2,3,4,5,6,7}

# out_of_range_files = []

# for fname in os.listdir(lc_dir):
#     if not fname.endswith('.tif'):
#         continue

#     path = os.path.join(lc_dir, fname)
#     try:
#         label = tiff.imread(path)
#         unique_vals = np.unique(label)

#         unexpected_vals = set(unique_vals) - expected_labels
#         if unexpected_vals:
#             print(f"⚠️ 发现异常类别 {unexpected_vals} 于文件 {fname}")
#             out_of_range_files.append((fname, list(unexpected_vals)))

#     except Exception as e:
#         print(f"❌ 无法读取 {fname}: {e}")

# if not out_of_range_files:
#     print("✅ 所有标签文件类别值都在预期范围内。")
# else:
#     print(f"共 {len(out_of_range_files)} 个文件包含异常类别。")


# import os
# import numpy as np
# import tifffile as tiff
# from collections import Counter

# lc_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/train/lc'

# label_counter = Counter()
# total_pixels = 0
# file_count = 0

# for fname in os.listdir(lc_dir):
#     if not fname.endswith('.tif'):
#         continue
#     try:
#         path = os.path.join(lc_dir, fname)
#         label = tiff.imread(path)
#         flat = label.flatten()
#         label_counter.update(flat)
#         total_pixels += flat.size
#         file_count += 1
#     except Exception as e:
#         print(f"❌ 读取失败: {fname} - {e}")

# # 统计结果输出
# print(f"\n共统计 {file_count} 个标签文件，像素总数为 {total_pixels}\n")
# print("类别值\t像素数量\t占比")
# print("-" * 30)

# for label_val, count in sorted(label_counter.items()):
#     freq = count / total_pixels
#     print(f"{label_val}\t{count:,}\t{freq:.4%}")



# import os
# import numpy as np
# import tifffile as tiff

# lc_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/test/lc'

# # 原始 → 目标标签映射
# remap_dict = {
#     0: -1,
#     1: 0,
#     2: 1,
#     3: 2,
#     5: 3,
#     6: 4,
#     8: 5,
#     9: 6
# }

# for fname in os.listdir(lc_dir):
#     if not fname.endswith('.tif'):
#         continue

#     path = os.path.join(lc_dir, fname)
#     try:
#         label = tiff.imread(path)
#         remapped = np.full_like(label, fill_value=-1)  # 默认填充为 -1

#         for orig, new in remap_dict.items():
#             remapped[label == orig] = new

#         # 覆盖保存或另存为新文件
#         tiff.imwrite(path, remapped.astype(np.int8))
#         print(f"✅ 映射完成：{fname}")

#     except Exception as e:
#         print(f"❌ 错误处理 {fname}: {e}")



import os
import numpy as np
import tifffile as tiff
from PIL import Image

lc_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/masks_png/'
out_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/masks_png_vis/'
os.makedirs(out_dir, exist_ok=True)

label_colors = {
    -1: (0, 0, 0),           # ignore
     0: (196, 90, 17),       # cropland
     1: (51, 129, 88),       # forest
     2: (177, 205, 61),      # grassland
     3: (228, 84, 96),       # wetland
     4: (91, 154, 214),      # water
     5: (225, 174, 110),     # unused land
     6: (239, 159, 2),       # built-up area
}

for fname in os.listdir(lc_dir):
    if not fname.endswith('.tif'):
        continue

    path = os.path.join(lc_dir, fname)
    label = tiff.imread(path)

    # 初始化 RGB 图像
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for label_val, color in label_colors.items():
        mask = label == label_val
        rgb[mask] = color

    # 保存为 PNG
    out_path = os.path.join(out_dir, fname.replace('.tif', '.png'))
    Image.fromarray(rgb).save(out_path)
    print(f"✅ 已保存可视图：{out_path}")


# import os
# import numpy as np
# import tifffile as tiff
# from PIL import Image

# input_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/dsm_pngs'
# output_dir = os.path.join(input_dir, 'band0_png')
# os.makedirs(output_dir, exist_ok=True)

# def normalize_and_convert(band):
#     band = band.astype(np.float32)
#     band -= band.min()
#     if band.max() != 0:
#         band /= band.max()
#     return (band * 255).astype(np.uint8)

# for fname in os.listdir(input_dir):
#     if not fname.endswith('.tif'):
#         continue

#     path = os.path.join(input_dir, fname)
#     data = tiff.imread(path)
#     band0 = data[..., 0]

#     band0_png = normalize_and_convert(band0)
#     out_path = os.path.join(output_dir, fname.replace('.tif', '.png'))
#     Image.fromarray(band0_png).save(out_path)
#     print(f"✅ 已保存：{out_path}")


# import os
# import numpy as np
# import tifffile as tiff
# from PIL import Image

# input_dir = '/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/images_png'
# output_dir = os.path.join(input_dir, 'rgb_png')
# os.makedirs(output_dir, exist_ok=True)

# def normalize_band(band):
#     band = band.astype(np.float32)
#     band -= band.min()
#     if band.max() != 0:
#         band /= band.max()
#     return (band * 255).astype(np.uint8)

# for fname in os.listdir(input_dir):
#     if not fname.endswith('.tif'):
#         continue

#     path = os.path.join(input_dir, fname)
#     data = tiff.imread(path)

#     r, g, b = data[..., 3], data[..., 2], data[..., 1]
#     rgb = np.stack([
#         normalize_band(r),
#         normalize_band(g),
#         normalize_band(b)
#     ], axis=-1)

#     out_path = os.path.join(output_dir, fname.replace('.tif', '.png'))
#     Image.fromarray(rgb).save(out_path)
#     print(f"✅ 已保存：{out_path}")


