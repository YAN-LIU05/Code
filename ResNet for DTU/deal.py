from PIL import Image
import os

# 设置路径
input_folders = ['Nordtank 2017','Nordtank 2018']   # 原图路径
output_folder = 'Nordtank_dealed'  # 切图保存路径
tile_size = 1024

os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有JPG图片
for folder in input_folders:
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            img = Image.open(image_path)
            width, height = img.size

            base_name = os.path.splitext(filename)[0]

            # 计算可切割的最大行列数（不补边）
            max_row = height // tile_size
            max_col = width // tile_size

            for row in range(max_row):
                for col in range(max_col):
                    left = col * tile_size
                    upper = row * tile_size
                    right = left + tile_size
                    lower = upper + tile_size

                    tile = img.crop((left, upper, right, lower))

                    # 列号从1开始
                    tile_name = f"{base_name}_{row}_{col}.JPG"
                    tile_path = os.path.join(output_folder, tile_name)

                    tile.save(tile_path, quality=95)

print("全部图片切割完毕！")
