with open('./SYNTHIA/train.txt', 'w') as f:
    for i in range(9400):
        print(f'RGB/000{i:04}.png    GT/LABELS/000{i:04}.png    superpixel/000{i:04}.png', file=f)