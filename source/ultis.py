# Create label file for dataset Corel-10k

import csv
filepath = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\Corel-10k-labels.csv"
with open(filepath, 'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'labels'])
    label = 0
    for i in range(0,10000,1):
        if (i % 100 == 0): label = label + 1
        writer.writerow([i, label])
# Convert image to RGB
def to_rgb(image):
    if image.mode == 'RGB':
        return image
    else:
        return image.convert('RGB')

# Calculate Mean and Standard Deviation of dataset:
# def calc_mean_dataset(filepath):
#     r_channel_sum = 0
#     g_channel_sum = 0
#     b_channel_sum = 0
#     count = 0
#     filepath = 'flickr30k-images'
#     for filename in os.listdir(filepath):
#         if filename[-3:] == 'jpg':
#             img = np.array(Image.open(os.path.join(filepath, filename)).convert('RGB'))
#             r_channel_sum += np.sum(img[:,:,0])
#             g_channel_sum += np.sum(img[:,:,1])
#             b_channel_sum += np.sum(img[:,:,2])
#             count += img.shape[0] * img.shape[1]
#     print(r_channel_sum/count)
#     print(g_channel_sum/count)
#     print(b_channel_sum/count)
# Mean : (113.2971859326401, 107.42922106881713, 98.14465223794616) -> (0.444, 0.421, 0.385)

# def calc_std_deviation(filepath):
#     r_channel_sum = 0
#     g_channel_sum = 0
#     b_channel_sum = 0
#     count = 0
#     filepath = 'flickr30k-images'
#     for filename in os.listdir(filepath):
#         if filename[-3:] == 'jpg':
#             img = np.array(Image.open(os.path.join(filepath, filename)).convert('RGB'))
#             r_channel_sum += np.sum(np.square(img[:,:,0] - 113.2971859326401))
#             g_channel_sum += np.sum(np.square(img[:,:,1] - 107.42922106881713))
#             b_channel_sum += np.sum(np.square(img[:,:,2] - 98.14465223794616))
#             count += img.shape[0] * img.shape[1]
#     print(np.sqrt(r_channel_sum/count))
#     print(np.sqrt(g_channel_sum/count))
#     print(np.sqrt(b_channel_sum/count))  
# Standard Deviation : (72.70319478374329, 70.71527787982022, 72.88658377627682) -> (0.285, 0.277, 0.286)