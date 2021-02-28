import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import scipy.io as scio
from skimage import data

def check_folder_exist(folder_paths: str) -> None:
    """check whether a folder is exist and create one if not

    Args:
        folder_paths (str): a string path
    """    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
                os.makedirs(folder_path)

def preprocess(images_array: np.array, subset: str = 'train') -> None:  
    """preprocess the subset dataset for the MS dataset

    Args:
        images_array (np.array): arraa of the images and annotaions
        subset (str, optional): the subset name. Defaults to 'train'.
    """
    print(f"Preprocess {subset}")
    for i in tqdm(range(images_array.shape[1])):
        name = images_array[0,i]['name'][0]
        width = images_array[0,i]['width'][0][0]
        height = images_array[0,i]['height'][0][0]
        folder_name = name[:4]   

        face_folder = save_dir + os.sep + 'Image_' + folder_name + os.sep + 'Face'
        face_cont_folder = save_dir + os.sep + 'Image_' + folder_name + os.sep + 'FaceCont'
        coor_folder = save_dir + os.sep + 'Image_' + folder_name + os.sep + 'Coordinate'
        image_folder = save_dir + os.sep + 'Image_' + folder_name + os.sep + 'Image'
        check_folder_exist([face_folder, face_cont_folder, coor_folder, image_folder])

        face_num = images_array[0,i]['Face'].shape[1]
        img = Image.open(image_path + os.sep + f'{subset}' + os.sep + name).convert('RGB')
        res_img = img.resize([SIZE_WIDTH,SIZE_HEIGHT])
        for j in range(face_num):
            images_array[0,i]['Face'][0,j]['label'].resize(1,)
            label = int(images_array[0,i]['Face'][0,j]['label'][0])
            label = 1 if label>1 else label

            Rect = images_array[0,i]['Face'][0,j]['rect']
            Rect.resize(4,)

            x, y, w, h = int(Rect[0]), int(Rect[1]), int(Rect[2]), int(Rect[3])
            x_min = max(1,int(x-w/2))
            x_max = min(width,int(x+w/2))
            y_min = max(1, int(y-h/2))
            y_max = min(height, int(y+h/2))

            c_x_min = int(max(1,int(x-3*w)))
            c_x_max = int(min(width, int(x+3*w)))
            c_y_min = max(1,y-2*h)
            c_y_max = min(height, y+6*h)

            tmp_face = img.crop([x_min,y_min,x_max,y_max]).resize([SIZE_WIDTH,SIZE_HEIGHT])
            no_face = '0' + str(j)
            FaceName = face_folder + os.sep + 'Image_' + folder_name + '_Face_' + no_face[len(no_face)-2:] + '_Label_' + str(int(label)) + '.jpg'
            tmp_face.save(FaceName)
            tmp_cont_face = img.crop([c_x_min,c_y_min,c_x_max,c_y_max]).resize([SIZE_WIDTH,SIZE_HEIGHT])
            face_cont_name = face_cont_folder + os.sep + 'Image_' + folder_name + '_Face_' + no_face[len(no_face) - 2:] + '_Label_' + str(int(label)) + '.jpg'
            tmp_cont_face.save(face_cont_name)
            canvas = np.zeros((height, width), dtype=np.uint8)
            canvas[y_min:y_max, x_min:x_max] = 255
            tmp_coor = Image.fromarray(np.uint8(canvas))
            tmp_coor = tmp_coor.resize([SIZE_WIDTH,SIZE_HEIGHT])
            coor_name = coor_folder + os.sep + 'Image_' + folder_name + '_Coor_' + no_face[len(no_face) - 2:] + '_Label_' + str(int(label)) + '.jpg'
            tmp_coor.save(coor_name)
            img_name = image_folder + os.sep + 'Image_' + folder_name + '_Img_' + no_face[len(no_face) - 2:] + '_Label_' + str(int(label)) + '.jpg'
            res_img.save(img_name)

if __name__ == '__main__':

    # Set image width and height
    SIZE_WIDTH = 224
    SIZE_HEIGHT = 224
    # Set file locations
    save_dir = r'\\MS DataSet\\MS DataSet\\MSDataSet_process'
    mat_file = r'\\MS DataSet\\MS DataSet\\data\\annotations'
    image_path = r'\\MS DataSet\\MS DataSet\\images'
    # Load matrix
    data = scio.loadmat(mat_file)
    # create chunks
    train_mat = data['train']
    val_mat = data['val']
    test_mat = data['test']

    preprocess(train_mat, 'train')
    preprocess(val_mat, 'val')
    preprocess(test_mat, 'test')