# preprocess-dataset

preprocess images and generate train, validation, test dataset
this is my first repository

## 1. [Data.py](https://github.com/silenceagle/preprocess-dataset/blob/master/Data.py)
### *Class* 'Data': 
generate a batch of (images, labels) data.

### *Class* 'Data3': 
generate a batch of (images, label1s, label2s) data.

## 2. [data_prepare.py](https://github.com/silenceagle/preprocess-dataset/blob/master/data_prepare.py)
### *function* norm_with_l2:
normalize datapoint to unit length (L2 norm equals 1)

## 3. [dataset_process.py](https://github.com/silenceagle/preprocess-dataset/blob/master/dataset_process.py)
### *function* data_norm_zscore
### *function* data_norm_minmax

### *function* \_random_choose_fix_proportion_train_test_files
### *function* random_choose_fix_number_image_files
### *function* random_choose_fix_number_image_files_with_category

### *function* split_dataset_to_train_test_with_category

### *function* generate_train_validation_dataset_singlesize_with_category
### *function* generate_dataset_singlesize_with_category
### *function* generate_dataset_singlesize_no_label

### *function* random_perm3

### *function* generate_dataset_multisize_with_cagtegory
### *function* generate_dataset_multisize
### *function* generate_train_validation_dataset_multisize
### *function* generate_train_test_validation_dataset_multisize_with_category

### *function* copy_npy_files_with_category

## [img_process.py](https://github.com/silenceagle/preprocess-dataset/blob/master/img_process.py)

### *function* img_gray_with_category
### *function* img_gray

### *function* get_and_save_amplitude_image_slc_with_category
### *function* get_and_save_amplitude_image_grd_with_category

### *function* gen_npy_file

### *function* \_\_img_crop_from_center
### *function* crop_imgs_and_save_smaller

### *function* \_\_matrix_padding_multi_size_soft
### *function* \_\_matrix_padding_force_to_get_fix_sized_matrix
### *function* \_\_matrix_padding
### *function* \_\_dataset_padding
### *function* padding_images_with_zero
### *function* padding_images_with_zero_with_category
### *function* padding_images_with_zero_to_square_size
### *function* padding_to_fix_sized_and_save_imgs_with_category

### *function* \_\_sample_img_with_slide_windows
### *function* sample_img_with_slide_window_and_save_npy_with_category
### *function* sample_with_slide_window_and_save_npy_stride

### *function* resize_img_and_save_to_folder_opensarship_slc_with_category
### *function* resize_img_and_save_to_folder
### *function* resize_img_and_save_to_folder_with_category

### *function* \_\_rotate_img_90_degree
### *function* rotate_img_90degree_and_save_to_folder
### *function* rotate_img_90degree_and_save_to_folder_with_category
### *function* rotate_img_180degree_and_save_to_folder
### *function* rotate_img_180degree_and_save_to_folder_with_category
### *function* rotate_img_270degree_and_save_to_folder
### *function* rotate_img_270degree_and_save_to_folder_with_category

### *function* \_\_flip_img
### *function* flip_img_and_save_to_folder
### *function* flip_img_and_save_to_folder_with_category
### *function* gen_dataset_img_size_statistic
### *function* change_img_file_extension

## [npz_process.py](https://github.com/silenceagle/preprocess-dataset/blob/master/npz_process.py)
### *function* npyz_print_data_shape

## [xml_process.py](https://github.com/silenceagle/preprocess-dataset/blob/master/xml_process.py)

### *function* pnpoly

### *function* gen_dataset_ssdd

### *function* gen_dataset_hrsc2016





