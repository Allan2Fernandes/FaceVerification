import ImagesFolder
import os
import random
import Image_handler
import numpy as np
from matplotlib import pyplot as plt

class BuildInputLists:
    def __init__(self, path, input_shape):
        self.base_path = path
        self.list_of_Image_folders = []
        self.global_positive_list = []
        self.global_negative_list = []
        self.input_shape = input_shape
        pass

    def build_list_Image_folders(self):
        list_of_folder_names = os.listdir(self.base_path)
        for index, image_folder_path in enumerate(list_of_folder_names):
            image_path = self.base_path + "/" + image_folder_path
            self.list_of_Image_folders.append(ImagesFolder.ImagesFolder(image_path))
            self.list_of_Image_folders[index].build_list_of_image_paths()
            pass
        self.max_num_of_unique_people = len(self.list_of_Image_folders)
        pass

    def build_positive_list(self):
        list_of_list_of_positive_pairs_paths = []
        for single_image_folder in self.list_of_Image_folders:
            single_image_folder.generate_list_of_positive_pairs()
            single_image_folder.generate_list_of_positive_pairs_paths()
            list_of_list_of_positive_pairs_paths.append(single_image_folder.get_list_of_positive_pairs_paths())
            pass

        # print(list_of_list_of_positive_pairs_paths[100][94]) #This is an example of a pair
        for list_of_positive_pairs in list_of_list_of_positive_pairs_paths:
            for pair in list_of_positive_pairs:
                self.global_positive_list.append(pair)
                pass
            pass
        #print(self.global_positive_list)
        pass


    def get_global_positive_list(self):
        return self.global_positive_list
        pass


    def get_negative_pair(self):
        random_set= random.sample(range(self.max_num_of_unique_people), 2)
        image_folder_index1 = random_set[0]
        image_folder_index2 = random_set[1]

        image_folder1 = self.list_of_Image_folders[image_folder_index1]
        image_folder2 = self.list_of_Image_folders[image_folder_index2]

        max_num_images_in_folder1 = image_folder1.get_max_number_of_images()
        max_num_images_in_folder2 = image_folder2.get_max_number_of_images()

        image_index_in_folder1 = random.randint(0, max_num_images_in_folder1-1)
        image_index_in_folder2 = random.randint(0, max_num_images_in_folder2-1)

        path1 = image_folder1.get_image_path_list()[image_index_in_folder1]
        path2 = image_folder2.get_image_path_list()[image_index_in_folder2]

        negative_pair = [path1, path2]
        #print(negative_pair)
        return negative_pair

    def build_negative_list(self, size_of_list):
        for i in range(size_of_list):
            self.global_negative_list.append(self.get_negative_pair())
            pass
        #print(self.global_negative_list)
        pass

    def get_global_negative_list(self):
        return self.global_negative_list

    def convert_path_list_to_image_list(self, train_size, validation_size):
        left_positive_list = self.get_left_positive_list()
        right_positive_list = self.get_right_positive_list()

        left_negative_list = self.get_left_negative_list()
        right_negative_list = self.get_right_negative_list()

        train_left_positive_array = []
        train_right_positive_array = []
        train_left_negative_array = []
        train_right_negative_array = []

        validation_left_positive_array = []
        validation_right_positive_array = []
        validation_left_negative_array = []
        validation_right_negative_array = []

        print("Building positive array")

        positive_list_indices = random.sample(range(0, len(self.global_positive_list)), train_size + validation_size)

        for index in positive_list_indices:
            if len(train_left_positive_array) < train_size:
                #Train Left positive paths
                image_handler = Image_handler.ImageHandler(left_positive_list[index], self.input_shape)
                image_handler.set_image_array()
                image_array = image_handler.get_image_array()
                train_left_positive_array.append(image_array)

                #Train Right positive paths
                image_handler = Image_handler.ImageHandler(right_positive_list[index], self.input_shape)
                image_handler.set_image_array()
                image_array = image_handler.get_image_array()
                train_right_positive_array.append(image_array)
            else:
                # Validation Left positive paths
                image_handler = Image_handler.ImageHandler(left_positive_list[index], self.input_shape)
                image_handler.set_image_array()
                image_array = image_handler.get_image_array()
                validation_left_positive_array.append(image_array)

                # Validation Right positive paths
                image_handler = Image_handler.ImageHandler(right_positive_list[index], self.input_shape)
                image_handler.set_image_array()
                image_array = image_handler.get_image_array()
                validation_right_positive_array.append(image_array)
                pass


        train_left_positive_array = np.array(train_left_positive_array)
        train_right_positive_array = np.array(train_right_positive_array)
        validation_left_positive_array = np.array(validation_left_positive_array)
        validation_right_positive_array = np.array(validation_right_positive_array)
        #print(train_left_positive_array.shape)
        #print(train_right_positive_array.shape)

        """
        for path in left_positive_list:
            image_handler = Image_handler.ImageHandler(path)
            image_handler.set_image_array()
            image_array = image_handler.get_image_array()
            train_left_positive_array.append(image_array)
            if len(train_left_positive_array) >= limit:
                break
            pass
        train_left_positive_array =  np.array(train_left_positive_array)

        print("Building right positive array")
        for path in right_positive_list:
            image_handler = Image_handler.ImageHandler(path)
            image_handler.set_image_array()
            image_array = image_handler.get_image_array()
            train_right_positive_array.append(image_array)
            if len(train_right_positive_array) >= limit:
                break
            pass
        train_right_positive_array = np.array(train_right_positive_array)
        """

        print("Building left negative array")
        for path in left_negative_list:
            if len(train_left_negative_array) < train_size:
                image_handler = Image_handler.ImageHandler(path, self.input_shape)
                image_handler.set_image_array()
                image_array = image_handler.get_image_array()
                train_left_negative_array.append(image_array)
            else:
                image_handler = Image_handler.ImageHandler(path, self.input_shape)
                image_handler.set_image_array()
                image_array = image_handler.get_image_array()
                validation_left_negative_array.append(image_array)
                if len(validation_left_negative_array) >= validation_size:
                    break
            pass
        train_left_negative_array = np.array(train_left_negative_array)
        validation_left_negative_array = np.array(validation_left_negative_array)
        print("Building right negative array")
        for path in right_negative_list:
            if len(train_right_negative_array) < train_size:
                image_handler = Image_handler.ImageHandler(path, self.input_shape)
                image_handler.set_image_array()
                image_array = image_handler.get_image_array()
                train_right_negative_array.append(image_array)
            else:
                image_handler = Image_handler.ImageHandler(path, self.input_shape)
                image_handler.set_image_array()
                image_array = image_handler.get_image_array()
                validation_right_negative_array.append(image_array)
                if len(validation_right_negative_array) >= validation_size:
                    break
            pass
        train_right_negative_array = np.array(train_right_negative_array)
        validation_right_negative_array = np.array(validation_right_negative_array)
        #self.display_image_from_array(train_left_negative_array[900, :, :, :])
        #self.display_image_from_array(train_right_negative_array[900, :, :, :])
        return train_left_positive_array, train_right_positive_array, train_left_negative_array, train_right_negative_array, validation_left_positive_array, validation_right_positive_array, validation_left_negative_array, validation_right_negative_array

    def build_input_and_label_matrices(self, train_size, validation_size):
        train_left_positive_array, train_right_positive_array, train_left_negative_array, train_right_negative_array, validation_left_positive_array, validation_right_positive_array, validation_left_negative_array, validation_right_negative_array = self.convert_path_list_to_image_list(train_size, validation_size)

        train_merged_left_list = []
        train_merged_right_list = []
        train_label_list = []

        validation_merged_left_list = []
        validation_merged_right_list = []
        validation_label_list = []


        for image_array in train_left_positive_array:
            train_merged_left_list.append(image_array)
            train_label_list.append(1)
            pass
        for image_array in train_left_negative_array:
            train_merged_left_list.append(image_array)
            train_label_list.append(0)
            pass
        train_merged_left_array = np.stack(train_merged_left_list, axis=0)
        train_label_array = np.array(train_label_list)

        for image_array in validation_left_positive_array:
            validation_merged_left_list.append(image_array)
            validation_label_list.append(1)
            pass

        for image_array in validation_right_positive_array:
            validation_merged_left_list.append(image_array)
            validation_label_list.append(0)
            pass

        validation_merged_left_array = np.stack(validation_merged_left_list, axis = 0)
        validation_label_array = np.array(validation_label_list)

        for image_array in train_right_positive_array:
            train_merged_right_list.append(image_array)
            pass
        for image_array in train_right_negative_array:
            train_merged_right_list.append(image_array)
            pass
        train_merged_right_array = np.stack(train_merged_right_list, axis=0)

        for image_array in validation_right_positive_array:
            validation_merged_right_list.append(image_array)
            pass
        for image_array in validation_right_negative_array:
            validation_merged_right_list.append(image_array)
            pass
        validation_merged_right_array = np.stack(validation_merged_right_list, axis = 0)
        return train_merged_left_array, train_merged_right_array, train_label_array, validation_merged_left_array, validation_merged_right_array, validation_label_array


    def get_left_positive_list(self):
        return self.extract_column(self.global_positive_list, 0)

    def get_right_positive_list(self):
        return self.extract_column(self.global_positive_list, 1)

    def get_left_negative_list(self):
        return self.extract_column(self.global_negative_list, 0)

    def get_right_negative_list(self):
        return self.extract_column(self.global_negative_list, 1)

    def extract_column(self, lst, column_number):
        return [item[column_number] for item in lst]

    def display_image_from_array(self, array):
        plt.imshow(array, interpolation='nearest')
        plt.show()




