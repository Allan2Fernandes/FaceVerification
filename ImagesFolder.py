import os
from itertools import combinations


class ImagesFolder:
    def __init__(self, path):
        self.folder_path = path
        self.list_of_image_paths = []
        self.num_of_images_in_folder = 0
        self.list_of_indices = []
        self.list_of_positive_pair_paths = []
        pass

    def build_list_of_image_paths(self): #Using the base folder, build alist of all the image paths inside that folder
        self.list_of_image_names = os.listdir(self.folder_path)
        counter = 0
        for name in self.list_of_image_names:
            self.list_of_image_paths.append(self.folder_path + "/" + name)
            self.list_of_indices.append(counter)
            counter +=1
            pass
        self.num_of_images_in_folder = counter
        pass

    def get_max_number_of_images(self):
        return self.num_of_images_in_folder

    def get_image_path_list(self):
        return self.list_of_image_paths

    def generate_list_of_positive_pairs(self):
        self.list_of_positive_pairs_indices = list(combinations(self.list_of_indices, 2))
        pass

    def generate_list_of_positive_pairs_paths(self):
        for index_pair in self.list_of_positive_pairs_indices:
            path1 = self.list_of_image_paths[index_pair[0]]
            path2 = self.list_of_image_paths[index_pair[1]]
            pair = [path1 , path2]
            self.list_of_positive_pair_paths.append(pair)
            pass
        pass

    def get_list_of_positive_pairs_paths(self):
        return self.list_of_positive_pair_paths











