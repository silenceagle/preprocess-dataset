"""
    Data.py
    Class Data for do next batch
    20181105
"""
import numpy as np


class Data(object):
    def __init__(self, images, labels):
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._steps_completed = 0
        self._index_in_step = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def steps_completed(self):
        return self._steps_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        "go through all the data"
        start = self._index_in_step
        # 对第一个step进行打乱
        if self._steps_completed == 0 and start == 0 and shuffle:
            # 返回一个array对象且间隔为1
            perm0 = np.arange(self._num_examples)
            # 打乱列表
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # 进入下一个step之前，有余下数据的处理
        if start + batch_size > self._num_examples:
            if start + batch_size < 2 * self._num_examples:
                # 完成一个step的标志位
                self._steps_completed += 1
                # 得到该step余下的数据
                rest_num_examples = self._num_examples - start
                images_rest_part = self._images[start:self._num_examples]
                labels_rest_part = self._labels[start:self._num_examples]
                # 对数据进行打乱
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._images = self._images[perm]
                    self._labels = self._labels[perm]
                # 开始下一个step，并凑齐一个batch
                start = 0
                self._index_in_step = batch_size - rest_num_examples
                end = self._index_in_step
                images_new_part = self._images[start:end]
                labels_new_part = self._labels[start:end]
                return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)
            else:
                reuse_times = np.int(np.floor((start + batch_size) / self._num_examples) - 1)
                self._steps_completed += reuse_times + 1
                images_rest_part = self._images[start:self._num_examples]
                labels_rest_part = self._labels[start:self._num_examples]
                batch_images = images_rest_part
                batch_labels = labels_rest_part
                for ind_resuse in range(reuse_times):
                    if shuffle:
                        perm = np.arange(self._num_examples)
                        np.random.shuffle(perm)
                        self._images = self._images[perm]
                        self._labels = self._labels[perm]
                    batch_images = np.concatenate((batch_images, self._images), axis=0)
                    batch_labels = np.concatenate((batch_labels, self._labels), axis=0)
                if (start + batch_size) % self._num_examples == 0:
                    self._index_in_step = 0
                    return batch_images, batch_labels
                else:
                    if shuffle:
                        perm = np.arange(self._num_examples)
                        np.random.shuffle(perm)
                        self._images = self._images[perm]
                        self._labels = self._labels[perm]
                    self._index_in_step = (start + batch_size) % self._num_examples
                    end = self._index_in_step
                    batch_images = np.concatenate((batch_images, self._images[0:end]), axis=0)
                    batch_labels = np.concatenate((batch_labels, self._labels[0:end]), axis=0)
                    return batch_images, batch_labels
        else:
            self._index_in_step += batch_size
            end = self._index_in_step
            return self._images[start:end], self._labels[start:end]
