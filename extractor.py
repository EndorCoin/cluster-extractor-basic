import os
import re
import struct
import tempfile
from cStringIO import StringIO

import numpy as np
import pandas as pd
import scipy.io as scio
from scipy.sparse import csr_matrix


class ClustersExtractor(object):
    MAT_INIT_NAME = 'blk_data'
    MAT_FIELDS = {
        'src': 2,
        'type': 1,
        'blk_type': 1,
        'field': 1,
        'fieldby': 1,
        'N': 1,
        'WN': 1,
        'deg': 1,
        'thrs': 2,
        'SUB_CLUSTERS_FILE': 1,
        'percInternal': 1
    }

    def __init__(self, cluster_files_path):
        self.__path = cluster_files_path

    def get_path(self):
        return self.__path

    def retrieve_clusters(self):
        pop_to_clusters_map = self.__build_pop_to_clusters_map()
        clusters_props = self.__build_mat_props_df()
        translation_pop = self.__get_translation_pop()
        return pop_to_clusters_map, clusters_props, translation_pop
    
    def __build_pop_to_clusters_map(self):
        dimensions_for_sparse_mat = self.__get_dimensions_for_sparse_matrix()
        indices = self.__get_ir_list()
        indptr = self.__get_jc_list()
        pop_cluster_map = self.__build_pop_clust_matrix(dimensions_for_sparse_mat, indices, indptr)
        return pop_cluster_map

    def __get_dimensions_for_sparse_matrix(self):
        dimensions_file_name = "MtFile.spmat"
        all_files = self.__get_files_by_name_in_path(dimensions_file_name)
        full_path = all_files[0]
        a = self.__open(full_path)
        f = StringIO(a.read())
        mat_sizes = struct.unpack('<2IQ', f.read(16))
        return {'n_rows': mat_sizes[0], 'n_cols': mat_sizes[1], 'nnz': mat_sizes[2]}
    
    def __get_ir_list(self):
        ir_file_name = "IrFile.spmat"
        all_files = self.__get_files_by_name_in_path(ir_file_name)

        full_path = all_files[0]
        remote_file = self.__open(full_path)
        data = remote_file.read(10 * 1024 * 1024)
        local_temp_path = os.path.join(tempfile.mkdtemp(), ir_file_name)
        with open(local_temp_path, 'w') as f:
            while data != '':
                f.write(data)
                data = remote_file.read(10 * 1024 * 1024)

        ir = np.fromfile(local_temp_path, dtype=np.int32)
        os.unlink(local_temp_path)
        return ir
    
    def __get_jc_list(self):
        jc_file_name = "JcFile.spmat"

        all_files = self.__get_files_by_name_in_path(jc_file_name)

        full_path = all_files[0]
        data = self.__open(full_path)
        local_temp_path = os.path.join(tempfile.mkdtemp(), jc_file_name)
        with open(local_temp_path, 'w') as f:
            f.write(data.read())

        jc = np.fromfile(local_temp_path, dtype=np.int64)
        os.unlink(local_temp_path)
        return jc

    @staticmethod
    def __build_pop_clust_matrix(dimensions_for_sparse_mat, indices, indptr):
        nrows = dimensions_for_sparse_mat['n_rows']
        ncols = dimensions_for_sparse_mat['n_cols']
        nnz = dimensions_for_sparse_mat['nnz']

        data = np.ones(nnz)
        try:
            mat = csr_matrix((data, indices, indptr), shape=(ncols, nrows))
        except Exception as e:
            msg = "Couldn't build population to cluster match due to: %s, aborting." % str(e)

            raise ValueError(msg)
        return mat
    
    def __build_mat_props_df(self):
        mat_names_list = self.__get_mat_files_names()

        if len(mat_names_list) > 1:
            prop_names = 'temp_'
            mat_files = [scio.loadmat(os.path.expanduser(mat_name)) for mat_name in mat_names_list]
            props_df = self.__build_multiple_mat_clusters_properties(mat_files, prop_names)
        else:
            prop_names = 'blk_data'
            mat_file = scio.loadmat(os.path.expanduser(mat_names_list[0]))
            props_df = self.__build_single_mat_clusters_properties(mat_file, prop_names)
        return props_df
    
    def __build_multiple_mat_clusters_properties(self, mat_files, prop_names):
        
        all_props_df = pd.DataFrame()
        for mat_file in mat_files:
            df = self.__build_single_mat_clusters_properties(mat_file, prop_names)
            all_props_df = all_props_df.append(df)
        all_props_df.index = range(len(all_props_df.index))
        return all_props_df
    
    def __build_single_mat_clusters_properties(self, mat_file, prop_names):
        try:
            clusters_props = mat_file[prop_names]
        except Exception:
            msg = "Field %s doesn't exist in mat file, but expected. Cannot continue" % prop_names
            raise ValueError(msg)

        cluster_prop_dict = {}
        for prop_name, counts in self.MAT_FIELDS.iteritems():
            try:
                mat_values = clusters_props[prop_name][0][0]
            except (KeyError, IndexError):
                ValueError("Field {} doesn't exist in mat file, please remove it from config file.".format(prop_name))
                break
            if counts == 1:
                cluster_prop_dict[prop_name] = mat_values.flatten()
            else:
                for i in np.arange(counts):
                    cluster_prop_dict[prop_name + '_' + str(i)] = mat_values[:, i]
        cluster_prop_df = pd.DataFrame(cluster_prop_dict)
        cluster_prop_df.index.names = ['cluster']
        cluster_prop_df['W_prcntg'] = cluster_prop_df['WN'].astype(float) / cluster_prop_df['N'].astype(float)
        del mat_file
        return cluster_prop_df
    
    def __get_translation_pop(self):
        spmat_help_name = ".spmathlp"
        all_files = self.__get_files_by_name_in_path(spmat_help_name)

        full_path = all_files[0]

        f_spmathlp_data = self.__open(full_path).read()
        local_temp_path = os.path.join(tempfile.mkdtemp(), spmat_help_name)
        with open(local_temp_path, 'w') as f:
            f.write(f_spmathlp_data)
        with open(local_temp_path, 'rb') as f_spmathlp:
            num = struct.unpack('<Q', f_spmathlp.read(8))
            # noinspection PyTypeChecker
            ids = np.fromfile(f_spmathlp, dtype=np.double)

            if num[0] != len(ids):
                msg = "translating ids went wrong. Found %d ids, where expected %d ids, aborting" % (len(ids), num[0])
                raise ValueError(msg)

        os.unlink(local_temp_path)

        return ids
        
    def __get_files_by_name_in_path(self, name):
        all_files_in_dir = list(self.__list_dir())

        relevant_files = [file_name for file_name in all_files_in_dir if name in file_name]
        return relevant_files

    @staticmethod
    def __open(path):
        real_path = os.path.expanduser(path)
        if not os.path.isfile(real_path):
            raise LookupError('"{}"  does not exist'.format(real_path))

        return open(real_path, 'rb')
    
    def __get_mat_files_names(self):
        mat_names_list = self.__get_files_by_name_in_path(self.MAT_INIT_NAME)
        mat_names_list.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        return mat_names_list
    
    def __list_dir(self):
        return [os.path.join(self.__path, f) for f in os.listdir(os.path.expanduser(self.__path))]
