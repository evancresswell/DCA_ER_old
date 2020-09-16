from __future__ import absolute_import
from pydca.er_dca.er_dca import ERDCA
from .input_files_path import InputFilesPath
import unittest

class MeanFieldDCATestCase(unittest.TestCase):
    """Test MeanFieldDCA instance behaviour
    """
    def setUp(self):
        #protein test files
        self.__protein_msa_file = InputFilesPath.protein_msa_file
        self.__protein_ref_file = InputFilesPath.protein_ref_file


        self.__erdca_instance_protein = ERDCA(
            self.__protein_msa_file,
            'protein',
        )


    def test_compute_sorted_DI_protein(self):
        """
        """
        sorted_DI = self.__erdca_instance_protein.compute_sorted_DI()

if __name__ == '__main__':
    unittes.main()
