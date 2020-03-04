# Script used to subtract elements of 2nd list from the first
# Both lists must be passed in through command line
import sys
import numpy as np

#------------------------------------------------------#
def subtract_lists(list1_name, list2_name):
	# Read in both lists
	s1 = np.loadtxt(list1_name,dtype='str')
	s2 = np.loadtxt(list2_name,dtype='str')
	s3 = np.copy(s1)
	
	# Subtract 2nd list from first list
	for pfam_id in s2:
		s3 = s3[s3!=pfam_id]	
	return s3
#------------------------------------------------------#
if __name__ == "__main__":
	list1_name = sys.argv[1]
	list2_name = sys.argv[2]
	out_list = sys.argv[3]

	output_list = subtract_lists(list1_name, list2_name)

	# Write resulting list
	f = open(out_list,'w')
	for pfam_id in output_list:
		f.write('%s\n'%pfam_id)
	f.close()
