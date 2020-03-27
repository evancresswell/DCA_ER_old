# protein-emachine
# github token: 4de5804a0bb1014bfacfc2466187741403f41b6c

#------------------------------------------------------#
#----- Generate Singularity Container to Run Code -----#
### In DCA_ER/er_images/ (location of Dockerfile)
# 	$ sudo docker build -t ubuntu-er .

### In DCA_ER/ (where the .simg files must be stored)
#	$ sudo docker tag #IMAGE ID# #REPO#:#TAG#
#	$ sudo docker push evancresswell/#REPO#:#TAG#

	#--- Singularity: Build .simg file from Docker ---#
	#	$ sudo singularity pull docker://evancresswell/#REPO#:#TAG#
	#	$ sudo singularity build dca_er-DCA.simg #REPO#_#TAG#.sif 
	#-------------------------------------------------#

#------------------------------------------------------#
#------------------------------------------------------#
