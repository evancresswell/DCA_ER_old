# protein-emachine
# github token: 4de5804a0bb1014bfacfc2466187741403f41b6c

#------------------------------------------------------#
#----- Generate Singularity Container to Run Code -----#
## In DCA_ER/er_images/ (location of Dockerfile)
### Make sure that you are not ssh or vpn connected
#	$ sudo docker build --no-cache --rm -t pydca-container .

## In DCA_ER/ (where the .simg files must be stored)
#	$ sudo docker tag <IMAGE ID> evancresswell/<REPO>:<TAG>
#		ex: $ sudo docker tag <IMAGE ID> evancresswell/pydca:py37
#	$ sudo docker push evancresswell/<REPO>:<TAG>
#		ex: $ sudo docker push evancresswell/pydca:py37

	#--- Singularity: Build .simg file from Docker ---#
	#	$ sudo singularity pull docker://evancresswell/#REPO#:#TAG#
		#	ex: sudo singularity pull docker://evancresswell/pydca:py37
	#	$ sudo singularity build dca_er-DCA.simg #REPO#_#TAG#.sif 
	#		ex: $ sudo singularity build pydca-py37.simg pydca_py37.sif 
	#-------------------------------------------------#

#------------------------------------------------------#
#------------------------------------------------------#
