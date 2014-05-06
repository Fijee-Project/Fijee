#!/bin/sh
#
#
#
#
#
#
################################################################################
#                                                                              #
#                     ENVIRONNEMENT VARIABLES                                  #
#                                                                              #
################################################################################
export CUDA_LIB=/usr/local/cuda-5.5/targets/x86_64-linux/lib
export CUDA_OLD_LIB=/usr/local/old/cuda-5.0/lib64
export AMDAPPSDKROOT=/data1/devel/AMD-APP-SDK/AMD-APP-SDK-v2.8.1.0-RC-lnx64/
export EIGEN3=/data1/devel/CPP/Eigen/
export VTK=/data1/devel/CPP/VTK/
export VTK_RELEASE=vtk-5.10
export CGAL=/data1/devel/CPP/CGAL/
export VIENNACL=/data1/devel/CPP/ViennaCL
export NIFTI=/data1/devel/CPP/nifti
# Solver PETSc
export PETSC_DIR=/usr/lib/petscdir/3.1/
export PETSC_ARCH=linux-gnu-c-opt
# Solver trilinos
export TRILINOS=/data1/devel/CPP/Trilinos
# FEniCS
export FENICS=/data1/devel/CPP/FEniCS

###################
# Freesurfer data #
###################
export SUBJECTS_DIR=/data1/data/subjects
export SUBJECT=GazzDCS0004mgh_GPU3
# MRI imaging
T1_DICOM=/data1/data/dicom/tDCS/GazzDCS0004mgh/MEMPRAGE_4e_p2_1mm_isoRMS_12
FIRST_T1_DICOM=IM-0012-0001.dcm
#
T2_DICOM=/data/dicom/tDCS/GazzDCS0004mgh/Hough/T2_SPACE_1mm_iso_15
FIRST_T2_DICOM=IM-0015-0001.dcm
# MRI Diffusion Dat Imaging. 
DTI_DICOM=/data1/data/dicom/tDCS/GazzDCS0004mgh/ep2dadvdiff511E_b2000_64dir_19
DTI_DICOM_FIRST=IM-0019-0001.dcm
DTI_NIFTI=/data1/data/nifti/tDCS/GazzDCS0004mgh/
DTI_NIFTI_FIRST=ep2dadvdiff511Eb200064dirs019a001.nii.gz
# Diffusion gradient table if images are not MGH DICOMs
DTI_NIFTI_BVAL=$DTI_NIFTI/ep2dadvdiff511Eb200064dirs019a001.bval
DTI_NIFTI_BVEC=$DTI_NIFTI/ep2dadvdiff511Eb200064dirs019a001.bvec
bval_tracula=$SUBJECT"_bval_tracula"
bvec_tracula=$SUBJECT"_bvec_tracula"
#
export NUM_PROCS=16 # number of parallel processors for OpenMP options
export FREESURFER_HOME=/usr/local/freesurfer
export FSFAST_HOME=/usr/local/freesurfer/fsfast
export FSF_OUTPUT_FORMAT=nii.gz
export MNI_DIR=/usr/local/freesurfer/mni
export FSL_DIR=/usr/share/fsl/5.0
export FSL_LIB=/usr/lib/fsl/5.0

###################
#     MNE         #
###################
export MNE_ROOT=/data1/devel/CPP/MNE/MNE-2.7

###################
#     SPM         #
###################
export SPM_DIR=/data1/devel/CPP/SPM/spm8

###################
#    FIJEE        #
###################
FIJEE=/data1/devel/CPP/Fijee/Fijee/
FIJEE_FEM=$SUBJECTS_DIR/$SUBJECT/fem
FIJEE_INPUT=$SUBJECTS_DIR/$SUBJECT/fem/input

############################
# LD_LIBRARY_PATH and PATH #
############################
export LD_LIBRARY_PATH=$CUDA_LIB:$CUDA_OLD_LIB:$AMDAPPSDKROOT/lib/x86_64/:$VTK/install/lib/$VTK_RELEASE/:$CGAL/install/lib:$MNE_ROOT/lib:$FSL_LIB:$FENICS/install/lib:$FENICS/install/lib:$LD_LIBRARY_PATH
#
export PATH=$NIFTI/install/bin:$MNE_ROOT/bin:$PATH


################################################################################
#                                                                              #
#        check: function checks the pipe line environment variables            #
#                                                                              #
################################################################################
check()
{
    echo "Checking the environment"
    #
    test ! -d $CUDA_LIB && echo "Directory $CUDA_LIB not found!";
    test ! -d $AMDAPPSDKROOT && echo "Directory $AMDAPPSDKROOT not found!";
    test ! -d $EIGEN3 && echo "Directory $EIGEN3 not found!";
    test ! -d $VTK && echo "Directory $VTK not found!";
    test ! -d $CGAL && echo "Directory $CGAL not found!";
    test ! -d $PETSC_DIR && echo "Directory $PETSC_DIR not found!";
    test ! -d $PETSC_DIR/$PETSC_ARCH && echo "Directory $PETSC_DIR$PETSC_ARCH not found!";
    test ! -d $TRILINOS && echo "Directory $TRILINOS not found!";
    test ! -d $NIFTI && echo "Directory $NIFTI not found!";

    #
    test ! -d $SUBJECTS_DIR && echo "Directory $SUBJECTS_DIR not found!";
    test ! -d $T1_DICOM && echo "Directory $T1_DICOM not found!";
    test ! -s $T1_DICOM/$FIRST_T1_DICOM && echo "File $T1_DICOM/$FIRST_T1_DICOM not found!";
    test ! -d $T2_DICOM && echo "Directory $T2_DICOM not found!";
    test ! -s $T2_DICOM/$FIRST_T2_DICOM && echo "File $T2_DICOM/$FIRST_T2_DICOM not found!";
    test ! -d $DTI_DICOM && echo "Directory $DTI_DICOM not found!";
    test ! -s $DTI_DICOM/$DTI_DICOM_FIRST && echo "File $DTI_DICOM/$DTI_DICOM_FIRST not found!";
    test ! -d $DTI_NIFTI && echo "Directory $DTI_NIFTI not found!";
    test ! -s $DTI_NIFTI$DTI_NIFTI_FIRST && echo "File $DTI_NIFTI/$DTI_NIFTI_FIRST not found!";
    test ! -s $DTI_NIFTI_BVAL && echo "File $DTI_NIFTI_BVAL not found!";
    test ! -s $DTI_NIFTI_BVEC && echo "File $DTI_NIFTI_BVEC not found!";
    test ! -d $FREESURFER_HOME && echo "Directory $FREESURFER_HOME not found!";
    test ! -d $FSFAST_HOME && echo "Directory $FSFAST_HOME not found!";
    test ! -d $MNI_DIR && echo "Directory $MNI_DIR not found!";
    test ! -d $FSL_DIR && echo "Directory $FSL_DIR not found!";
    test ! -d $FSL_LIB && echo "Directory $FSL_LIB not found!";
    test ! -d $MNE_ROOT && echo "Directory $MNE_ROOT not found!";

    # commands
    command -v matlab >/dev/null 2>&1 || { echo >&2 "Matlab required but it's not installed.  Aborting."; exit 1; }
    command -v $FREESURFER_HOME/bin/recon-all >/dev/null 2>&1 || \
	{ echo >&2 "Freesurfer required but it's not installed.  Aborting."; exit 1; }
    command -v $NIFTI/install/bin/nifti1_test >/dev/null 2>&1 || \
	{ echo >&2 "Nifti tools required but not installed.  Aborting."; exit 1; }

}


################################################################################
#                                                                              #
#  FREESURFER EXECUTION - brain segmentation and diffusion data reconstruction #
#                                                                              #
################################################################################
#
# Launch FREESURFER Process
freesurfer()
{
    echo "Run Freesurfer"
    #
    FREESURFER_EXEC_TIME_FILE=FreeSurfer_$SUBJECT.timelog
    /usr/bin/time -v -o $PWD/$FREESURFER_EXEC_TIME_FILE \
	$FREESURFER_HOME/bin/recon-all -subjid $SUBJECT \
	-i $T1_DICOM/$FIRST_T1_DICOM -sd $SUBJECTS_DIR/ -all \
	-t2 $T2_DICOM/$FIRST_T2_DICOM \
	-T2pial \
	-use-cuda \
	1> FreeSurfer_$SUBJECT.log \
	2> FreeSurfer_$SUBJECT.err

# Other usefull options
#    -use-cuda \
#    -openmp $NUM_PROCS \

    #
    # Create study directories
    mkdir -p $FIJEE_FEM
    mkdir -p $FIJEE_INPUT
    mkdir -p $FIJEE_INPUT/{STL,mri}

#
# Produces STL surfaces of the cortex
    surf=$SUBJECTS_DIR/$SUBJECT/surf
    list_surfaces="rh.pial lh.pial rh.smoothwm lh.smoothwm"
    if [ -d $surf ]
    then 
	for surface in $list_surfaces;
	do test -s $surf/$surface && ($FREESURFER_HOME/bin/mris_convert \
	    $surf/$surface $FIJEE_INPUT/STL/$surface.stl) \
	    || echo "File $surface not found!";
	done;
    else
	echo "$surf directory does not exist!"
	exit 1
    fi

#
# converts the file all segmentation mgz into a NIFTI file
    aseg="aseg"
#
    test -s $SUBJECTS_DIR/$SUBJECT/mri/$aseg.mgz && \
	($FREESURFER_HOME/bin/mri_convert \
	$SUBJECTS_DIR/$SUBJECT/mri/$aseg.mgz $FIJEE_INPUT/mri/$aseg.nii) \
	|| echo "File $aseg not found!";
#
    test -s $SUBJECTS_DIR/$SUBJECT/mri/$aseg.mgz && \
	($FREESURFER_HOME/bin/mri_convert \
	$SUBJECTS_DIR/$SUBJECT/mri/$aseg.mgz $FIJEE_INPUT/mri/$aseg.img) \
	|| echo "File $aseg not found!";

#
# converts the files T1 and T2  mgz into a NIFTI file
    T1W="T1"
    T2W="T2"
#
    test -s $SUBJECTS_DIR/$SUBJECT/mri/$T1W.mgz && \
	($FREESURFER_HOME/bin/mri_convert \
	$SUBJECTS_DIR/$SUBJECT/mri/$T1W.mgz $SUBJECTS_DIR/$SUBJECT/mri/$T1W.nii) \
	|| echo "File $T1W not found!";
#
    test -s $SUBJECTS_DIR/$SUBJECT/mri/$T2W.mgz && \
	($FREESURFER_HOME/bin/mri_convert \
	$SUBJECTS_DIR/$SUBJECT/mri/$T2W.mgz $SUBJECTS_DIR/$SUBJECT/mri/$T2W.nii) \
	|| echo "File $T2W not found!";
}

#
# Diffusion data reconstruction
freesurfer_diffusion()
{
    echo "Run Freesurfer for diffusion"
#
# Check we have the gradient diffusion tables
    test -f $DTI_NIFTI_BVAL && \
	bval_bvec=" --b $DTI_NIFTI_BVAL $DTI_NIFTI_BVEC " \
	|| echo "no bval and bvec files provided. Run: dcm2nii -d n -a n -o $DTI_NIFTI $DTI_DICOM/$DTI_DICOM_FIRST"
#
# dt_recon
    FREESURFER_DTI_EXEC_TIME_FILE=FreeSurfer_dti_$SUBJECT.timelog
    /usr/bin/time -v -o $PWD/$FREESURFER_DTI_EXEC_TIME_FILE \
	$FREESURFER_HOME/bin/dt_recon --i $DTI_DICOM/$DTI_DICOM_FIRST --s $SUBJECT \
	--o $SUBJECTS_DIR/$SUBJECT/Diffusion_tensor/ $bval_bvec \
	1> FreeSurfer_dti_$SUBJECT.log \
	2> FreeSurfer_dti_$SUBJECT.err
}

#
# TRACULA
tracula()
{
    echo "Run tracula"
#
# Rotates the diffusion gradient tables
    awk 'NR==1 {print $0}' $DTI_NIFTI_BVAL | sed -e"s/ /\n/g" | awk '{print $0}' > $bval_tracula
#
    awk '{
         if (max_nf < NF) max_nf = NF
     	max_nr = NR
     	for (x = 1; x <= NF; x++) vector[NR,x] = $x
     }
     END {
         for (x = 1; x <= max_nf; x++) {
             for (y = 1; y <= max_nr; y++) printf("%s \t", vector[y, x])
             printf("\n")
         }
     }' $DTI_NIFTI_BVEC > $bvec_tracula

#
# Build tracul's configuration file
    tracula_conf=$SUBJECT"_tracula.conf"
#
    echo "setenv SUBJECTS_DIR $SUBJECTS_DIR" > $tracula_conf
    echo "setenv SUBJECT $SUBJECT" >> $tracula_conf
    echo "set dtroot = $SUBJECTS_DIR/$SUBJECT/Tracula" >> $tracula_conf
    echo "set subjlist = ( $SUBJECT )" >> $tracula_conf
    echo "set runlist = (1)" >> $tracula_conf
    echo "set dcmroot = $DTI_DICOM" >> $tracula_conf
    echo "set dcmlist = ( $DTI_DICOM_FIRST )" >> $tracula_conf
    echo "set bvecfile = $PWD/$bvec_tracula" >> $tracula_conf
    echo "set bvalfile = $PWD/$bval_tracula" >> $tracula_conf
    echo "set dob0 = 0" >> $tracula_conf
#    echo "set b0mlist = ()" >> $tracula_conf
#    echo "set b0plist = ()" >> $tracula_conf
#    echo "set echospacing = 0.7" >> $tracula_conf
    echo "set doeddy = 1" >> $tracula_conf
    echo "set dorotbvecs = 1" >> $tracula_conf
    echo "set thrbet = 0.5" >> $tracula_conf
    echo "set doregflt = 0" >> $tracula_conf
    echo "set doregbbr = 1" >> $tracula_conf
    echo "set doregmni = 1" >> $tracula_conf
    echo "set mnitemp = $FSL_DIR/data/standard/MNI152_T1_1mm_brain.nii.gz" >> $tracula_conf
    echo "set doregcvs = 0" >> $tracula_conf
#    echo "set cvstemp =" >> $tracula_conf
#    echo "set cvstempdir =" >> $tracula_conf
    echo "set usemaskanat = 1" >> $tracula_conf
    echo "set pathlist = ( lh.cst_AS rh.cst_AS lh.unc_AS rh.unc_AS lh.ilf_AS rh.ilf_AS fmajor_PP fminor_PP lh.atr_PP rh.atr_PP lh.ccg_PP rh.ccg_PP lh.cab_PP rh.cab_PP lh.slfp_PP rh.slfp_PP lh.slft_PP rh.slft_PP )" >> $tracula_conf
    echo "set ncpts = (6 6 5 5 5 5 7 5 5 5 5 5 4 4 5 5 5 5)" >> $tracula_conf
    echo "set trainfile = $FREESURFER_HOME/trctrain/trainlist.txt" >> $tracula_conf
    echo "set nstick = 2" >> $tracula_conf
    echo "set nburnin = 200" >> $tracula_conf
    echo "set nsample = 7500" >> $tracula_conf
    echo "set nkeep = 5" >> $tracula_conf
    echo "set reinit = 0" >> $tracula_conf

#
# trac-all -{prep,bedp,path} -c tracula.conf -log tracula.log -noappendlog -sd /data1/data/subjects
    TRACULA_PREP_EXEC_TIME_FILE=tracula_prep_$SUBJECT.timelog
    TRACULA_BEDP_EXEC_TIME_FILE=tracula_bedp_$SUBJECT.timelog
    TRACULA_PATH_EXEC_TIME_FILE=tracula_path_$SUBJECT.timelog
# PREP
    /usr/bin/time -v -o $PWD/$TRACULA_PREP_EXEC_TIME_FILE \
	$FREESURFER_HOME/bin/trac-all -prep -c $tracula_conf -no-isrunning \
	1> tracula_prep_$SUBJECT.log \
	2> tracula_prep_$SUBJECT.err
# BEDP
    /usr/bin/time -v -o $PWD/$TRACULA_BEDP_EXEC_TIME_FILE \
	$FREESURFER_HOME/bin/trac-all -bedp -c $tracula_conf -no-isrunning \
	1> tracula_BEDP_$SUBJECT.log \
	2> tracula_BEDP_$SUBJECT.err
# PATH
    /usr/bin/time -v -o $PWD/$TRACULA_path_EXEC_TIME_FILE \
	$FREESURFER_HOME/bin/trac-all -path -c $tracula_conf -no-isrunning \
	1> tracula_path_$SUBJECT.log \
	2> tracula_path_$SUBJECT.err
}

################################################################################
#                                                                              #
#                MNE EXECUTION - scalp and skull                               #
#                                                                              #
################################################################################
#
# Launch MNE Process
mne()
{
    echo "Run MNE"
    #
    MEN_EXEC_TIME_FILE=mne_$SUBJECT.timelog
    /usr/bin/time -v -o $PWD/$MEN_EXEC_TIME_FILE \
	$MNE_ROOT/bin/mne_watershed_bem -atlas \
	1> MNE_$SUBJECT.log \
	2> MNE_$SUBJECT.err

#
# We produce STL surfaces for the skalp and skull
    mne_surf=$SUBJECTS_DIR/$SUBJECT/bem/watershed
    list_mne_surfaces="_brain_surface _inner_skull_surface _outer_skin_surface _outer_skull_surface"

    if [ -d $mne_surf ]
    then 
	for surface in $list_mne_surfaces;
	do test -s $mne_surf/$SUBJECT$surface && \
	    ($FREESURFER_HOME/bin/mris_convert $mne_surf/$SUBJECT$surface $FIJEE_INPUT/STL/$SUBJECT$surface.stl) \
	    || echo "File $surface not found!";
	done;
    else
	echo "$mne_surf directory does not exist!"
	exit 1
    fi
}
################################################################################
#                                                                              #
#                               SPM EXECUTION                                  #
#                                                                              #
################################################################################
#
#
#
spm()
{
    echo "Run SPM8"
    #
    test -s $SUBJECTS_DIR/$SUBJECT/mri/T1.nii && \
	T1W=$PWD/spm/T1.nii \
	|| echo "File T1 not found!";
    #
    test -s $SUBJECTS_DIR/$SUBJECT/mri/T2.nii && \
	T2W=$PWD/spm/T2.nii \
	|| echo "File T2 not found!";
    #
    SPM_COMMAND="addpath ('$SPM_DIR'); addpath ('$FIJEE/scripts/matlab/'); " 
    SPM_COMMAND=$SPM_COMMAND"nii_newseg('$T1W','false','$FIJEE/scripts/matlab/eTPM.nii','$T2W'); " 
    SPM_COMMAND=$SPM_COMMAND"nii_smooth('c3T1.nii',2.0); " 
    SPM_COMMAND=$SPM_COMMAND"nii_smooth('c4T1.nii',1.5); " 
    SPM_COMMAND=$SPM_COMMAND"nii_smooth('c5T1.nii',3.0); " 
    SPM_COMMAND=$SPM_COMMAND"nii_smooth('c6T1.nii',1.5); exit; " 
    #
    SPM_EXEC_TIME_FILE=$PWD/SPM_$SUBJECT.timelog
    mkdir spm
    ln -s $SUBJECTS_DIR/$SUBJECT/mri/T1.nii $T1W
    ln -s $SUBJECTS_DIR/$SUBJECT/mri/T2.nii $T2W
    cd spm
    #
    /usr/bin/time -v -o $SPM_EXEC_TIME_FILE \
	matlab -nodesktop -nosplash -r "$SPM_COMMAND" \
	1> ../SPM_$SUBJECT.log \
	2> ../SPM_$SUBJECT.err
    #
    for i in {3..6} ; do \
    test -s sc${i}T1.nii && \
	($NIFTI/install/bin/nifti1_test -a2 sc${i}T1 sc${i}T1) \
	|| echo "SPM file num $i not found!"; \
	done
    
    #
    cd $OLDPWD
    mv spm $FIJEE_FEM/input/
}
################################################################################
#                                                                              #
#                               FIJEE EXECUTION                                #
#                                                                              #
################################################################################
#
#
#
#fijee()
#{
#}
################################################################################
#                                                                              #
#                           END OF PROCESSING                                  #
#                                                                              #
################################################################################
end_of_process()
{
    echo ""
    echo "End of processing"
    echo "Freesurfer data should be in: $SUBJECTS_DIR/$SUBJECT"
}

################################################################################
#                                                                              #
#                                                                              #
#                                                                              #
################################################################################
help()
{
    echo "usage: $0 [options]"
    echo 
    echo "     --help                  shows this help"
    echo "     --check                 check the environment variables and files" 
    echo "     --run                   run the fijee's pipelin"
    echo 
}

################################################################################
#                                                                              #
#                          Script options parsing                              #
#                                                                              #
################################################################################
#
# Parse the options
if [ $# -eq 0 ] 
then help
else
    while [ $# -gt 0 ]
    do
	case "$1" in  
	    --run)
		freesurfer
		freesurfer_diffusion
#		tracula
		mne
		spm
		end_of_process
		exit 1
		;;
	    --check)
		check
		exit 1
		;;
	    --help)
		help
		exit 1
		;;
	esac
	shift
    done
fi
