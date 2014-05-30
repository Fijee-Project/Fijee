//  Copyright (c) 2014, Yann Cobigo 
//  All rights reserved.     
//   
//  Redistribution and use in source and binary forms, with or without       
//  modification, are permitted provided that the following conditions are met:   
//   
//  1. Redistributions of source code must retain the above copyright notice, this   
//     list of conditions and the following disclaimer.    
//  2. Redistributions in binary form must reproduce the above copyright notice,   
//     this list of conditions and the following disclaimer in the documentation   
//     and/or other materials provided with the distribution.   
//   
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED   
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE   
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR   
//  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;   
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND   
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS   
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   
//     
//  The views and conclusions contained in the software and documentation are those   
//  of the authors and should not be interpreted as representing official policies,    
//  either expressed or implied, of the FreeBSD Project.  
#ifndef ENUM_H_
#define ENUM_H_
//
//
//
enum Freesurfer_segmentation
  {
    //No. Label Name:                                //R   G   B   A
    No_SEGMENTATION                          = 0   ,  //0   0   0   0
    LEFT_CEREBRAL_EXTERIOR                   = 1   ,  //70  130 180 0
    LEFT_CEREBRAL_WHITE_MATTER               = 2   ,  //245 245 245 0
    LEFT_CEREBRAL_CORTEX                     = 3   ,  //205 62  78  0
    LEFT_LATERAL_VENTRICLE                   = 4   ,  //120 18  134 0
    LEFT_INF_LAT_VENT                        = 5   ,  //196 58  250 0
    LEFT_CEREBELLUM_EXTERIOR                 = 6   ,  //0   148 0   0
    LEFT_CEREBELLUM_WHITE_MATTER             = 7   ,  //220 248 164 0
    LEFT_CEREBELLUM_CORTEX                   = 8   ,  //230 148 34  0
    LEFT_THALAMUS                            = 9   ,  //0   118 14  0
    LEFT_THALAMUS_PROPER                     = 10  ,  //0   118 14  0
    LEFT_CAUDATE                             = 11  ,  //122 186 220 0
    LEFT_PUTAMEN                             = 12  ,  //236 13  176 0
    LEFT_PALLIDUM                            = 13  ,  //12  48  255 0
    THIRD_VENTRICLE                          = 14  ,  //204 182 142 0
    FOURTH_VENTRICLE                         = 15  ,  //42  204 164 0
    BRAIN_STEM                               = 16  ,  //119 159 176 0
    LEFT_HIPPOCAMPUS                         = 17  ,  //220 216 20  0
    LEFT_AMYGDALA                            = 18  ,  //103 255 255 0
    LEFT_INSULA                              = 19  ,  //80  196 98  0
    LEFT_OPERCULUM                           = 20  ,  //60  58  210 0
    LINE_1                                   = 21  ,  //60  58  210 0
    LINE_2                                   = 22  ,  //60  58  210 0
    LINE_3                                   = 23  ,  //60  58  210 0
    CSF                                      = 24  ,  //60  60  60  0
    LEFT_LESION                              = 25  ,  //255 165 0   0
    LEFT_ACCUMBENS_AREA                      = 26  ,  //255 165 0   0
    LEFT_SUBSTANCIA_NIGRA                    = 27  ,  //0   255 127 0
    LEFT_VENTRALDC                           = 28  ,  //165 42  42  0
    LEFT_UNDETERMINED                        = 29  ,  //135 206 235 0
    LEFT_VESSEL                              = 30  ,  //160 32  240 0
    LEFT_CHOROID_PLEXUS                      = 31  ,  //0   200 200 0
    LEFT_F3ORB                               = 32  ,  //100 50  100 0
    LEFT_LOG                                 = 33  ,  //135 50  74  0
    LEFT_AOG                                 = 34  ,  //122 135 50  0
    LEFT_MOG                                 = 35  ,  //51  50  135 0
    LEFT_POG                                 = 36  ,  //74  155 60  0
    LEFT_STELLATE                            = 37  ,  //120 62  43  0
    LEFT_PORG                                = 38  ,  //74  155 60  0
    LEFT_AORG                                = 39  ,  //122 135 50  0
    RIGHT_CEREBRAL_EXTERIOR                  = 40  ,  //70  130 180 0
    RIGHT_CEREBRAL_WHITE_MATTER              = 41  ,  //0   225 0   0
    RIGHT_CEREBRAL_CORTEX                    = 42  ,  //205 62  78  0
    RIGHT_LATERAL_VENTRICLE                  = 43  ,  //120 18  134 0
    RIGHT_INF_LAT_VENT                       = 44  ,  //196 58  250 0
    RIGHT_CEREBELLUM_EXTERIOR                = 45  ,  //0   148 0   0
    RIGHT_CEREBELLUM_WHITE_MATTER            = 46  ,  //220 248 164 0
    RIGHT_CEREBELLUM_CORTEX                  = 47  ,  //230 148 34  0
    RIGHT_THALAMUS                           = 48  ,  //0   118 14  0
    RIGHT_THALAMUS_PROPER                    = 49  ,  //0   118 14  0
    RIGHT_CAUDATE                            = 50  ,  //122 186 220 0
    RIGHT_PUTAMEN                            = 51  ,  //236 13  176 0
    RIGHT_PALLIDUM                           = 52  ,  //13  48  255 0
    RIGHT_HIPPOCAMPUS                        = 53  ,  //220 216 20  0
    RIGHT_AMYGDALA                           = 54  ,  //103 255 255 0
    RIGHT_INSULA                             = 55  ,  //80  196 98  0
    RIGHT_OPERCULUM                          = 56  ,  //60  58  210 0
    RIGHT_LESION                             = 57  ,  //255 165 0   0
    RIGHT_ACCUMBENS_AREA                     = 58  ,  //255 165 0   0
    RIGHT_SUBSTANCIA_NIGRA                   = 59  ,  //0   255 127 0
    RIGHT_VENTRALDC                          = 60  ,  //165 42  42  0
    RIGHT_UNDETERMINED                       = 61  ,  //135 206 235 0
    RIGHT_VESSEL                             = 62  ,  //160 32  240 0
    RIGHT_CHOROID_PLEXUS                     = 63  ,  //0   200 221 0
    RIGHT_F3ORB                              = 64  ,  //100 50  100 0
    RIGHT_LOG                                = 65  ,  //135 50  74  0
    RIGHT_AOG                                = 66  ,  //122 135 50  0
    RIGHT_MOG                                = 67  ,  //51  50  135 0
    RIGHT_POG                                = 68  ,  //74  155 60  0
    RIGHT_STELLATE                           = 69  ,  //120 62  43  0
    RIGHT_PORG                               = 70  ,  //74  155 60  0
    RIGHT_AORG                               = 71  ,  //122 135 50  0
    FIFTH_VENTRICLE                          = 72  ,  //120 190 150 0
    LEFT_INTERIOR                            = 73  ,  //122 135 50  0
    RIGHT_INTERIOR                           = 74  ,  //122 135 50  0
    WM_HYPOINTENSITIES                       = 77  ,  //200 70  255 0
    LEFT_WM_HYPOINTENSITIES                  = 78  ,  //255 148 10  0
    RIGHT_WM_HYPOINTENSITIES                 = 79  ,  //255 148 10  0
    NON_WM_HYPOINTENSITIES                   = 80  ,  //164 108 226 0
    LEFT_NON_WM_HYPOINTENSITIES              = 81  ,  //164 108 226 0
    RIGHT_NON_WM_HYPOINTENSITIES             = 82  ,  //164 108 226 0
    LEFT_F1                                  = 83  ,  //255 218 185 0
    RIGHT_F1                                 = 84  ,  //255 218 185 0
    OPTIC_CHIASM                             = 85  ,  //234 169 30  0
    CORPUS_CALLOSUM                          = 192 ,  //250 255 50  0
    LEFT_FUTURE_WMSA                         = 86  ,  //200 120  255 0
    RIGHT_FUTURE_WMSA                        = 87  ,  //200 121  255 0
    FUTURE_WMSA                              = 88  ,  //200 122  255 0
    LEFT_AMYGDALA_ANTERIOR                   = 96  ,  //205 10  125 0
    RIGHT_AMYGDALA_ANTERIOR                  = 97  ,  //205 10  125 0
    DURA                                     = 98  ,  //160 32  240 0
    LEFT_WM_INTENSITY_ABNORMALITY            = 100 ,  //124 140 178 0
    LEFT_CAUDATE_INTENSITY_ABNORMALITY       = 101 ,  //125 140 178 0
    LEFT_PUTAMEN_INTENSITY_ABNORMALITY       = 102 ,  //126 140 178 0
    LEFT_ACCUMBENS_INTENSITY_ABNORMALITY     = 103 ,  //127 140 178 0
    LEFT_PALLIDUM_INTENSITY_ABNORMALITY      = 104 ,  //124 141 178 0
    LEFT_AMYGDALA_INTENSITY_ABNORMALITY      = 105 ,  //124 142 178 0
    LEFT_HIPPOCAMPUS_INTENSITY_ABNORMALITY   = 106 ,  //124 143 178 0
    LEFT_THALAMUS_INTENSITY_ABNORMALITY      = 107 ,  //124 144 178 0
    LEFT_VDC_INTENSITY_ABNORMALITY           = 108 ,  //124 140 179 0
    RIGHT_WM_INTENSITY_ABNORMALITY           = 109 ,  //124 140 178 0
    RIGHT_CAUDATE_INTENSITY_ABNORMALITY      = 110 ,  //125 140 178 0
    RIGHT_PUTAMEN_INTENSITY_ABNORMALITY      = 111 ,  //126 140 178 0
    RIGHT_ACCUMBENS_INTENSITY_ABNORMALITY    = 112 ,  //127 140 178 0
    RIGHT_PALLIDUM_INTENSITY_ABNORMALITY     = 113 ,  //124 141 178 0
    RIGHT_AMYGDALA_INTENSITY_ABNORMALITY     = 114 ,  //124 142 178 0
    RIGHT_HIPPOCAMPUS_INTENSITY_ABNORMALITY  = 115 ,  //124 143 178 0
    RIGHT_THALAMUS_INTENSITY_ABNORMALITY     = 116 ,  //124 144 178 0
    RIGHT_VDC_INTENSITY_ABNORMALITY          = 117 ,  //124 140 179 0
    EPIDERMIS                                = 118 ,  //255 20  147 0
    CONN_TISSUE                              = 119 ,  //205 179 139 0
    SC_FAT_MUSCLE                            = 120 ,  //238 238 209 0
    CRANIUM                                  = 121 ,  //200 200 200 0
    CSF_SA                                   = 122 ,  //74  255 74  0
    MUSCLE                                   = 123 ,  //238 0   0   0
    EAR                                      = 124 ,  //0   0   139 0
    ADIPOSE                                  = 125 ,  //173 255 47  0
    SPINAL_CORD                              = 126 ,  //133 203 229 0
    SOFT_TISSUE                              = 127 ,  //26  237 57  0
    NERVE                                    = 128 ,  //34  139 34  0
    BONE                                     = 129 ,  //30  144 255 0
    AIR                                      = 130 ,  //147 19  173 0
    ORBITAL_FAT                              = 131 ,  //238 59  59  0
    TONGUE                                   = 132 ,  //221 39  200 0
    NASAL_STRUCTURES                         = 133 ,  //238 174 238 0
    GLOBE                                    = 134 ,  //255 0   0   0
    TEETH                                    = 135 ,  //72  61  139 0
    LEFT_CAUDATE_PUTAMEN                     = 136 ,  //21  39  132 0
    RIGHT_CAUDATE_PUTAMEN                    = 137 ,  //21  39  132 0
    LEFT_CLAUSTRUM                           = 138 ,  //65  135 20  0
    RIGHT_CLAUSTRUM                          = 139 ,  //65  135 20  0
    CORNEA                                   = 140 ,  //134 4   160 0
    DIPLOE                                   = 142 ,  //221 226 68  0
    VITREOUS_HUMOR                           = 143 ,  //255 255 254 0
    LENS                                     = 144 ,  //52  209 226 0
    AQUEOUS_HUMOR                            = 145 ,  //239 160 223 0
    OUTER_TABLE                              = 146 ,  //70  130 180 0
    INNER_TABLE                              = 147 ,  //70  130 181 0
    PERIOSTEUM                               = 148 ,  //139 121 94  0
    ENDOSTEUM                                = 149 ,  //224 224 224 0
    R_C_S                                    = 150 ,  //255 0   0   0
    IRIS                                     = 151 ,  //205 205 0   0
    SC_ADIPOSE_MUSCLE                        = 152 ,  //238 238 209 0
    SC_TISSUE                                = 153 ,  //139 121 94  0
    ORBITAL_ADIPOSE                          = 154 ,  //238 59  59  0
    LEFT_INTCAPSULE_ANT                      = 155 ,  //238 59  59  0
    RIGHT_INTCAPSULE_ANT                     = 156 ,  //238 59  59  0
    LEFT_INTCAPSULE_POS                      = 157 ,  //62  10  205 0
    RIGHT_INTCAPSULE_POS                     = 158 ,  //62  10  205 0
    // THESE LABELS ARE FOR BABIES/CHILDREN  =     ,  //
    LEFT_CEREBRAL_WM_UNMYELINATED            = 159 ,  //0   118 14  0
    RIGHT_CEREBRAL_WM_UNMYELINATED           = 160 ,  //0   118 14  0
    LEFT_CEREBRAL_WM_MYELINATED              = 161 ,  //220 216 21  0
    RIGHT_CEREBRAL_WM_MYELINATED             = 162 ,  //220 216 21  0
    LEFT_SUBCORTICAL_GRAY_MATTER             = 163 ,  //122 186 220 0
    RIGHT_SUBCORTICAL_GRAY_MATTER            = 164 ,  //122 186 220 0
    SKULL                                    = 165 ,  //255 165 0   0
    POSTERIOR_FOSSA                          = 166 ,  //14  48  255 0
    SCALP                                    = 167 ,  //166 42  42  0
    HEMATOMA                                 = 168 ,  //121 18  134 0
    LEFT_BASAL_GANGLIA                       = 169 ,  //236 13  127 0
    RIGHT_BASAL_GANGLIA                      = 176 ,  //236 13  126 0
    // LABEL NAMES AND COLORS FOR BRAINSTEM  =     ,  //CONSITUENTS
    // NO.  LABEL NAME:                      =     ,  //R   G   B   A
    BRAINSTEM                                = 170 ,  //119 159 176 0
    DCG                                      = 171 ,  //119 0   176 0
    VERMIS                                   = 172 ,  //119 100 176 0
    MIDBRAIN                                 = 173 ,  //119 200 176 0
    PONS                                     = 174 ,  //119 159 100 0
    MEDULLA                                  = 175 ,  //119 159 200 0
    //176 RIGHT_BASAL_GANGLIA   FOUND IN BAB =     ,  //IES/CHILDREN SECTION ABOVE
    LEFT_CORTICAL_DYSPLASIA                  = 180 ,  //73  61  139 0
    RIGHT_CORTICAL_DYSPLASIA                 = 181 ,  //73  62  139 0
    //192 CORPUS_CALLOSUM  LISTED AFTER #85  =     ,  //ABOVE
    LEFT_HIPPOCAMPAL_FISSURE                 = 193 ,  //0   196 255 0
    LEFT_CADG_HEAD                           = 194 ,  //255 164 164 0
    LEFT_SUBICULUM                           = 195 ,  //196 196 0   0
    LEFT_FIMBRIA                             = 196 ,  //0   100 255 0
    RIGHT_HIPPOCAMPAL_FISSURE                = 197 ,  //128 196 164 0
    RIGHT_CADG_HEAD                          = 198 ,  //0   126 75  0
    RIGHT_SUBICULUM                          = 199 ,  //128 96  64  0
    RIGHT_FIMBRIA                            = 200 ,  //0   50  128 0
    ALVEUS                                   = 201 ,  //255 204 153 0
    PERFORANT_PATHWAY                        = 202 ,  //255 128 128 0
    PARASUBICULUM                            = 203 ,  //255 255 0   0
    PRESUBICULUM                             = 204 ,  //64  0   64  0
    SUBICULUM                                = 205 ,  //0   0   255 0
    CA1                                      = 206 ,  //255 0   0   0
    CA2                                      = 207 ,  //128 128 255 0
    CA3                                      = 208 ,  //0   128 0   0
    CA4                                      = 209 ,  //196 160 128 0
    GC_ML_DG                                 = 210 ,  //32  200 255 0
    HATA                                     = 211 ,  //128 255 128 0
    FIMBRIA                                  = 212 ,  //204 153 204 0
    LATERAL_VENTRICLE                        = 213 ,  //121 17  136 0
    MOLECULAR_LAYER_HP                       = 214 ,  //128 0   0   0
    HIPPOCAMPAL_FISSURE                      = 215 ,  //128 32  255 0
    ENTORHINAL_CORTEX                        = 216 ,  //255 204 102 0
    MOLECULAR_LAYER_SUBICULUM                = 217 ,  //128 128 128 0
    AMYGDALA                                 = 218 ,  //104 255 255 0
    CEREBRAL_WHITE_MATTER                    = 219 ,  //0   226 0   0
    CEREBRAL_CORTEX                          = 220 ,  //205 63  78  0
    INF_LAT_VENT                             = 221 ,  //197 58  250 0
    PERIRHINAL                               = 222 ,  //33  150 250 0
    CEREBRAL_WHITE_MATTER_EDGE               = 223 ,  //226 0   0   0
    BACKGROUND                               = 224 ,  //100 100 100 0
    ECTORHINAL                               = 225 ,  //197 150 250 0
    HP_TAIL                                  = 226 ,  //170 170 255 0
    FORNIX                                   = 250 ,  //255 0   0   0
    CC_POSTERIOR                             = 251 ,  //0   0   64  0
    CC_MID_POSTERIOR                         = 252 ,  //0   0   112 0
    CC_CENTRAL                               = 253 ,  //0   0   160 0
    CC_MID_ANTERIOR                          = 254 ,  //0   0   208 0
    CC_ANTERIOR                              = 255    //0   0   255 0
  };
//
//
//
enum Brain_segmentation 
  {
    NO_SEGMENTATION           = 0,  //
    OUTSIDE_SCALP             = 1,  //
    OUTSIDE_SKULL             = 2,  //
    CEREBROSPINAL_FLUID       = 3,  //  24 CSF 
    WHITE_MATTER              = 4,  //  77 WM_HYPOINTENSITIES 
                                    //  41 RIGHT_CEREBRAL_WHITE_MATTER 
                                    //   2  LEFT_CEREBRAL_WHITE_MATTER
    GRAY_MATTER               = 5,  //  42 RIGHT_CEREBRAL_CORTEX 
                                    //   3  LEFT_CEREBRAL_CORTEX 
    BRAIN_STEM_SUBCORTICAL    = 6,  //  16  BRAIN_STEM
    HIPPOCAMPUS               = 7,  //  53 RIGHT_HIPPOCAMPUS 
                                    //  17  LEFT_HIPPOCAMPUS
    AMYGDALA_SUBCORTICAL      = 8,  //  5  RIGHT_AMYGDALA 
                                    //  18  LEFT_AMYGDALA
    CAUDATE                   = 9,  //  50 RIGHT_CAUDATE 
                                    //  11  LEFT_CAUDATE 
    PUTAMEN                   = 10, //  12  LEFT_PUTAMEN 
                                    //  51 RIGHT_PUTAMEN
    THALAMUS                  = 11, //  10  LEFT_THALAMUS_PROPER  
                                    //  49 RIGHT_THALAMUS_PROPER 
    ACCUMBENS                 = 12, //  26  LEFT_ACCUMBENS_AREA 
                                    //  58 RIGHT_ACCUMBENS_AREA 
    PALLIDUM                  = 13, //  13  LEFT_PALLIDUM 
                                    //  52 RIGHT_PALLIDUM
    VENTRICLE                 = 14, //   4  LEFT_LATERAL_VENTRICLE 
                                    //  43 RIGHT_LATERAL_VENTRICLE 
                                    //   5  LEFT_INF_LAT_VENT 
                                    //  44 RIGHT_INF_LAT_VENT
                                    //  15  FOURTH_VENTRICLE 
                                    //  14  THIRD_VENTRICLE 
                                    //  72  FIFTH_VENTRICLE 
    PLEXUS                    = 15, //  63 RIGHT_CHOROID_PLEXUS 
                                    //  31  LEFT_CHOROID_PLEXUS
    FORNIX_SUBCORTICAL        = 16, // 250 FORNIX
    CORPUS_COLLOSUM           = 17, // 251 CC_POSTERIOR 
                                    // 252 CC_MID_POSTERIOR 
                                    // 253 CC_CENTRAL 
                                    // 254 CC_MID_ANTERIOR 
                                    // 255 CC_ANTERIOR 
    VESSEL                    = 18, //  30  LEFT_VESSEL 
                                    //  62 RIGHT_VESSEL 
    CEREBELLUM_GRAY_MATTER    = 19, //   8  LEFT_CEREBELLUM_CORTEX 
                                    //  47 RIGHT_CEREBELLUM_CORTEX
    CEREBELLUM_WHITE_MATTER   = 20, //   7  LEFT_CEREBELLUM_WHITE_MATTER 
                                    //  46 RIGHT_CEREBELLUM_WHITE_MATTER
    VENTRAL_DIENCEPHALON      = 21, //  28  LEFT_VENTRALDC 
                                    //  60 RIGHT_VENTRALDC
    OPTIC_CHIASM_SUBCORTICAL  = 22, //  85 OPTIC_CHIASM
    //
    //
    ELECTRODE                 = 100,
    //
    //
    SPONGIOSA_SKULL           = 130,
    AIR_IN_SKULL              = 131,
    EYE                       = 132,
  }; 
//
//
//
enum Mesh_output 
  {
    NO_OUTPUT         = 0,
    MESH_OUTPUT       = 1,
    MESH_SUBDOMAINS   = 2,
    MESH_VTU          = 3,
    MESH_CONDUCTIVITY = 4,
    MESH_DIPOLES      = 5,
  };
//
//
//
enum Electrode_type
  {
    NO_TYPE     = 0,
    SPHERE      = 1,
    CYLINDER    = 2,
  };


#endif
