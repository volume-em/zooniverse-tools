# CEM Zooniverse Tools
Scripts and notebooks for data wrangling from Zooniverse.

After organizing images/flipbooks into a single directory, our workflow is:

**[1. (Optional) Anonymize names](#anonymize)**

Dataset names sometimes have sensitive information like PI or collaborator
name. The ```anonymize_names.py``` script replaces those details with
a random alphanumeric string. A lookup table for old and new dataset names
are saved in a .csv file.

```bash
python ../zooniverse_tools/anonymize_names.py flipbooks/
```

**[2. Prepare images/flipbooks for upload to Zooniverse](#prepare)**

For grayscale images use the ```prep_images.py``` script or for flipbooks use 
```prep_flipbooks.py```. These scripts (a) resize all images/flipbooks to a fixed
square size large enough to be easily viewable in the Zooniverse web interface (b) 
optionally, normalize the intensity of images for improved contrast, and (c) save 
images as jpg or flipbooks as a sequence of jpgs.


```bash
python ../zooniverse_tools/prep_flipbooks.py flipbooks/ images/ --size 512 --contrast
```

**3. Upload the images to Zooniverse**

Use the ```upload_images.ipynb``` notebook to upload to a Zooniverse
subject set. Fill in the required fields in the first code cell and
then run cells sequentially. The result will be a Zooniverse subject
set with the prepared images/flipbooks uploaded and ready for annotation.

**4. Download the annotations**

Login to Zooniverse and perform a classification export, see [here])(https://help.zooniverse.org/next-steps/data-exports/).

**5. Create stacks of images and consensus segmentations for proofreading**

Assuming the primary Zooniverse task was instance segmentation by drawing polygons and 
there was a secondary task to rate annotation confidence, use the ```create_proofreading_stacks.py```
script. This is the most important script in the workflow. It takes the polygons,
converts them to instance segmentations, and then creates a consensus instance segmentation
from multiple independent annotations. The output from this script is (a) stack of images/flipbooks
in a single .tif file (b) stack of consensus segmentation images/flipbooks in a matching .tif file (c)
a consensus_attributes.csv file that stores info about image name, shape, annotation consensus strength,
median confidence score, etc. The stacks will be 4D (N, L, H, W) for flipbooks and 3D (N, H, W)
for regular grayscale images.

```bash
python ../zooniverse-tools/create_proofreading_stacks.py flipbook-classifications.csv \
        flipbooks/ unproofed/ --size 512 --flipbook-n 5
```

**6. (Optional) Split the stacks into smaller chunks**

Zooniverse subject sets ideally contain 500-1000 images. The proofreading of these
large stacks can be distributed and shared between within a group by splitting 
into manageable chunks. Use the ```split_to_chunks.py``` script for this. It takes in
the 3 output files from ```create_proofreading_stacks.py``` and breaks them into
parts.

```bash
python ../zooniverse-tools/split_to_chunks.py unproofed/flipbook_image.tif \
        unproofed/flipbook_cs_masks.tif unproofed/flipbook_consensus_attributes.csv \
        unproofed_chunks/ --cs 50
```

**7. Proofread the consensus segmentations**

Use napari to view and proofread image/flipbook segmentations. (Reminder: only 
the middle slice or each flipbook includes an annotation). The proofreading tools
provided by [empanada-napari](https://github.com/volume-em/empanada-napari/tree/main)
are highly recommended.

**8. (Optional) Concatenate corrected segmentation chunks into a single file**

**This only applies if step 6 was completed.** Since only segmentations are 
modified during proofreading, the ```concat_mask_chunks.py``` script only applies
to labelmap chunks. It assumes that all proofread chunks are stored in the same directory
with a particular substring that can be used to identify/distinguish them from 
other files.

```bash
python ../zooniverse-tools/concat_mask_chunks.py proofed/ masks
```

**9. Save images and segmentations to directories**

The final step is to split the image and segmentation stacks
into individual tiff images that can be used for training
deep learning models (i.e., with [empanada](https://github.com/volume-em/empanada)). The
```save_stacks.py``` script takes in the image stack and consensus_attributes.csv generated
in step 5 and the proofread segmentation stack created in step 7 or 8. The output is the following
directory structure:

```
training_data
│
└───dataset1
│   │
│   └───images
│   │   │   image1.tiff
│   │   │   image2.tiff
│   │   │   ...
│   │
│   └───masks
│   │   │   image1.tiff
│   │   │   image2.tiff
│   │   │   ...
└───dataset2
│   │
│   └───images
│   │   │   image1.tiff
│   │   │   image2.tiff
│   │   │   ...
│   │
│   └───masks
│   │   │   image1.tiff
│   │   │   image2.tiff
│   │   │   ...
```

```bash
python ../zooniverse-tools/save_stacks.py proofed/flipbook_image.tif \
        proofed/flipbook_proofed_masks.tif proofed/flipbook_consensus_attributes.csv \
        training_data/

```

**NOTE: If the directory training_data already exists, the segmentations will simply
be added into the existing structure with datasets still remaining correctly grouped.**
