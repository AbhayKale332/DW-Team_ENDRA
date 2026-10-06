# FAQs

## Dataset related :

1. Reference DEM: Besides SRTM 30m, may we use other open DEMs such as Copernicus DEM GLO-30 for scale calibration?
-> **Yes**, you can use any DEM source.

2. Reference data: Will reference LiDAR/DSM data for the urban, sparse, hilly and forested test cases be shared during screening, or should teams use openly available sources?
->You can use **any openly available data sources** for that.

3. Any reference data will be provided?
-> We **will not provide** any additional data.


## Model related :
1.  Would a foundation backbone fine-tuned for height be acceptable, or must the backbone be depth-pretrained?
-> You can **use anything**, but make sure that for GeoTIFF images, the values match DEM heights. For non-GeoTIFF images, relative heights can be at any acceptable scale.

2.  Can we use GAN models rather than depth-pretrained models?
-> You can use **any methodology** that suits you to produce good results. We have just shown one possible way to achieve the expected results.

## Evaluation related : 

1. Evaluation imagery: Which sensor and approximate spatial resolution will the ISRO RGB images used for final evaluation have (for example, Cartosat at 0.5m/1 m/2.5 m)?
-> Cartosat 2S imageries with resolution of 0.6m will be used for final evaluation but your model should work on any image which have spatial resolution of 0.35m to 10m (**Please do not overfit model with 0.6m data**).

2. Evaluation target: Will accuracy be assessed on the absolute DSM (terrain + structures) or on above-ground heights of structures (nDSM)?
-> We will provide Geotiff images during evaluation so we will evaluate with absolute DSM such as SRTM or Copernicus. but your solution should accept non-geotiff images too.

3. Whether a suitable georeferenced RGB dataset with corresponding reference DEM/DSM/LiDAR height data will be provided during the final evaluation, or whether teams are expected to source the reference data independently?
-> **No**, we will only provide georeferenced RGB Images (like tiff files) or normal RGB Images (like png) for inferencing, so team have to source the reference data independently.

4. Is the dataset at https://github.com/IMG-PROCESS-SAC/SIH2026/ representative of what will be used during final evaluation, or will a separate held-out dataset be used for judging?
-> There will be a separate dataset for evaluation.

5. Is there a minimum accuracy/RMSE threshold expected for the DSM estimation, or is scoring purely relative to other participating teams?
-> It is **relative** to other participants.

6. Would acquisition timestamp and/or sun-angle information generally be preserved in GeoTIFF metadata, or should we design our pipeline to also handle imagery where such fields are stripped?
-> We will provide nadir-viewed images, so please expect that sun-angle and time of acquisition will not be there in metadata.


## Execution related : 

1. Deployment: Is a locally hosted web application acceptable as a "standalone deployment", or is a desktop executable expected?
-> **both** are acceptable.

2. can we add any additional features which are not mentioned in the problem statement?
-> yes, you can add but there will be no additional marking for that. but it will definitely make impact on the viewer.

3. For georeferenced inputs such as GeoTIFF, should the solution be expected to handle the complete geospatial metadata and coordinate reference system directly?
-> Yes, that you can do easily with existing libraries.


## Presentation related :

1. For the final presentation, would you advise a more technically detailed deck (architecture, loss design, ablations) or one focused on the problem, the output, and measured accuracy, with technical detail minimized?
-> For the initial submission, include only the problem understanding, implementation idea, proposed architecture, and any preliminary work you have done in **last** (updated from **one extra**) slide, or you can add a remote repository link in last slide (please maintain proper documentation for your work in markdown file). However, after selection, more detailed technical documentation will be needed.


 







