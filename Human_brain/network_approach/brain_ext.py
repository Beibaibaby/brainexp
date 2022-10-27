from nilearn import datasets
from nilearn import image as nimg
from nilearn import plotting as nplot
parcel_dir = '../resources/rois/'
atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011(parcel_dir)
atlas_yeo_2011.keys()
#Define where to slice the image
cut_coords=(8, -4, 9)
#Show a colorbar
colorbar=True
#Color scheme to show when viewing image

atlas_yeo = atlas_yeo_2011['thick_7']
from nilearn.regions import connected_label_regions
region_labels = connected_label_regions(atlas_yeo)

nplot.plot_roi(region_labels,
			cut_coords=(-20,-10,0,10,20,30,40,50,60,70),
			display_mode='z',
			colorbar=True,
			cmap='Paired',
			title='Relabeled Yeo Atlas')

region_labels.to_filename('../resources/rois/yeo_2011/Yeo_JNeurophysiol11_MNI152/relabeled_yeo_atlas.nii.gz')