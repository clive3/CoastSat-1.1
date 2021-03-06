from skimage.filters import gaussian, threshold_otsu
from datetime import date

from coastsat.SDS_shoreline import *

from coastsat import NOC_download, NOC_preprocess, NOC_classify, NOC_tools

from utils.print_utils import printProgress, printSuccess, printWarning, printError
from utils.name_utils import jpegFilePath, geojsonFilePath, pickleDumpName

def extract_shoreline_optical(settings, ref=False, batch=False):

    metadata = NOC_download.load_metadata(settings)

    ref_date_start = date.fromisoformat(settings['date_range'][0])
    ref_date_end = date.fromisoformat(settings['date_range'][1])
    meta_date_start = date.fromisoformat(settings['dates'][0])
    meta_date_end = date.fromisoformat(settings['dates'][1])

    if batch:
        if not (meta_date_start >= ref_date_start and meta_date_end <= ref_date_end):
            raise Exception

    sat_name = settings['sat_name']
    median_dir_path = settings['median_dir_path']

    classes = settings['classes']
    output_epsg = settings['output_epsg']
    band_dict = settings['bands'][sat_name]
    first_key = next(iter(band_dict))
    pixel_size = band_dict[first_key][1]
    settings['pixel_size'] = pixel_size
    bands_20m = band_dict['20m'][0]
    for SWIR_index, SWIR_band in enumerate(bands_20m):
        if settings['SWIR'] == SWIR_band:
            break

    file_names = metadata['file_names']
    models_file_path = os.path.join(os.getcwd(), 'classification', 'models')

    pansharpen = settings['pansharpen']
    if pansharpen:
        shoreline_folder = 'pan'
    else:
        shoreline_folder = 'standard'

    shoreline_dir_path = os.path.join(median_dir_path, 'shorelines', shoreline_folder)
    if not os.path.exists(shoreline_dir_path):
        os.makedirs(shoreline_dir_path)

    # create a subfolder to store the .jpg images showing the detection
    jpeg_file_path = os.path.join(median_dir_path, 'jpg_files', 'detection')
    if not os.path.exists(jpeg_file_path):
        os.makedirs(jpeg_file_path)
    # close all open figures
    plt.close('all')

    # load classifier
    if sat_name in ['L5', 'L7', 'L8']:
        classifier = joblib.load(os.path.join(models_file_path, 'NN_4classes_Landsat.pkl'))

    elif sat_name == 'S2':
        classifier = joblib.load(os.path.join(models_file_path, 'NN_6classes_S2.pkl'))

    file_paths = []
    for band_index, band_key in enumerate(band_dict.keys()):
        file_paths.append(os.path.join(median_dir_path, sat_name, band_key, file_names[band_index]))

    # convert settings['min_beach_area'] from  metres to pixels
    min_beach_area_pixels = np.ceil(settings['min_beach_area'] / pixel_size ** 2)

    printProgress('image loaded')
    # preprocess image (cloud mask + pansharpening/downsampling)
    image_ms, georef, cloud_mask, image_extra, image_QA, image_nodata = \
                                            NOC_preprocess.preprocess_optical(settings, file_paths,
                                                                              pansharpen=pansharpen,
                                                                              SWIR_band=SWIR_band,
                                                                              SWIR_index=SWIR_index)

    # get image spatial reference system (epsg code) from metadata dict
    image_epsg = int(metadata['epsg'])
    settings['image_epsg'] = image_epsg
    settings['georef'] = georef

    # compute cloud_cover percentage (with no data pixels)
    cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                     (cloud_mask.shape[0] * cloud_mask.shape[1]))
    if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
        return []

    # remove no data pixels from the cloud mask
    # (for example L7 bands of no data should not be accounted for)
    cloud_mask_adv = np.logical_xor(cloud_mask, image_nodata)
    # compute updated cloud cover percentage (without no data pixels)
    cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                            (cloud_mask.shape[0] * cloud_mask.shape[1]))
    # skip image if cloud cover is above user-defined threshold
    if cloud_cover > settings['cloud_thresh']:
        return []

    buffer_shape = cloud_mask.shape
    image_ref_buffer = create_shoreline_buffer(settings, buffer_shape, ref=ref)
    mndwi_buffer = create_mndwi_buffer(settings, buffer_shape)

    printProgress('classifying image')
    # classify image with NN classifier
    _, image_labels = NOC_classify.classify_image_NN(image_ms, classes, cloud_mask,
                                                     min_beach_area_pixels, classifier)

    # find the shoreline interactively
    shoreline, _, skip_image = adjust_detection_optical(settings, image_ms, cloud_mask, image_labels,
                                                        image_ref_buffer, mndwi_buffer,
                                                        ref=ref,
                                                        batch=batch)
    if skip_image:
        printWarning('shoreline skipped')
    else:
        gdf = NOC_tools.output_to_gdf(shoreline, metadata)
        shoreline_file_path = geojsonFilePath(settings, batch=batch)

        if ~gdf.empty:
            gdf.crs = {'init': 'epsg:' + str(output_epsg)}
            gdf.to_file(shoreline_file_path, driver='GeoJSON', encoding='utf-8')

            printSuccess(f'shoreline saved {shoreline_file_path}')
        else:
            printWarning('no shorelines to be seen ...')

    # close figure window if still open
    if plt.get_fignums():
        plt.close()


def adjust_detection_optical(settings, image_ms, cloud_mask, image_labels,
                             image_ref_buffer, mndwi_buffer, ref=False, batch=False):

    sat_name = settings['sat_name']
    site_name = settings['site_name']
    pansharpen = settings['pansharpen']
    image_epsg = settings['image_epsg']
    output_epsg = settings['output_epsg']
    georef = settings['georef']

    if ref:
        date_start = settings['date_range'][0]
        date_end = settings['date_range'][1]
    else:
        date_start = settings['dates'][0]
        date_end = settings['dates'][1]

    image_RGB = SDS_preprocess.rescale_image_intensity(image_ms[:, :, [2, 1, 0]], cloud_mask, 99.9)
    image_classified = np.ones(image_RGB.shape)

    # compute MNDWI grayscale image
    image_mndwi = SDS_tools.nd_index(image_ms[:, :, 4], image_ms[:, :, 1], cloud_mask)
    # buffer MNDWI using reference shoreline
    image_mndwi_buffered = np.copy(image_mndwi)
    image_mndwi_buffered[~mndwi_buffer] = np.nan

    if batch:
        ##########################################################################
        ##
        ## BATCH MODE
        ##
        ##########################################################################
        skip_image = False
        t_mndwi = settings['reference_threshold']
        contours_mndwi = measure.find_contours(image_mndwi_buffered, level=t_mndwi, mask=image_ref_buffer)
        shoreline = create_shoreline(settings, contours_mndwi)
        printProgress('shoreline extracted')
    else:
        ##########################################################################
        ##
        ## INTERACTIVE MODE
        ##
        ##########################################################################
        # for each class add it to image_classified and
        # extract the MDWI values for all pixels in that class
        classes = settings['classes']
        class_keys = classes.keys()

        mndwi_pixels = {}
        for key in class_keys:
            class_label = classes[key][0]
            class_index = class_label - 1
            class_colour = classes[key][1]

            image_classified[image_labels[:, :, class_index], 0] = class_colour[0]
            image_classified[image_labels[:, :, class_index], 1] = class_colour[1]
            image_classified[image_labels[:, :, class_index], 2] = class_colour[2]

            mndwi_pixels[class_label] = image_mndwi[image_labels[:, :, class_index]]

        # create figure
        if plt.get_fignums():
            # if it exists, open the figure
            fig = plt.gcf()
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            ax3 = fig.axes[2]
            ax4 = fig.axes[3]
        else:
            # else create a new figure
            fig = plt.figure()
            fig.set_size_inches([18, 9])
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])
            gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
            ax4 = fig.add_subplot(gs[1, :])
        ##########################################################################
        # to do: rotate image if too wide
        ##########################################################################

        # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
        nan_color = 1.0
        image_RGB = np.where(np.isnan(image_RGB), nan_color, image_RGB)
        image_class = np.where(np.isnan(image_classified), 1, image_classified)

        # plot image 1 (RGB)
        ax1.imshow(image_RGB)
        ax1.axis('off')
        ax1.set_title(site_name, fontweight='bold', fontsize=16)

        # plot image 2 (classification)
        ax2.imshow(image_class)
        ax2.axis('off')
        ax2.set_title(date_start + ' to ' + date_end, fontweight='bold', fontsize=12)

        # plot image 3 (MNDWI)
        ax3.imshow(image_mndwi, cmap='bwr')
        ax3.axis('off')
        ax3.set_title('MNDWI', fontweight='bold', fontsize=16)
        ax3.imshow(image_ref_buffer, cmap='binary', alpha=0.3)

        # plot histogram of MNDWI values
        bin_width = 0.01
        ax4.set_facecolor('0.75')
        ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
        ax4.set(ylabel='pixels', yticklabels=[], xlim=[-1, 1])

        for class_display_order in range(len(classes)):

            for key in class_keys:

                if classes[key][2] == class_display_order:

                    class_name = key
                    class_label = classes[key][0]
                    class_colour = classes[key][1]
                    class_pixels = mndwi_pixels[class_label]

                    if key in ['white-water', 'sand']:
                        alpha = 0.75
                    elif key in['nature', 'urban']:
                        alpha = 0.9
                    else:
                        alpha = 1.0

            if len(class_pixels) > 0:
                bins = np.arange(np.nanmin(class_pixels), np.nanmax(class_pixels) + bin_width, bin_width)
                ax4.hist(class_pixels, bins=bins, density=True,  color=class_colour, label=class_name, alpha=alpha)

        if ref:
            contours_mndwi, t_mndwi = find_contours_optical(image_ms, image_labels, cloud_mask, image_ref_buffer)
            reference_threshold = t_mndwi

        else:
            reference_threshold = settings['reference_threshold']
            t_mndwi = reference_threshold
            contours_mndwi = measure.find_contours(image_mndwi_buffered, level=t_mndwi, mask=image_ref_buffer)

        # process the water contours into a shoreline
        settings['min_length_sl'] = settings['min_length_shoreline']
        shoreline = process_shoreline(contours_mndwi, cloud_mask, georef, image_epsg, settings)
        # convert shoreline to pixels
        if len(shoreline) > 0:
            sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                        output_epsg,
                                                                        image_epsg)[:, [0, 1]], georef)
        else:
            sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        # plot the shoreline on the images
        sl_plot1 = ax1.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
        sl_plot2 = ax2.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
        sl_plot3 = ax3.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
        t_line = ax4.axvline(x=t_mndwi, ls='--', c='k', lw=1.5, label=f'threshold')
        thresh_label = ax4.text(t_mndwi+bin_width, 4, str(f'{t_mndwi:4.3f}'), rotation=90)

        ax4.axvline(x=reference_threshold, ls='--', c='r', lw=1.5, label=f'ref threshold {reference_threshold:4.3f}')

        ax4.legend(loc=1)
        plt.draw()  # to update the plot
        # adjust the threshold manually by letting the user change the threshold
        ax4.set_title(
            'Click on the plot below to change the location of the threhsold and adjust the shoreline detection. When finished, press <Enter>')
        while True:
            # let the user click on the threshold plot
            pt = ginput(n=1, show_clicks=True, timeout=-1)
            # if a point was clicked
            if len(pt) > 0:
                # update the threshold value
                t_mndwi = pt[0][0]
                # if user clicked somewhere wrong and value is not between -1 and 1
                if np.abs(t_mndwi) >= 1: continue

                # update the plot
                t_line.set_xdata([t_mndwi, t_mndwi])
                thresh_label.set(x=t_mndwi+bin_width, text=str(f'{t_mndwi:4.3f}'))
                # map contours with new threshold
                contours = measure.find_contours(image_mndwi_buffered, level=t_mndwi,  mask=image_ref_buffer)
                # remove contours that contain NaNs (due to cloud pixels in the contour)
                contours = process_contours(contours)
                # process the water contours into a shoreline
                shoreline = process_shoreline(contours, cloud_mask, georef, image_epsg, settings)
                # convert shoreline to pixels
                if len(shoreline) > 0:
                    sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                                output_epsg,
                                                                                image_epsg)[:, [0, 1]], georef)
                else:
                    sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
                # update the plotted shoreline
                sl_plot1[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
                sl_plot2[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
                sl_plot3[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])

                fig.canvas.draw_idle()
            else:
                ax4.set_title('MNDWI pixel intensities and threshold')
                break

        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        key_event = {}

        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key

        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        skip_image = False

        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k', fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k', fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                               transform=ax1.transAxes,
                               bbox=dict(boxstyle="square", ec='k', fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()

            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled checking shoreline detection')
            else:
                plt.waitforbuttonpress()

         # if not skipping save the jpeg shpowing selection
        if not skip_image:

            if pansharpen:
                sat_name += '_PS'
            jpeg_type = 'detection'
            jpeg_file_path = jpegFilePath(settings, jpeg_type, sat_name, date_start, date_end)
            fig.savefig(jpeg_file_path, dpi=150)

        plt.close()

        if not ref and not skip_image:
            printProgress('shoreline extracted')

    return shoreline, t_mndwi, skip_image


def find_contours_optical(image_ms, image_labels, cloud_mask, ref_shoreline_buffer):

    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # calculate Modified Normalized Difference Water Index
    image_mndwi = SDS_tools.nd_index(image_ms[:, :, 4], image_ms[:, :, 1], cloud_mask)

    # calculate Normalized Difference Modified Water Index
    image_ndwi = SDS_tools.nd_index(image_ms[:, :, 3], image_ms[:, :, 1], cloud_mask)

    # stack indices together
    image_indices = np.stack((image_ndwi, image_mndwi), axis=-1)
    vec_indices = image_indices.reshape(nrows*ncols, 2)

    # reshape labels into vectors
#    vec_land1 = image_labels[:, :, 0].reshape(ncols * nrows)
    vec_land2 = image_labels[:, :, 1].reshape(ncols * nrows)
    vec_land3 = image_labels[:, :, 2].reshape(ncols * nrows)
#    vec_sand = image_labels[:, :, 5].reshape(ncols * nrows)
    vec_all_land = np.logical_or(vec_land2, vec_land3)

    vec_water = image_labels[:, :, 4].reshape(ncols * nrows)

    vec_image_ref_buffer = ref_shoreline_buffer.reshape(ncols * nrows)
    # select land and water pixels that are within the buffer
    int_land = vec_indices[np.logical_and(vec_image_ref_buffer,vec_all_land),:]
    int_sea = vec_indices[np.logical_and(vec_image_ref_buffer,vec_water),:]
    
    int_land = vec_indices[vec_all_land]
    int_sea = vec_indices[vec_water]

    # make sure both classes have the same number of pixels before thresholding
    if len(int_land) > 0 and len(int_sea) > 0:
        if np.argmin([int_land.shape[0],int_sea.shape[0]]) == 1:
            int_land = int_land[np.random.choice(int_land.shape[0],int_sea.shape[0], replace=False),:]
        else:
            int_sea = int_sea[np.random.choice(int_sea.shape[0],int_land.shape[0], replace=False),:]
    

    # threshold the sand/water intensities
    int_all = np.append(int_land, int_sea, axis=0)
    int_all = int_all[~np.isnan(int_all)]
    t_mwi = threshold_otsu(int_all)

    # find contour with MS algorithm
    image_mdwi_buffered = np.copy(image_mndwi)
    image_mdwi_buffered[~ref_shoreline_buffer] = np.nan
    contours_mwi = measure.find_contours(image_mdwi_buffered, level=t_mwi, mask=ref_shoreline_buffer)

    # remove contour points that are NaNs (around clouds)
    contours_mwi = process_contours(contours_mwi)

    return contours_mwi, t_mwi


def extract_shoreline_sar(settings, ref=False, batch=False):

    metadata = NOC_download.load_metadata(settings)
    output_epsg = settings['output_epsg']

    median_dir_path = settings['median_dir_path']
    sat_name = settings['sat_name']
    pixel_size = settings['pixel_size']

    file_names = metadata['file_names']
    settings['image_epsg'] = int(metadata['epsg'])
    settings['pixel_size'] = pixel_size

    printProgress('mapping shoreline')

    for file_index, file_name in enumerate(file_names):

        file_path = os.path.join(median_dir_path, sat_name, file_name)

        # close all open figures
        plt.close('all')

        ## read the geotiff
        sar_image, georef = NOC_preprocess.preprocess_sar(file_path)
        settings['georef'] = georef

        # calculate a buffer around the reference shoreline if it has already been generated
        buffer_shape = (sar_image.shape[0], sar_image.shape[1])
        image_ref_buffer = create_shoreline_buffer(settings, buffer_shape)

        shoreline, _, _ = adjust_detection_sar(settings, sar_image, image_ref_buffer,
                                               ref=ref,
                                               batch=batch)

        gdf = NOC_tools.output_to_gdf(shoreline, metadata)
        shoreline_file_path = geojsonFilePath(settings, batch=batch)

        if ~gdf.empty:
            gdf.crs = {'init':'epsg:'+str(output_epsg)} # set layer projection
            # save GEOJSON layer to file
            gdf.to_file(shoreline_file_path, driver='GeoJSON', encoding='utf-8')

            printSuccess('shoreline saved')
        else:
            printWarning('no shorelines to be seen ...')

    # close figure window if still open
    if plt.get_fignums():
        plt.close()


def adjust_detection_sar(settings, sar_image, image_ref_buffer, ref=False, batch=False):

    image_epsg = settings['image_epsg']
    output_epsg = settings['output_epsg']
    georef = settings['georef']

    sat_name = settings['sat_name']
    polarisation = settings['polarisation']
    date_start = settings['dates'][0]
    data_end = settings['dates'][1]

    site_name = settings['site_name']

    # compute classified image
    if polarisation == 'VV':
        band_index = 0
        colour = [1, 0, 1]
    elif polarisation == 'VH':
        band_index = 1
        colour = [0, 1, 1]
    else:
        printError(f'select SAR band correctly: {polarisation}')

    image_pol = np.copy(sar_image[:,:,0])

    vec_shape = (sar_image.shape[0] * sar_image.shape[1])
    vec_pol = image_pol.reshape(vec_shape)

    if batch:
        ##########################################################################
        ##
        ## BATCH MODE
        ##
        ##########################################################################
        skip_image = False
        t_sar = settings['reference_threshold']
        contours_sar = measure.find_contours(image_pol, level=t_sar, mask=image_ref_buffer)
        shoreline = create_shoreline(settings, contours_sar)

    else:

        ##########################################################################
        ##
        ## INTERACTIVE MODE
        ##
        ##########################################################################
        # create figure
        if plt.get_fignums():
            # if it exists, open the figure
            fig = plt.gcf()
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            ax3 = fig.axes[2]
            ax4 = fig.axes[3]
        else:
            # else create a new figure
            fig = plt.figure()
            fig.set_size_inches([18, 9])
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1])
            gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
            ax4 = fig.add_subplot(gs[1, :])
        ##########################################################################
        # to do: rotate image if too wide
        ##########################################################################

        # plot image 1 (RGB)
        ax1.imshow(image_pol)
        ax1.axis('off')
        ax1.set_title(site_name, fontweight='bold', fontsize=16)

        # plot image 1 (grey)
        ax2.imshow(image_pol, cmap='gray')
        ax2.axis('off')
        ax2.set_title(date_start + ' to ' + data_end, fontweight='bold', fontsize=16)

        # plot image 3 (blue/red)
        ax3.imshow(image_pol, cmap='bwr')
        ax3.axis('off')
        ax3.set_title(polarisation, fontweight='bold', fontsize=16)
        ax3.imshow(image_ref_buffer, cmap='binary', alpha=0.3)

        # plot histogram of sigma values
        ax4.set_facecolor('0.75')
        ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
        ax4.set(ylabel='pixels', yticklabels=[], xlim=[np.nanmin(sar_image), np.nanmax(sar_image)])

        bin_width = 0.1
        bins = np.arange(np.nanmin(vec_pol), np.nanmax(vec_pol) + bin_width, bin_width)
        ax4.hist(vec_pol, bins=bins, density=True, color=colour, label=polarisation)

        if ref:
            reference_threshold = 0
            t_sar = filters.threshold_otsu(image_pol)
        else:
            reference_threshold = settings['reference_threshold']
            t_sar = reference_threshold

        contours_sar = measure.find_contours(image_pol, level=t_sar, mask=image_ref_buffer)

        # process the water contours into a shoreline
        shoreline = create_shoreline(settings, contours_sar)

        # convert shoreline to pixels
        if len(shoreline) > 0:
            sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                        output_epsg,
                                                                        image_epsg)[:, [0, 1]], georef)
        else:
            printWarning('no shoreline yet')
            sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])

        # plot the shoreline on the images
        sl_plot1 = ax1.plot(sl_pix[:, 0], sl_pix[:, 1], 'r', markersize=3)
        sl_plot2 = ax2.plot(sl_pix[:, 0], sl_pix[:, 1], 'r', markersize=3)
        sl_plot3 = ax3.plot(sl_pix[:, 0], sl_pix[:, 1], 'r', markersize=3)

        t_line = ax4.axvline(x=t_sar, ls='--', c='k', lw=1.5, label='threshold')
        thresh_label = ax4.text(t_sar + bin_width, 0.25, str(f'{t_sar:4.2f}'), rotation=90)
        ax4.axvline(x=reference_threshold, ls='--', c='r', lw=1.5, label=f'ref threshold {t_sar:4.2f}')

        ax4.legend(loc=1)
        plt.draw()  # to update the plot
        # adjust the threshold manually by letting the user change the threshold
        ax4.set_title(
            'Click on the plot below to change the location of the threhsold and adjust the shoreline detection. When finished, press <Enter>')

        while True:
            # let the user click on the threshold plot
            pt = ginput(n=1, show_clicks=True, timeout=-1)
            # if a point was clicked
            if len(pt) > 0:
                # update the threshold value
                t_sar = pt[0][0]

                # update the plot
                t_line.set_xdata([t_sar, t_sar])
                thresh_label.set(x=t_sar + bin_width, text=str(f'{t_sar:4.2f}'))
                # map contours with new threshold
                contours_sar = measure.find_contours(image_pol, level=t_sar, mask=image_ref_buffer)
                # process the water contours into a shoreline
                shoreline = create_shoreline(settings, contours_sar)

                if len(shoreline) > 0:
                    sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                                output_epsg,
                                                                                image_epsg)[:, [0, 1]], georef)
                else:
                    sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
                # update the plotted shoreline
                sl_plot1[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
                sl_plot2[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
                sl_plot3[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])

                fig.canvas.draw_idle()
            else:
                ax4.set_title('sigma0 pixel intensities and threshold')
                break

        skip_image = False

        if ref:
            key_event = {}

            def press(event):
                # store what key was pressed in the dictionary
                key_event['pressed'] = event.key

            while True:
                btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                    transform=ax1.transAxes,
                                    bbox=dict(boxstyle="square", ec='k', fc='w'))
                btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                    transform=ax1.transAxes,
                                    bbox=dict(boxstyle="square", ec='k', fc='w'))
                btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                   transform=ax1.transAxes,
                                   bbox=dict(boxstyle="square", ec='k', fc='w'))
                plt.draw()
                fig.canvas.mpl_connect('key_press_event', press)
                plt.waitforbuttonpress()
                # after button is pressed, remove the buttons
                btn_skip.remove()
                btn_keep.remove()
                btn_esc.remove()

                # keep/skip image according to the pressed key, 'escape' to break the loop
                if key_event.get('pressed') == 'right':
                    skip_image = False
                    break
                elif key_event.get('pressed') == 'left':
                    skip_image = True
                    break
                elif key_event.get('pressed') == 'escape':
                    plt.close()
                    raise StopIteration('User cancelled checking shoreline detection')
                else:
                    plt.waitforbuttonpress()

        if not ref or not skip_image:

            jpeg_type = 'detection'
            jpeg_file_path = jpegFilePath(settings, jpeg_type, sat_name, date_start, data_end)
            fig.savefig(jpeg_file_path, dpi=150)

    return shoreline, t_sar, skip_image


def find_reference_shoreline(settings):

    median_dir_path = settings['median_dir_path']
    sat_name = settings['sat_name']
    site_name = settings['site_name']
    ref_date_start = date.fromisoformat(settings['date_range'][0])
    ref_date_end = date.fromisoformat(settings['date_range'][1])

    models_file_path = os.path.join(os.getcwd(), 'classification', 'models')

    # create a subfolder to store the .jpg images showing the detection
    jpeg_file_path = os.path.join(median_dir_path, 'jpg_files', 'detection')
    if not os.path.exists(jpeg_file_path):
        os.makedirs(jpeg_file_path)

    if sat_name == 'S1':

        sigma = settings['sigma']
        polarisation = settings['polarisation']
        if polarisation == 'VV':
            polarisation_band_index = 0
        elif polarisation == 'VH':
            polarisation_band_index = 1
        else:
            printError(f'select SAR band correctly: {polarisation}')

    elif sat_name == 'S2':

        classes = settings['classes']
        band_dict = settings['bands'][sat_name]
        first_key = next(iter(band_dict))
        pixel_size = band_dict[first_key][1]
        settings['pixel_size'] = pixel_size
        bands_20m = band_dict['20m'][0]
        for SWIR_index, SWIR_band in enumerate(bands_20m):
            if settings['SWIR'] == SWIR_band:
                break

        classifier = joblib.load(os.path.join(models_file_path, 'NN_6classes_S2.pkl'))
        min_beach_area_pixels = np.ceil(settings['min_beach_area'] / pixel_size ** 2)
        pansharpen = settings['pansharpen']

    else:

        classes = settings['classes']
        band_dict = settings['bands'][sat_name]
        first_key = next(iter(band_dict))
        pixel_size = band_dict[first_key][1]
        settings['pixel_size'] = pixel_size

        classifier = joblib.load(os.path.join(models_file_path, 'NN_4classes_Landsat.pkl'))
        min_beach_area_pixels = np.ceil(settings['min_beach_area'] / pixel_size ** 2)
        pansharpen = settings['pansharpen']


    with open(os.path.join(median_dir_path, site_name + '_metadata_' + sat_name + '.pkl'), 'rb') as f:
        metadata_dict = pickle.load(f)

    metadata_sat = metadata_dict[sat_name]
    file_names = metadata_sat['file_names']
    settings['image_epsg'] = int(metadata_sat['epsg'][0])

    printProgress('metadata loaded')

    # close all open figures
    plt.close('all')

    file_path_list = []
    for file_index, file_name in enumerate(file_names):

        date_start = metadata_sat['date_start'][file_index]
        date_end = metadata_sat['date_end'][file_index]
        meta_date_start = date.fromisoformat(date_start)
        meta_date_end = date.fromisoformat(date_end)

        printProgress(f'processing dates: ({meta_date_start}, {meta_date_end})')

        georef = None
        if meta_date_start >= ref_date_start and meta_date_end <= ref_date_end:

            settings['dates'] = [date_start, date_end]

            if sat_name == 'S1':
                file_path = os.path.join(median_dir_path, sat_name, file_name)
                file_path_list.append(file_path)

                sar_image, georef = NOC_preprocess.preprocess_sar(file_path)
                image_shape = (sar_image.shape[0], sar_image.shape[1], len(file_path_list))

            else:
                file_paths = []
                for band_key in band_dict:
                    file_paths.append(os.path.join(median_dir_path, sat_name,
                                                   band_key, file_name + '_' + band_key + '.tif'))
                file_path_list.append(file_paths)

                optical_image, georef = NOC_preprocess.preprocess_sar(file_paths[0])
                image_shape = (optical_image.shape[0], optical_image.shape[1],
                               optical_image.shape[2]+1, len(file_path_list))

    if not georef.any():
        printError(f'could not create reference shoreline possibly wrong date range or epsg code given')

    settings['georef'] = georef

    threshold_images = np.ndarray(image_shape)

    for file_index, file_paths in enumerate(file_path_list):

        if sat_name == 'S1':

            sar_image, _ = NOC_preprocess.preprocess_sar(file_paths)
            threshold_images[:, :, file_index] = sar_image[:,:,polarisation_band_index]

        elif sat_name == 'S2':

            printProgress(f'pansharpening SWIR - using {SWIR_band} - image {file_index+1}')
            optical_image, _, _, _, _, _ = \
                NOC_preprocess.preprocess_optical(settings, file_paths,
                                                  pansharpen=pansharpen,
                                                  SWIR_band=SWIR_band,
                                                  SWIR_index=SWIR_index,
                                                  ref=True)
            threshold_images[:, :, :, file_index] = optical_image

        else:

            ## NOT PROPERLY IMPLEMENTED YET

            optical_image, _, _, _, _, _ = \
                NOC_preprocess.preprocess_single(file_paths, sat_name, settings['cloud_mask_issue'])
            threshold_images[:, :, :, file_index] = optical_image


    # calculate a buffer around the reference shoreline if it has already been generated
    buffer_shape = (threshold_images.shape[0], threshold_images.shape[1])

    if sat_name == 'S1':
        image_ref_buffer = np.ones(buffer_shape, dtype=np.bool)
        threshold_images = gaussian(threshold_images, sigma=sigma, mode='reflect')

        reference_shoreline, reference_threshold, skip_image = \
                                    adjust_detection_sar(settings, threshold_images, image_ref_buffer,
                                                         ref=True)

    else:

        threshold_cloud_mask = np.zeros(buffer_shape, dtype=np.bool)

        image_ref_buffer = create_shoreline_buffer(settings, buffer_shape, ref=True)
        mndwi_buffer = create_mndwi_buffer(settings, buffer_shape)

        threshold_images = np.mean(threshold_images, axis=3)

        printProgress('classifying image')
        _, image_labels = NOC_classify.classify_image_NN(threshold_images, classes, threshold_cloud_mask,
                                                         min_beach_area_pixels, classifier)

        reference_shoreline, reference_threshold, skip_image = \
                    adjust_detection_optical(settings, threshold_images, threshold_cloud_mask,
                                             image_labels, image_ref_buffer, mndwi_buffer,
                                             ref=True)

    if not skip_image:

        reference_shoreline_file_name = pickleDumpName('reference_shoreline', site_name, sat_name)
        with open(os.path.join(median_dir_path, reference_shoreline_file_name), 'wb') as f:
            pickle.dump(reference_shoreline, f)

        if sat_name == 'S1':
            printSuccess(f'reference shoreline saved, reference threshold: {reference_threshold:3.2f}')
        else:
            printSuccess(f'reference shoreline saved, reference threshold: {reference_threshold:4.3f}')
    else:
        printWarning('reference shoreline not saved')

    # close figure window if still open
    if plt.get_fignums():
        plt.close()

    settings['reference_threshold'] = reference_threshold


def create_shoreline(settings, contours):

    image_epsg = settings['image_epsg']
    georef = settings['georef']

    # convert pixel coordinates to world coordinates
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = SDS_tools.convert_epsg(contours_world, image_epsg, settings['output_epsg'])
    # remove contours that have a perimeter < min_length_sl (provided in settings dict)
    # this enables to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k, 0], wl[k, 1]) for k in range(len(wl))]
        a = LineString(coords)  # shapely LineString structure
        if a.length >= settings['min_length_shoreline']:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points, contours_long[k][:, 0])
        y_points = np.append(y_points, contours_long[k][:, 1])
    shoreline = np.transpose(np.array([x_points, y_points]))

    return shoreline


def create_shoreline_buffer(settings, im_shape, ref=False):

    reference_shoreline = load_reference_shoreline(settings, ref=ref)
    settings['reference_shoreline'] = reference_shoreline

    image_epsg = int(settings['image_epsg'])
    output_epsg = int(settings['output_epsg'])
    georef = settings['georef']
    pixel_size = settings['pixel_size']

    if reference_shoreline.any():

        if ref:
            sat_name = 'S1'
        else:
            sat_name = settings['sat_name']
        printProgress(f'{sat_name} reference shoreline loaded')

        # convert reference shoreline to pixel coordinates
        ref_sl = settings['reference_shoreline']

        ref_sl_conv = SDS_tools.convert_epsg(ref_sl, output_epsg, image_epsg)[:, :-1]
        ref_sl_pix = SDS_tools.convert_world2pix(ref_sl_conv, georef)
        ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)

        # make sure that the pixel coordinates of the reference shoreline are inside the image
        idx_row = np.logical_and(ref_sl_pix_rounded[:,0] > 0, ref_sl_pix_rounded[:,0] < im_shape[1])
        idx_col = np.logical_and(ref_sl_pix_rounded[:,1] > 0, ref_sl_pix_rounded[:,1] < im_shape[0])
        idx_inside = np.logical_and(idx_row, idx_col)
        ref_sl_pix_rounded = ref_sl_pix_rounded[idx_inside,:]

        # create binary image of the reference shoreline (1 where the shoreline is 0 otherwise)
        im_binary = np.zeros(im_shape)
        for j in range(len(ref_sl_pix_rounded)):
            im_binary[ref_sl_pix_rounded[j,1], ref_sl_pix_rounded[j,0]] = 1
        im_binary = im_binary.astype(bool)

        # dilate the binary image to create a buffer around the reference shoreline
        max_dist_ref_pixels = np.ceil(settings['max_dist_ref']/pixel_size)
        se = morphology.disk(max_dist_ref_pixels)
        im_buffer = morphology.binary_dilation(im_binary, se)

    else:

        im_buffer = np.ones(im_shape).astype(bool)

    return im_buffer

def create_mndwi_buffer(settings, im_shape):

    reference_shoreline = load_reference_shoreline(settings)

    if reference_shoreline.any():

        # convert reference shoreline to pixel coordinates
        ref_sl = reference_shoreline
        image_epsg = int(settings['image_epsg'])
        output_epsg = int(settings['output_epsg'])
        georef = settings['georef']

        ref_sl_conv = SDS_tools.convert_epsg(ref_sl, output_epsg, image_epsg)[:, :-1]
        ref_sl_pix = SDS_tools.convert_world2pix(ref_sl_conv, georef)
        ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)

        # make sure that the pixel coordinates of the reference shoreline are inside the image
        idx_row = np.logical_and(ref_sl_pix_rounded[:,0] > 0, ref_sl_pix_rounded[:,0] < im_shape[1])
        idx_col = np.logical_and(ref_sl_pix_rounded[:,1] > 0, ref_sl_pix_rounded[:,1] < im_shape[0])
        idx_inside = np.logical_and(idx_row, idx_col)
        ref_sl_pix_rounded = ref_sl_pix_rounded[idx_inside,:]

        # create binary image of the reference shoreline (1 where the shoreline is 0 otherwise)
        im_binary = np.zeros(im_shape)
        for j in range(len(ref_sl_pix_rounded)):
            im_binary[ref_sl_pix_rounded[j,1], ref_sl_pix_rounded[j,0]] = 1
        im_binary = im_binary.astype(bool)

        # dilate the binary image to create a buffer around the reference shoreline
        max_dist_ref_pixels = 100
        se = morphology.disk(max_dist_ref_pixels)
        im_buffer = morphology.binary_dilation(im_binary, se)

    else:

        im_buffer = np.ones(im_shape).astype(bool)

    return im_buffer


def load_reference_shoreline(settings, ref=False):

    site_name = settings['site_name']
    if ref:
        sat_name = 'S1'
    else:
        sat_name = settings['sat_name']
    median_dir_path = settings['median_dir_path']

    # check if reference shoreline already exists in the corresponding folder
    ref_shoreline_file_name = site_name + '_reference_shoreline_' + sat_name +'.pkl'
    # if it exist, load it and return it
    if ref_shoreline_file_name in os.listdir(median_dir_path):

        with open(os.path.join(median_dir_path, ref_shoreline_file_name), 'rb') as f:
            ref_shoreline = pickle.load(f)
        reference_shoreline = ref_shoreline
    else:
        printWarning('no reference shoreline found')
        reference_shoreline = np.zeros(1)

    return reference_shoreline
